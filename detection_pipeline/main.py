import argparse
import json
import multiprocessing as mp
import os
import shutil
import signal
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full

from skimage import io
from tqdm import tqdm
from ultralytics import YOLO

from fireball_detection.detect import detect_fireballs
import structlog


structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.CallsiteParameterAdder(parameters=[
            structlog.processors.CallsiteParameter.FUNC_NAME,
            structlog.processors.CallsiteParameter.MODULE
        ]),
        structlog.processors.JSONRenderer(),

    ]
)

logger: structlog.stdlib.BoundLogger = structlog.get_logger()
logger.info("test")


@dataclass
class Args:
    folder_path: str
    model_path: str
    processes: int
    verbose: bool
    plot: bool


SENTINEL = None


def detect(model: YOLO, folder_path: Path, output_folder: Path, fireball_file: str) -> None:

    fireball_name = fireball_file.split(".")[0]
    image_path = Path(folder_path, fireball_file)
    image = io.imread(image_path)
    fireballs = detect_fireballs(image, model)

    if not fireballs:
        return

    fireball_folder = Path(output_folder, fireball_name)
    os.mkdir(fireball_folder)

    shutil.copy(image_path, fireball_folder)

    for f in fireballs:
        coords = list(map(int, f.box))
        x1, y1, x2, y2 = coords
        cropped_image = image[y1:y2, x1:x2]
        thumb_path = Path(fireball_folder, f"{fireball_name}_{int(f.conf * 100)}_{'-'.join(map(str, coords))}.jpg")
        io.imsave(thumb_path, cropped_image)

    detections = [vars(fireball) for fireball in fireballs]
    output_data = {"detections": detections}

    output_json = json.dumps(output_data, indent=4)

    with open(Path(fireball_folder, fireball_name + ".json"), 'w') as json_file:
        json_file.write(output_json)


def run_detections(
        fireball_queue: mp.Queue, 
        bar_queue: mp.Queue,
        model_path: Path,
        folder_path: Path,
        output_folder: Path
    ) -> None:
    
    model = YOLO(Path(model_path), task="detect")

    try:
        while True:
            fireball_file = fireball_queue.get()
            if fireball_file is SENTINEL:
                break
            detect(model, folder_path, output_folder, fireball_file)
            bar_queue.put_nowait(1)
    except (Full, Empty) as e:
        print(type(e))
        return


def update_bar(bar_queue: mp.Queue, total: int) -> None:
    pbar = tqdm(total=total, desc="testing on full images")
    while True:
        bar_queue.get(True)
        pbar.update(1)


def main():
    parser = argparse.ArgumentParser(
        description="Detect fireballs from images in a folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the fireball image file.")
    parser.add_argument("--model_path", type=str, help="Path to the YOLO model file.")
    parser.add_argument('--processes', type=int, default=8, help="Number of processes to use as workers")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--plot", action="store_true", help="Plot and display the bounding boxes on the image.")
    
    args = Args(**vars(parser.parse_args()))

    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    folder_path = Path(args.folder_path)
    if not folder_path.exists() or folder_path.is_file():
        print(f"{folder_path} is not a valid folder.")
        return

    output_folder = Path(folder_path, folder_path.name)
    if not output_folder.exists():
        os.mkdir(output_folder)

    fireball_images = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]

    fireball_queue = mp.Queue()
    for fireball_image in fireball_images:
        fireball_queue.put_nowait(fireball_image)
    
    for _ in range(args.processes):
        fireball_queue.put(SENTINEL)

    bar_queue = mp.Queue()
    bar_process = mp.Process(target=update_bar, args=(bar_queue, len(fireball_images)), daemon=True)
    bar_process.start()

    processes: list[mp.Process] = []

    for _ in range(args.processes):
        process = mp.Process(
            target=run_detections, 
            args=(
                fireball_queue,
                bar_queue,
                Path(args.model_path),
                folder_path,
                output_folder
            )
        )
        processes.append(process)
        process.start()

    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        fireball_queue.close()
        bar_queue.close()
        for process in processes:
            process.terminate()
            process.join()
        os.kill(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    pass
    # main()