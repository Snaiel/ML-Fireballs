import argparse
import gc
import json
import multiprocessing as mp
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import io
from tqdm import tqdm
from ultralytics import YOLO

from fireball_detection.detect import detect_fireballs, plot_boxes
from fireball_detection.val import VAL_FIREBALL_DETECTION_FOLDER
from object_detection.dataset import DATA_FOLDER


SENTINEL = None


def test_fireball(val_folder: Path, model: YOLO, fireball_file: str, detected_boxes: list, preds: list, border_size: int) -> None:
    fireball_name = fireball_file.split(".")[0]
        
    if fireball_name + ".txt" in detected_boxes and fireball_name + ".jpg" in preds:
        # print(f"{fireball_name} already detected")
        return

    # print(f"detecting {fireball_name}")

    image = io.imread(Path(val_folder, "images", fireball_file))
    
    fireballs = detect_fireballs(image, model, border_size)
    
    with open(Path(val_folder, "boxes", fireball_name + ".txt"), "w") as boxes_file:
        lines = []
        for fireball in fireballs:
            # print(fireball_name, fireball)
            lines.append(repr(fireball))
        boxes_file.write("\n".join(lines))

    fig, ax = plot_boxes(image, fireballs)

    with open(Path(val_folder, "pp_bb", fireball_name + ".txt")) as file:
        pp_bb = [float(x) for x in file.readline().split(" ")]

    ax.add_patch(
        Rectangle(
            (pp_bb[0], pp_bb[1]),
            pp_bb[2] - pp_bb[0],
            pp_bb[3] - pp_bb[1],
            linewidth=1,
            edgecolor='lime',
            facecolor='none'
        )
    )
    
    fig.savefig(Path(val_folder, "preds", fireball_name + ".jpg"), bbox_inches='tight', pad_inches=0, dpi=600)
    
    plt.cla()
    plt.clf()
    plt.close('all')
    del fig, ax, image, fireballs
    gc.collect()


def run_tests(
        fireball_queue: mp.Queue, 
        bar_queue: mp.Queue,
        val_folder: Path,
        yolo_pt_path: str,
        border_size: int
    ) -> None:
    
    model = YOLO(Path(yolo_pt_path))

    detected_boxes = os.listdir(Path(val_folder, "boxes"))
    preds = os.listdir(Path(val_folder, "preds"))

    try:
        while True:
            fireball_file = fireball_queue.get()
            if fireball_file is SENTINEL:
                break
            test_fireball(val_folder, model, fireball_file, detected_boxes, preds, border_size)
            bar_queue.put_nowait(1)
    except (Full, Empty) as e:
        print(type(e))
        return


def update_bar(bar_queue: mp.Queue, total: int) -> None:
    pbar = tqdm(total=total, desc="testing on full images")
    while True:
        bar_queue.get(True)
        pbar.update(1)


def test(val_folder_name: str, yolo_pt_path: str, num_processes: int, border_size: int) -> None:
    matplotlib.use("agg")

    if not Path(VAL_FIREBALL_DETECTION_FOLDER).exists():
        print(f"{VAL_FIREBALL_DETECTION_FOLDER.relative_to(DATA_FOLDER.parent)} does not exist. Run\n")
        print(f"python3 -m fireball_detection.val.val_full_images create")
        return

    print(f"testing {val_folder_name}...\n")

    val_folder = Path(VAL_FIREBALL_DETECTION_FOLDER, val_folder_name)

    fireball_files = os.listdir(Path(val_folder, "images"))
    fireball_queue = mp.Queue()
    for fireball_file in fireball_files:
        fireball_queue.put_nowait(fireball_file)
    
    for _ in range(num_processes):
        fireball_queue.put(SENTINEL)

    bar_queue = mp.Queue()
    bar_process = mp.Process(target=update_bar, args=(bar_queue, len(fireball_files)), daemon=True)
    bar_process.start()

    processes: list[mp.Process] = []

    for _ in range(num_processes):
        process = mp.Process(target=run_tests, args=(fireball_queue, bar_queue, val_folder, yolo_pt_path, border_size))
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


def main() -> None:
    @dataclass
    class Args:
        val_folder_name: str = None
        yolo_pt_path: str = None
        processes: int = None
        border_size: int = None

    parser = argparse.ArgumentParser(
        description="A script to test detections on full images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--val_folder_name', type=str, choices=os.listdir(VAL_FIREBALL_DETECTION_FOLDER), required=True, help="Specify folder to run on.")
    parser.add_argument('--yolo_pt_path', type=str, required=True, help='Path to the YOLO model .pt file')
    parser.add_argument('--processes', type=int, default=8, help="Number of processes to use as workers")
    parser.add_argument('--border_size', type=int, default=5, help="Specify the border size")

    args = Args(**vars(parser.parse_args()))

    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    test(args.val_folder_name, args.yolo_pt_path, args.processes, args.border_size)


if __name__ == "__main__":
    main()