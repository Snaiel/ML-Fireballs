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


@dataclass
class Args:
    val_folder_name: str = None
    model_path: str = None
    processes: int = None
    border_size: int = None
    num_samples: int | float = None


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


    pp_bb_path = Path(val_folder, "pp_bb", fireball_name + ".txt")
    
    if pp_bb_path.exists():
        with open(pp_bb_path) as file:
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
        model_path: str,
        border_size: int
    ) -> None:
    
    model = YOLO(Path(model_path), task="detect")

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


def test(args: Args) -> None:
    matplotlib.use("agg")

    if not Path(VAL_FIREBALL_DETECTION_FOLDER).exists():
        print(f"{VAL_FIREBALL_DETECTION_FOLDER.relative_to(DATA_FOLDER.parent)} does not exist. Run\n")
        print(f"python3 -m fireball_detection.val.val_full_images create")
        return

    print(f"testing {args.val_folder_name}...\n")

    val_folder = Path(VAL_FIREBALL_DETECTION_FOLDER, args.val_folder_name)

    full_fireball_files = os.listdir(Path(val_folder, "images"))


    num_samples = args.num_samples
    if isinstance(num_samples, float):
        # Calculate number of samples based on proportion
        num_samples = max(1, int(len(full_fireball_files) * num_samples))

    # Ensure the number of samples does not exceed available files
    num_samples = min(num_samples, len(full_fireball_files))

    # Select the subset of fireball files
    fireball_files = full_fireball_files[:num_samples]


    fireball_queue = mp.Queue()
    for fireball_file in fireball_files:
        fireball_queue.put_nowait(fireball_file)
    
    for _ in range(args.processes):
        fireball_queue.put(SENTINEL)

    bar_queue = mp.Queue()
    bar_process = mp.Process(target=update_bar, args=(bar_queue, len(fireball_files)), daemon=True)
    bar_process.start()

    processes: list[mp.Process] = []

    for _ in range(args.processes):
        process = mp.Process(target=run_tests, args=(fireball_queue, bar_queue, val_folder, args.model_path, args.border_size))
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


def _parse_num_samples(value):
    try:
        # Try to interpret as an integer
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError(f"'{value}' must be non-negative.")
        return ivalue
    except ValueError:
        try:
            # Try to interpret as a float
            fvalue = float(value)
            if not (0.0 < fvalue <= 1.0):
                raise argparse.ArgumentTypeError(f"'{value}' must be a positive float between 0 and 1 if it is a proportion.")
            return fvalue
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer or float value.")


def main() -> None:

    parser = argparse.ArgumentParser(
        description="A script to test detections on full images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--val_folder_name', type=str, choices=os.listdir(VAL_FIREBALL_DETECTION_FOLDER), required=True, help="Specify folder to run on.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the YOLO model (.pt .onnx .engine) file')
    parser.add_argument('--processes', type=int, default=8, help="Number of processes to use as workers")
    parser.add_argument('--border_size', type=int, default=5, help="Specify the border size")
    parser.add_argument('--num_samples', type=_parse_num_samples, default=1.0, help="Specify the number of samples to verify. Integer for number, float (0-1] for proportion.")

    args = Args(**vars(parser.parse_args()))

    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    test(args)


if __name__ == "__main__":
    main()