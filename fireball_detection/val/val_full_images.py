import argparse
import gc
import multiprocessing as mp
import os
import shutil
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
from object_detection.dataset import DATA_FOLDER, GFO_JPEGS, GFO_PICKINGS
from object_detection.dataset.create_kfold_dataset import \
    retrieve_fireball_splits
from object_detection.dataset.point_pickings import PointPickings


SENTINEL = None

KFOLD_FIREBALL_DETECTION_FOLDER = Path(DATA_FOLDER, "kfold_fireball_detection")

discard_fireballs = {
    "24_2015-03-18_140528_DSC_0352" # image requires two bounding boxes
}


def get_split_folder(split: int) -> Path:
    return Path(KFOLD_FIREBALL_DETECTION_FOLDER, f"split{split}")


def test_fireball(model: YOLO, fireball_file: str, detected_boxes: list, preds: list, split: int, border_size: int) -> None:
    split_folder = get_split_folder(split)
    fireball_name = fireball_file.split(".")[0]
        
    if fireball_name + ".txt" in detected_boxes and fireball_name + ".jpg" in preds:
        # print(f"{fireball_name} already detected")
        return

    # print(f"detecting {fireball_name}")

    image = io.imread(Path(split_folder, "images", fireball_file))
    
    fireballs = detect_fireballs(image, model, border_size)
    
    with open(Path(split_folder, "boxes", fireball_name + ".txt"), "w") as boxes_file:
        lines = []
        for fireball in fireballs:
            # print(fireball_name, fireball)
            lines.append(repr(fireball))
        boxes_file.write("\n".join(lines))

    fig, ax = plot_boxes(image, fireballs)

    pp = PointPickings(Path(GFO_PICKINGS, fireball_name + ".csv"))
    ax.add_patch(
        Rectangle(
            (pp.bb_min_x, pp.bb_min_y),
            pp.bb_max_x - pp.bb_min_x,
            pp.bb_max_y - pp.bb_min_y,
            linewidth=1,
            edgecolor='lime',
            facecolor='none'
        )
    )
    
    fig.savefig(Path(split_folder, "preds", fireball_name + ".jpg"), bbox_inches='tight', pad_inches=0, dpi=600)
    
    plt.cla()
    plt.clf()
    plt.close('all')
    del fig, ax, pp, image, fireballs
    gc.collect()


def run_tests(
        fireball_queue: mp.Queue, 
        bar_queue: mp.Queue,
        detected_boxes: list,
        preds: list,
        split: int,
        yolo_pt_path: str,
        border_size: int
    ) -> None:
    
    model = YOLO(Path(yolo_pt_path))

    try:
        while True:
            fireball_file = fireball_queue.get()
            if fireball_file is SENTINEL:
                break
            test_fireball(model, fireball_file, detected_boxes, preds, split, border_size)
            bar_queue.put_nowait(1)
    except (Full, Empty) as e:
        print(type(e))
        return


def update_bar(bar_queue: mp.Queue, total: int) -> None:
    pbar = tqdm(total=total, desc="testing on full images")
    while True:
        bar_queue.get(True)
        pbar.update(1)


def test(split: int, yolo_pt_path: str, num_processes: int, border_size: int) -> None:
    matplotlib.use("agg")

    if not Path(KFOLD_FIREBALL_DETECTION_FOLDER).exists():
        print(f"{KFOLD_FIREBALL_DETECTION_FOLDER.relative_to(DATA_FOLDER.parent)} does not exist. Run\n")
        print(f"python3 -m fireball_detection.val.val_full_images create")
        return

    print(f"\ntesting split{split}...")

    split_folder = get_split_folder(split)

    detected_boxes = os.listdir(Path(split_folder, "boxes"))
    preds = os.listdir(Path(split_folder, "preds"))

    fireball_files = os.listdir(Path(split_folder, "images"))
    fireball_queue = mp.Queue()
    for fireball_file in fireball_files:
        fireball_queue.put_nowait(fireball_file)
    
    for _ in range(num_processes):
        fireball_queue.put(SENTINEL)
    
    print("setting up processes...")

    bar_queue = mp.Queue()
    bar_process = mp.Process(target=update_bar, args=(bar_queue, len(fireball_files)), daemon=True)
    bar_process.start()

    processes: list[mp.Process] = []

    for _ in range(num_processes):
        process = mp.Process(target=run_tests, args=(fireball_queue, bar_queue, detected_boxes, preds, split, yolo_pt_path, border_size))
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


def create():
    files, splits = retrieve_fireball_splits()

    if Path(KFOLD_FIREBALL_DETECTION_FOLDER).exists():
        shutil.rmtree(KFOLD_FIREBALL_DETECTION_FOLDER)
    os.mkdir(KFOLD_FIREBALL_DETECTION_FOLDER)

    for split, (_, test_indexes) in splits:
        split_folder = get_split_folder(split)
        os.mkdir(split_folder)
        for sub_folder in ("images", "pp_bb", "boxes", "preds"):
            os.mkdir(Path(split_folder, sub_folder))
    
        for fireball_file in tqdm([files[i] for i in test_indexes], desc=f"split{split} samples"):
            fireball_name = fireball_file.split(".")[0]
            if fireball_name in discard_fireballs:
                continue
            shutil.copyfile(Path(GFO_JPEGS, fireball_file), Path(split_folder, "images", fireball_file))
            pp = PointPickings(Path(GFO_PICKINGS, fireball_name + ".csv"))
            with open(Path(split_folder, "pp_bb", fireball_name + ".txt"), "w") as pp_bb_file:
                pp_bb_file.write(f"{pp.bb_min_x} {pp.bb_min_y} {pp.bb_max_x} {pp.bb_max_y}")


def main():
    @dataclass
    class Args:
        command: str
        split: int | None = None
        yolo_pt_path: str | None = None
        processes: int | None = None
        border_size: int | None = None

    parser = argparse.ArgumentParser(
        description="A script to test detections on full images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command")

    # Create subcommand
    subparsers.add_parser('create', help="Create the test")

    # Test subcommand
    parser_test = subparsers.add_parser('test', help="Run test")
    parser_test.add_argument('--split', type=int, required=True, help="Specify the split number")
    parser_test.add_argument('--yolo_pt_path', type=str, required=True, help='Path to the YOLO model .pt file')
    parser_test.add_argument('--processes', type=int, required=True, help="Number of processes to use as workers")
    parser_test.add_argument('--border_size', type=int, required=True, help="Specify the border size")

    args = Args(**vars(parser.parse_args()))

    # Call the function associated with the command
    if args.command == "create":
        create()
    elif args.command == "test":
        test(args.split, args.processes, args.border_size)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()