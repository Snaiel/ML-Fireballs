import argparse
import gc
import multiprocessing as mp
import os
import shutil
import signal
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import io
from tqdm import tqdm
from ultralytics import YOLO

from fireball_detection.detect import detect_fireballs, plot_boxes
from object_detection.dataset import DATA_FOLDER, GFO_JPEGS, GFO_PICKINGS
from object_detection.dataset.kfold import retrieve_fireball_splits
from object_detection.dataset.point_pickings import PointPickings
from queue import Full, Empty


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


def run_tests(queue: mp.Queue, bar_queue: mp.Queue, detected_boxes: list, preds: list, split: int, border_size: int) -> None:
    
    model_path = Path(DATA_FOLDER, "kfold_runs", f"split{split}", "weights", "last.pt")
    model = YOLO(model_path)

    try:
        while True:
            fireball_file = queue.get(False)
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


def test(split: int, num_processes: int, border_size: int) -> None:
    matplotlib.use("agg")

    print(f"testing split{split}...")
    model_path = Path(DATA_FOLDER, "kfold_runs", f"split{split}", "weights", "last.pt")
    print(f"using model from path \"{model_path}\"")

    split_folder = get_split_folder(split)

    detected_boxes = os.listdir(Path(split_folder, "boxes"))
    preds = os.listdir(Path(split_folder, "preds"))

    fireball_files = os.listdir(Path(split_folder, "images"))
    queue = mp.Queue()
    for fireball_file in fireball_files:
        queue.put_nowait(fireball_file)
    
    print("setting up processes...")

    bar_queue = mp.Queue()
    bar_process = mp.Process(target=update_bar, args=(bar_queue, len(fireball_files)), daemon=True)
    bar_process.start()

    processes: list[mp.Process] = []

    for _ in range(num_processes):
        process = mp.Process(target=run_tests, args=(queue, bar_queue, detected_boxes, preds, split, border_size))
        processes.append(process)
        process.start()

    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        queue.close()
        bar_queue.close()
        for process in processes:
            process.terminate()
            process.join()
        os.kill(os.getpid(), signal.SIGTERM)


def test_all(processes: int, border_size: int) -> None:
    for split in range(5):
        p = mp.Process(target=test, args=(split, processes, border_size))
        p.start()
        p.join()


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
    parser = argparse.ArgumentParser(
        description="A script to test detections on full images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command")

    # Create subcommand
    parser_create = subparsers.add_parser('create', help="Create the test")
    parser_create.set_defaults(func=lambda _: create())

    # Test subcommand
    parser_test = subparsers.add_parser('test', help="Run test")
    parser_test.add_argument('--split', type=int, required=True, help="Specify the split number")
    parser_test.add_argument('--processes', type=int, required=True, help="Number of processes to use as workers")
    parser_test.add_argument('--border_size', type=int, required=True, help="Specify the border size")
    parser_test.set_defaults(func=lambda args: test(args.split, args.processes, args.border_size))

    # Test_all subcommand
    parser_test_all = subparsers.add_parser('test_all', help="Run test on all folds")
    parser_test_all.add_argument('--processes', type=int, required=True, help="Number of processes to use as workers")
    parser_test_all.add_argument('--border_size', type=int, required=True, help="Specify the border size")
    parser_test_all.set_defaults(func=lambda args: test_all(args.processes, args.border_size))

    args = parser.parse_args()

    # Call the function associated with the command
    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()