import argparse
import gc
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from skimage import io
from ultralytics import YOLO

from fireball_detection.detect import detect_fireballs, plot_boxes
from object_detection.dataset import DATA_FOLDER, GFO_JPEGS, GFO_PICKINGS
from object_detection.dataset.point_pickings import PointPickings
from object_detection.dataset.kfold import get_train_val_test_split

THUMB_TEST_SET_FOLDER = Path(DATA_FOLDER, "thumb_test_set")

THUMB_TEST_IMAGES_FOLDER = Path(THUMB_TEST_SET_FOLDER, "images")
THUMB_TEST_PP_BB_FOLDER = Path(THUMB_TEST_SET_FOLDER, "pp_bb")
THUMB_TEST_BOXES_FOLDER = Path(THUMB_TEST_SET_FOLDER, "boxes")
THUMB_TEST_PREDS_FOLDER = Path(THUMB_TEST_SET_FOLDER, "preds")


def get_iou(pred_box, gt_box):
    """
    https://github.com/Treesfive/calculate-iou/blob/master/get_iou.py

    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou


def test_fireball(model: YOLO, fireball_file: str, detected_boxes: list, preds: list) -> None:
    fireball_name = fireball_file.split(".")[0]
        
    if fireball_name + ".txt" in detected_boxes and fireball_name + ".jpg" in preds:
        print(f"{fireball_name} already detected")
        return

    print(f"detecting {fireball_name}")

    image = io.imread(Path(THUMB_TEST_IMAGES_FOLDER, fireball_file))
    fireballs = detect_fireballs(image, model)

    with open(Path(THUMB_TEST_BOXES_FOLDER, fireball_name + ".txt"), "w") as boxes_file:
        lines = []
        for fireball in fireballs:
            print(fireball_name, fireball)
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
    
    fig.savefig(Path(THUMB_TEST_PREDS_FOLDER, fireball_name + ".jpg"), bbox_inches='tight', pad_inches=0, dpi=600)
    
    plt.cla()
    plt.clf()
    plt.close('all')
    del fig, ax, pp, image, fireballs
    gc.collect()


def run_tests(queue: mp.Queue, detected_boxes: list, preds: list) -> None:
    model = YOLO(Path(Path(__file__).parents[1], "data", "e15.pt"))
    while True:
        try:
            fireball_file = queue.get(False)
            test_fireball(model, fireball_file, detected_boxes, preds)
        except:
            break
        

def create():
    if Path(THUMB_TEST_SET_FOLDER).exists():
        shutil.rmtree(THUMB_TEST_SET_FOLDER)
    os.mkdir(THUMB_TEST_SET_FOLDER)
    os.mkdir(THUMB_TEST_IMAGES_FOLDER)
    os.mkdir(THUMB_TEST_PP_BB_FOLDER)
    os.mkdir(THUMB_TEST_BOXES_FOLDER)
    os.mkdir(THUMB_TEST_PREDS_FOLDER)
    
    fireball_dataset = get_train_val_test_split(6555)
    for fireball_file in fireball_dataset["test"]:
        fireball_name = fireball_file.split(".")[0]
        shutil.copyfile(Path(GFO_JPEGS, fireball_file), Path(THUMB_TEST_IMAGES_FOLDER, fireball_file))
        pp = PointPickings(Path(GFO_PICKINGS, fireball_name + ".csv"))
        with open(Path(THUMB_TEST_PP_BB_FOLDER, fireball_name + ".txt"), "w") as pp_bb_file:
            pp_bb_file.write(f"{pp.bb_min_x} {pp.bb_min_y} {pp.bb_max_x} {pp.bb_max_y}")


def test():
    matplotlib.use("agg")
    
    detected_boxes = os.listdir(THUMB_TEST_BOXES_FOLDER)
    preds = os.listdir(THUMB_TEST_PREDS_FOLDER)

    fireball_files = os.listdir(THUMB_TEST_IMAGES_FOLDER)
    manager = mp.Manager()
    queue = manager.Queue()
    for fireball_file in fireball_files:
        queue.put_nowait(fireball_file)
    
    processes = 10
    
    pool = mp.Pool(processes)
    tests = []
    for _ in range(processes):
        tests.append(pool.apply_async(run_tests, args=(queue, detected_boxes, preds)))

    for test in tests:
        test.get()


def main():
    parser = argparse.ArgumentParser(description="A script to test detections on full images using the thumb test set.")
    subparsers = parser.add_subparsers(dest="command")

    # Create subcommand
    parser_create = subparsers.add_parser('create', help="Create the test")
    parser_create.set_defaults(func=create)

    # Test subcommand
    parser_test = subparsers.add_parser('test', help="Run test")
    parser_test.set_defaults(func=test)

    args = parser.parse_args()

    # Call the function associated with the command
    if args.command:
        args.func()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()