import gc
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from skimage import io

from fireball_detection.detect import detect_fireballs, plot_boxes
from object_detection.dataset import DATA_FOLDER, GFO_JPEGS, GFO_PICKINGS
from object_detection.dataset.point_pickings import PointPickings
from object_detection.dataset.utils import get_train_val_test_split

THUMB_TEST_SET_FOLDER = Path(DATA_FOLDER, "thumb_test_set")

detected_boxes = os.listdir(Path(THUMB_TEST_SET_FOLDER, "boxes"))
preds = os.listdir(Path(THUMB_TEST_SET_FOLDER, "preds"))


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


def test_fireball(fireball_file: str) -> None:
    fireball_name = fireball_file.split(".")[0]
        
    if fireball_name + ".txt" in detected_boxes and fireball_name + ".jpg" in preds:
        print(f"{fireball_name} already detected")
        return

    print(f"detecting {fireball_name}")

    image = io.imread(Path(THUMB_TEST_SET_FOLDER, "images", fireball_file))
    boxes = detect_fireballs(image)

    pp = PointPickings(Path(GFO_PICKINGS, fireball_name + ".csv"))
    
    for box in boxes:
        box.append(get_iou(box, (pp.bb_min_x, pp.bb_min_y, pp.bb_max_x, pp.bb_max_y)))

    with open(Path(THUMB_TEST_SET_FOLDER, "boxes", fireball_name + ".txt"), "w") as boxes_file:
        lines = []
        for box in boxes:
            print(fireball_name, box)
            lines.append(" ".join(map(str, box)))
        boxes_file.write("\n".join(lines))

    fig, ax = plot_boxes(image, boxes)

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

    fig.savefig(Path(THUMB_TEST_SET_FOLDER, "preds", fireball_name + ".jpg"), bbox_inches='tight', pad_inches=0, dpi=600)
    
    plt.cla()
    plt.clf()
    del fig, ax, pp, image, boxes
    gc.collect()


def run_tests(queue: mp.Queue) -> None:
    while not queue.empty():
        fireball_file = queue.get()
        test_fireball(fireball_file)
        


def main() -> None:
    if Path(THUMB_TEST_SET_FOLDER).exists():
        shutil.rmtree(THUMB_TEST_SET_FOLDER)
    os.mkdir(THUMB_TEST_SET_FOLDER)
    os.mkdir(Path(THUMB_TEST_SET_FOLDER, "images"))
    os.mkdir(Path(THUMB_TEST_SET_FOLDER, "pp_bb"))
    os.mkdir(Path(THUMB_TEST_SET_FOLDER, "boxes"))
    os.mkdir(Path(THUMB_TEST_SET_FOLDER, "preds"))
    
    fireball_dataset = get_train_val_test_split(6555)
    for fireball_file in fireball_dataset["test"]:
        fireball_name = fireball_file.split(".")[0]
        shutil.copyfile(Path(GFO_JPEGS, fireball_file), Path(THUMB_TEST_SET_FOLDER, "images", fireball_file))
        pp = PointPickings(Path(GFO_PICKINGS, fireball_name + ".csv"))
        with open(Path(THUMB_TEST_SET_FOLDER, "pp_bb", fireball_name + ".txt"), "w") as pp_bb_file:
            pp_bb_file.write(f"{pp.bb_min_x} {pp.bb_min_y} {pp.bb_max_x} {pp.bb_max_y}")

    fireball_files = os.listdir(Path(THUMB_TEST_SET_FOLDER, "images"))
    manager = mp.Manager()
    queue = manager.Queue()
    for fireball_file in fireball_files:
        queue.put_nowait(fireball_file)
    
    processes = 10
    
    pool = mp.Pool(processes)
    tests = []
    for _ in range(processes):
        tests.append(pool.apply_async(run_tests, args=(queue,)))

    for test in tests:
        test.get()


if __name__ == "__main__":
    main()