import os
import shutil
from math import sqrt
from pathlib import Path

from fireball_detection.detect import intersect
from fireball_detection.thumb_test_set import (THUMB_TEST_BOXES_FOLDER,
                                               THUMB_TEST_PP_BB_FOLDER,
                                               THUMB_TEST_PREDS_FOLDER,
                                               THUMB_TEST_SET_FOLDER)

FALSE_NEGATIVES_FOLDER = Path(THUMB_TEST_SET_FOLDER, "false_negatives")
LONG_FALSE_NEGATIVES_FOLDER = Path(FALSE_NEGATIVES_FOLDER, "long")
SMALL_FALSE_NEGATIVES_FOLDER = Path(FALSE_NEGATIVES_FOLDER, "small")


total_fireballs = len(os.listdir(THUMB_TEST_BOXES_FOLDER))
total_boxes = 0
true_positives = 0
false_negatives = 0

long_total_fireballs = 0
long_total_boxes = 0
long_true_positives = 0
long_false_negatives = 0

small_total_fireballs = 0
small_total_boxes = 0
small_true_positives = 0
small_false_negatives = 0


if os.path.isdir(FALSE_NEGATIVES_FOLDER):
    shutil.rmtree(FALSE_NEGATIVES_FOLDER)
os.mkdir(FALSE_NEGATIVES_FOLDER)
os.mkdir(LONG_FALSE_NEGATIVES_FOLDER)
os.mkdir(SMALL_FALSE_NEGATIVES_FOLDER)


for fireball_file in os.listdir(THUMB_TEST_BOXES_FOLDER):
    fireball_name = fireball_file.split(".")[0]
    long = False

    boxes = []
    with open(Path(THUMB_TEST_BOXES_FOLDER, fireball_file)) as file:
        lines = file.readlines()
        for line in lines:
            boxes.append([float(x) for x in line.split(" ")[:4]])
    
    pp_bb = []
    with open(Path(THUMB_TEST_PP_BB_FOLDER, fireball_file)) as file:
        pp_bb = [float(x) for x in file.readline().split(" ")]
        length = sqrt((pp_bb[2] - pp_bb[0])**2 + (pp_bb[3] - pp_bb[1])**2)
        if length >= 400:
            long = True
    
    total_boxes += len(boxes)
    if long:
        long_total_fireballs += 1
        long_total_boxes += len(boxes)
    else:
        small_total_fireballs += 1
        small_total_boxes += len(boxes)

    intersects = False
    for box in boxes:
        if intersect(box, pp_bb):
            intersects = True
            break

    if intersects:
        true_positives += 1
        if long:
            long_true_positives += 1
        else:
            small_true_positives += 1
    else:
        false_negatives += 1
        if long:
            long_false_negatives += 1
            shutil.copy(Path(THUMB_TEST_PREDS_FOLDER, fireball_name + ".jpg"), LONG_FALSE_NEGATIVES_FOLDER)
        else:
            small_false_negatives += 1
            shutil.copy(Path(THUMB_TEST_PREDS_FOLDER, fireball_name + ".jpg"), SMALL_FALSE_NEGATIVES_FOLDER)


print(total_fireballs)
print(total_boxes)
print(true_positives)
print(false_negatives)
print()
print(long_total_fireballs)
print(long_total_boxes)
print(long_true_positives)
print(long_false_negatives)
print()
print(small_total_fireballs)
print(small_total_boxes)
print(small_true_positives)
print(small_false_negatives)