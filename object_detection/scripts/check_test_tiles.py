import os
from pathlib import Path

import cv2
import numpy as np
import skimage.io as io
from ultralytics import YOLO
from ultralytics.utils.ops import xywhn2xyxy

from fireball_detection.detect import intersect


model = YOLO(Path(Path(__file__).parents[2], "data", "e15.pt"))

TEST_IMAGES_FOLDER = Path(Path(__file__).parents[2], "data", "object_detection", "images", "test")
TEST_LABELS_FOLDER = Path(Path(__file__).parents[2], "data", "object_detection", "labels", "test")

image_files = os.listdir(TEST_IMAGES_FOLDER)
image_files = [i for i in image_files if not "negative" in i]
print("total positive samples:", len(image_files))

images = {}


for i in image_files:
    image = io.imread(Path(TEST_IMAGES_FOLDER, i))
    image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    images[i] = image

true_positives = 0
false_positives = 0
total_boxes = 0

false_negative_files = []

for file, image in images.items():
    fireball = file.split(".")[0]
    with open(Path(TEST_LABELS_FOLDER, fireball + ".txt")) as label_file:
        xyxy = xywhn2xyxy(
            np.array([float(i) for i in label_file.read().split(" ")[1:]]), #xywh
            400,
            400
        )

    results = model.predict(image, verbose=False, imgsz=416)
    boxes = results[0].boxes.xyxy.cpu()

    total_boxes += len(boxes)

    if len(boxes) == 0:
        false_negative_files.append(fireball)

    ack_true_positive = False
    for box in boxes:
        if intersect(xyxy, box):
            if not ack_true_positive:
                true_positives += 1
                ack_true_positive = True
        else:
            false_positives += 1

print("total boxes:", total_boxes)
print("true positives:", true_positives)
print("false positives:", false_positives)

with open(Path(Path(__file__).parents[2], "data", "false_negatives.txt"), 'w') as file:
    file.write("\n".join(false_negative_files))