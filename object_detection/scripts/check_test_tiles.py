import os
from pathlib import Path

import cv2
import numpy as np
import skimage.io as io
from ultralytics import YOLO
from ultralytics.utils.ops import xywhn2xyxy

from fireball_detection.detect import intersect

from tqdm import tqdm


model = YOLO(Path(Path(__file__).parents[2], "runs", "detect", "train22", "weights", "best.pt"))

VAL_IMAGES_FOLDER = Path(Path(__file__).parents[2], "data", "kfold_object_detection", "fold0", "images", "val")
VAL_LABELS_FOLDER = Path(Path(__file__).parents[2], "data", "kfold_object_detection", "fold0", "labels", "val")

image_files = os.listdir(VAL_IMAGES_FOLDER)
image_files = [i for i in image_files if not "negative" in i]
print("total positive samples:", len(image_files))

images = {}


for i in tqdm(image_files, desc="loading images"):
    image = io.imread(Path(VAL_IMAGES_FOLDER, i))
    # image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    images[i] = image

true_positives = 0
false_positives = 0
total_boxes = 0

false_negative_files = []

for file, image in tqdm(images.items(), desc="running predictions"):
    fireball = file.split(".")[0]
    with open(Path(VAL_LABELS_FOLDER, fireball + ".txt")) as label_file:
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

# with open(Path(Path(__file__).parents[2], "data", "false_negatives.txt"), 'w') as file:
#     file.write("\n".join(false_negative_files))