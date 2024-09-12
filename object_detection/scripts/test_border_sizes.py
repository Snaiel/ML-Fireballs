import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.ops import xywhn2xyxy

from fireball_detection.detect import intersects
from object_detection.utils import add_border


parser = argparse.ArgumentParser(description='For a given fold, test recall of border sizes 0-32 inclusive.')
parser.add_argument('--fold', type=int, required=True, help='The fold number to use for K-Fold cross-validation (0, 1, 2, 3, 4)')
args = parser.parse_args()


# Load the YOLO model from the given weights path
model = YOLO(Path(Path(__file__).parents[2], "runs", "detect", f"train2{args.fold}", "weights", "best.pt"))

# Set the fold for K-Fold cross-validation
KFOLD_FOLDER = Path(Path(__file__).parents[2], "data", "kfold_object_detection", f"fold{args.fold}")
VAL_IMAGES_FOLDER = Path(KFOLD_FOLDER, "images", "val")  # Validation images directory
VAL_LABELS_FOLDER = Path(KFOLD_FOLDER, "labels", "val")  # Validation labels directory

print("kfold folder:", KFOLD_FOLDER)

# Load the list of image files from the validation folder, excluding negatives
image_files = os.listdir(VAL_IMAGES_FOLDER)
image_files = [i for i in image_files if not "negative" in i]

# Load all images into a dictionary
images = {}
for i in tqdm(image_files, desc="loading images"):
    image = io.imread(Path(VAL_IMAGES_FOLDER, i))  # Read the image
    images[i] = image  # Store the processed image

border_sizes = range(32)  # Border sizes from 0 to 32 inclusive
recalls = []  # To store recall values for each border size

# Process each border size
for b_size in tqdm(border_sizes, desc="evaluating each border size", position=0):
    # Initialize variables to count true positives, false positives, and total boxes
    true_positives = 0
    false_positives = 0
    total_boxes = 0

    false_negative_files = []  # List to keep track of files with false negatives

    # Run predictions on images with the current border size
    for file, image in tqdm(images.items(), desc=f"border size {b_size}", position=1):
        # Add a constant border around the image
        bordered_image = add_border(image, b_size)

        fireball = file.split(".")[0]  # Extract the base filename to retrieve labels
        # Read the corresponding label file and convert xywh to xyxy format
        with open(Path(VAL_LABELS_FOLDER, fireball + ".txt")) as label_file:
            xyxy = xywhn2xyxy(
                np.array([float(i) for i in label_file.read().split(" ")[1:]]),
                400,
                400
            )

        # Use the model to predict bounding boxes for the bordered image
        results = model.predict(bordered_image, verbose=False, imgsz=416)
        boxes = results[0].boxes.xyxy.cpu()  # Extract predicted boxes

        total_boxes += len(boxes)  # Update total number of boxes

        # Check for false negatives (no predicted boxes)
        if len(boxes) == 0:
            false_negative_files.append(fireball)  # Add to false negatives if no boxes predicted

        ack_true_positive = False  # Flag to acknowledge true positive detection
        for box in boxes:
            # Check intersection between ground truth and predicted boxes
            if intersects(xyxy, box):
                if not ack_true_positive:  # Count only one true positive for each image
                    true_positives += 1
                    ack_true_positive = True
            else:
                false_positives += 1  # Count as false positive if no intersection

    # Calculate recall and store in recalls list
    recall = true_positives / len(image_files) if len(image_files) > 0 else 0
    recalls.append(recall)
    print(f"Border size: {b_size}, Recall: {recall}")

print("recalls:")
for i in recalls:
    print(i)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(border_sizes, recalls, marker='o')
plt.title("Recall vs Border Size")
plt.xlabel("Border Size")
plt.ylabel("Recall")
plt.xticks(border_sizes)
plt.grid()
plt.show()
