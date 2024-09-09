import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.ops import xywhn2xyxy

from object_detection.box_utils import iom


@dataclass
class Sample:
    name: str
    image: np.ndarray
    boxes: list
    ground_truth: list = None


@dataclass
class Args:
    border_size: int
    fold: int
    samples: str
    iom: float
    show_false_negatives: bool
    save_false_negatives: bool


# Set up argparse to parse command-line arguments
parser = argparse.ArgumentParser(description="Run object detection with YOLO and evaluate results.")
parser.add_argument('--border_size', type=int, default=0, help='Size of the border to add around images')
parser.add_argument('--fold', type=int, default=0, help='K-Fold cross-validation fold number to use')
parser.add_argument('--samples', choices=['positive', 'negative', 'both'], default='both',
                    help='Specify whether to include positive, negative, or both types of images')
parser.add_argument('--iom', type=float, default=0.0, help='Specify the IoM threshold for evaluation')
parser.add_argument('--show_false_negatives', action='store_true', help='Show plots of false negatives')
parser.add_argument('--save_false_negatives', action='store_true', help='Save names of false negatives')

args = Args(**vars(parser.parse_args()))

print("args:", args)


# Load the YOLO model from the given weights path
model = YOLO(Path(Path(__file__).parents[2], "runs", "detect", f"train2{args.fold}", "weights", "best.pt"))

# Set the fold for K-Fold cross-validation
KFOLD_FOLDER = Path(Path(__file__).parents[2], "data", "kfold_object_detection", f"fold{args.fold}")
VAL_IMAGES_FOLDER = Path(KFOLD_FOLDER, "images", "val")  # Validation images directory
VAL_LABELS_FOLDER = Path(KFOLD_FOLDER, "labels", "val")  # Validation labels directory


print("kfold folder:", KFOLD_FOLDER)

# Load the list of image files from the validation folder
image_files = os.listdir(VAL_IMAGES_FOLDER)

# Filter image files based on the include argument
if args.samples == 'positive':
    image_files = [i for i in image_files if "negative" not in i]
elif args.samples == 'negative':
    image_files = [i for i in image_files if "negative" in i]

# Calculate samples
total_samples = len(image_files)
positive_samples = sum(1 for i in image_files if "negative" not in i)
negative_samples = sum(1 for i in image_files if "negative" in i)

print("total samples:", total_samples)
print("positive samples:", positive_samples)
print("negative samples:", negative_samples)


# Initialize a dictionary to store images with added borders
images = {}

b_size = args.border_size  # Size of border to add around images from command-line argument
print("border size:", b_size)

# Load images and add borders
for i in tqdm(image_files, desc="loading images"):
    image = io.imread(Path(VAL_IMAGES_FOLDER, i))  # Read the image
    # Add a constant border around the image
    image = cv2.copyMakeBorder(image, b_size, b_size, b_size, b_size, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    images[i] = image  # Store the processed image

# Initialize variables to count true positives, false positives, and total boxes
true_positives = 0
false_positives = 0
total_boxes = 0


false_negative_samples: list[Sample] = []  # List to keep track of files with false negatives


samples: list[Sample] = []


# Run predictions on loaded images
for file, image in tqdm(images.items(), desc="running predictions"):
    fireball = file.split(".")[0]  # Extract the base filename to retrieve labels
    # Read the corresponding label file and convert xywh to xyxy format

    # Use the model to predict bounding boxes for the image
    results = model.predict(image, verbose=False, imgsz=416)
    boxes = results[0].boxes.xyxy.cpu()  # Extract predicted boxes

    total_boxes += len(boxes)  # Update total number of boxes

    sample = Sample(fireball, image, boxes)

    # Any boxes in negative tiles count as false positives
    if "negative" in fireball:
        false_positives += len(boxes)
        samples.append(sample)
        continue
    
    with open(Path(VAL_LABELS_FOLDER, fireball + ".txt")) as label_file:
        xyxy = xywhn2xyxy(
            np.array([float(i) for i in label_file.read().split(" ")[1:]]),  # Read and parse the label coordinates
            400,
            400
        )
    sample.ground_truth = xyxy

    # Check for false negatives (no predicted boxes)
    if len(boxes) == 0:
        false_negative_samples.append(sample)  # Add to false negatives if no boxes predicted
        samples.append(sample)
        continue

    ack_true_positive = False  # Flag to acknowledge true positive detection
    for box in boxes:
        # Check intersection between ground truth and predicted boxes
        if iom(xyxy, box) > args.iom:
            if not ack_true_positive:  # Count only one true positive for each image
                true_positives += 1
                ack_true_positive = True
        else:
            false_positives += 1  # Count as false positive if no intersection
    
    if not ack_true_positive:
        false_negative_samples.append(sample)
    
    samples.append(sample)


false_negatives = len(false_negative_samples)

# Print results of the evaluation
print("total boxes:", total_boxes)
print("true positives:", true_positives)
print("false positives:", false_positives)
print("false negatives:", false_negatives)
print("recall:", true_positives / positive_samples)
print("precision:", true_positives / (true_positives + false_positives))


if args.save_false_negatives:
    with open(Path(Path(__file__).parents[2], "data", "false_negatives.txt"), 'w') as file:
        file.write("\n".join([i.name for i in false_negative_samples]))


if args.show_false_negatives:
    for sample in false_negative_samples:
        fig, ax = plt.subplots(1)
        ax.imshow(sample.image)
        
        # Plot ground truth boxes in green
        if sample.ground_truth is not None:
            gt_box = sample.ground_truth
            rect = patches.Rectangle((gt_box[0], gt_box[1]), gt_box[2] - gt_box[0], gt_box[3] - gt_box[1], 
                                        linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
        
        # Plot predicted boxes in red
        for pred_box in sample.boxes:
            rect = patches.Rectangle((pred_box[0], pred_box[1]), pred_box[2] - pred_box[0], pred_box[3] - pred_box[1], 
                                    linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        
        plt.title(sample.name)
        plt.show()