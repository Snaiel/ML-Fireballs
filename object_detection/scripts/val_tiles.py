import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.ops import xywhn2xyxy

from fireball_detection.detect import intersects
from object_detection.utils import add_border, iom, iou


discard_fireballs = {
    "24_2015-03-18_140528_DSC_0352" # image requires two bounding boxes
}


def main():
    @dataclass
    class Sample:
        name: str
        image: np.ndarray
        boxes: list
        ground_truth: list = None


    @dataclass
    class Args:
        border_size: int
        split: int
        samples: str
        metric: str
        threshold: float | None
        show_false_negatives: bool
        save_false_negatives: bool


    # Set up argparse to parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run object detection with YOLO and evaluate results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--split', type=int, required=True, help='K-Fold cross-validation split number to use')
    parser.add_argument('--border_size', type=int, default=0, help='Size of the border to add around images')
    parser.add_argument('--samples', choices=['positive', 'negative', 'both'], default='both',
                        help='Specify whether to include positive, negative, or both types of images')
    parser.add_argument('--metric', type=str, choices=['iom', 'iou', 'intersects'], required=True, help='Metric to be used')
    parser.add_argument('--threshold', type=float, help='Threshold value between 0.0 and 1.0')
    parser.add_argument('--show_false_negatives', action='store_true', help='Show plots of false negatives')
    parser.add_argument('--save_false_negatives', action='store_true', help='Save names of false negatives')

    args = Args(**vars(parser.parse_args()))

    if args.metric in ['iom', 'iou']:
        if args.threshold is None:
            parser.error(f"--threshold is required when --metric is '{args.metric}'")
        elif not (0.0 <= args.threshold <= 1.0):
            raise ValueError('Threshold must be between 0.0 and 1.0')
    elif args.metric == 'intersects' and args.threshold is not None:
        parser.error("--threshold should not be provided when --metric is 'intersects'")

    print("args:", vars(args))
    print()


    # Load the YOLO model from the given weights path
    model = YOLO(Path(Path(__file__).parents[2], "runs", "detect", f"train2{args.split}", "weights", "best.pt"))

    # Set the split for K-Fold cross-validation
    KFOLD_FOLDER = Path(Path(__file__).parents[2], "data", "kfold_object_detection", f"split{args.split}")
    VAL_IMAGES_FOLDER = Path(KFOLD_FOLDER, "images", "val")  # Validation images directory
    VAL_LABELS_FOLDER = Path(KFOLD_FOLDER, "labels", "val")  # Validation labels directory


    print("kfold folder:", KFOLD_FOLDER)
    print()

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

    print(f"{'Total samples:':<30} {total_samples}")
    print(f"{'Positive samples:':<30} {positive_samples}")
    print(f"{'Negative samples:':<30} {negative_samples}")
    print()


    # Initialize a dictionary to store images with added borders
    images = {}

    b_size = args.border_size  # Size of border to add around images from command-line argument

    # Load images and add borders
    for i in tqdm(image_files, desc="loading images"):
        image = io.imread(Path(VAL_IMAGES_FOLDER, i))  # Read the image
        # Add a constant border around the image
        image = add_border(image, b_size)
        images[i] = image  # Store the processed image

    # Initialize variables to count true positives, false positives, and total boxes
    detected_samples = 0
    total_boxes = 0
    true_positives = 0


    false_negative_samples: list[Sample] = []  # List to keep track of files with false negatives


    samples: list[Sample] = []


    fireball_names = set()
    detected_fireball_names = set()


    # Run predictions on loaded images
    for file, image in tqdm(images.items(), desc="running predictions"):
        fireball = file.split(".")[0]  # Extract the base filename to retrieve labels
        # Read the corresponding label file and convert xywh to xyxy format

        fireball_name = "_".join(fireball.split("_")[:5])
        ignore_for_total_fireball_detection = False
        if fireball_name in discard_fireballs:
            ignore_for_total_fireball_detection = True
        else:
            fireball_names.add(fireball_name)

        # Use the model to predict bounding boxes for the image
        results = model.predict(image, verbose=False, imgsz=416)
        boxes = results[0].boxes.xyxy.cpu()  # Extract predicted boxes

        total_boxes += len(boxes)  # Update total number of boxes

        sample = Sample(fireball, image, boxes)

        # Any boxes in negative tiles count as false positives
        if "negative" in fireball:
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

        ack_true_positive = False
        for box in boxes:
            if (args.metric == "intersects" and intersects(xyxy, box)) or \
                (args.metric == "iom" and iom(xyxy, box) >= args.threshold) or \
                (args.metric == "iou" and iou(xyxy, box) >= args.threshold):
                true_positives += 1
                if not ack_true_positive:
                    detected_samples += 1
                    ack_true_positive = True
        
        if ack_true_positive:
            if not ignore_for_total_fireball_detection:
                detected_fireball_names.add(fireball_name)
        else:
            false_negative_samples.append(sample)
        
        samples.append(sample)


    print()
    print(f"{'Detected samples:':<30} {detected_samples}")
    print(f"{'False negatives:':<30} {positive_samples - detected_samples}")
    print(f"{'Recall on individual samples:':<30} {detected_samples / positive_samples:.5f}")
    print()
    print(f"{'Total fireballs:':<30} {len(fireball_names)}")
    print(f"{'Detected fireballs:':<30} {len(detected_fireball_names)}")
    print(f"{'False negatives:':<30} {len(fireball_names) - len(detected_fireball_names)}")
    print(f"{'Recall on entire fireballs:':<30} {len(detected_fireball_names) / len(fireball_names):.5f}")
    print()
    print(f"{'Total boxes:':<30} {total_boxes}")
    print(f"{'True positives:':<30} {true_positives}")
    print(f"{'False positives:':<30} {total_boxes - true_positives}")
    print(f"{'Precision:':<30} {true_positives / total_boxes:.5f}")



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


if __name__ == "__main__":
    main()