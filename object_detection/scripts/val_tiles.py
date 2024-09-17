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
    "24_2015-03-18_140528_DSC_0352" # image requires two bounding boxes. logic below assumes one fireball per image.
}


@dataclass
class Args:
    command: str
    border_size: int
    split: int | None
    samples: str
    metric: str
    threshold: float | None
    show_false_negatives: bool
    save_false_negatives: bool


@dataclass
class Sample:
    name: str
    image: np.ndarray
    boxes: list
    ground_truth: list = None


def analyse_split(args: Args) -> dict:
    
    model = YOLO(Path(Path(__file__).parents[2], "runs", "detect", f"train2{args.split}", "weights", "last.pt"))

    KFOLD_FOLDER = Path(Path(__file__).parents[2], "data", "kfold_object_detection", f"split{args.split}")
    VAL_IMAGES_FOLDER = Path(KFOLD_FOLDER, "images", "val")
    VAL_LABELS_FOLDER = Path(KFOLD_FOLDER, "labels", "val")


    print("kfold folder:", KFOLD_FOLDER)
    print()


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

    # Load images and add borders
    for i in tqdm(image_files, desc="loading images"):
        image = io.imread(Path(VAL_IMAGES_FOLDER, i))
        # Add a constant border around the image
        image = add_border(image, args.border_size)
        images[i] = image


    detected_samples = 0
    total_boxes = 0
    true_positives = 0

    false_negative_samples_list: list[Sample] = []
    samples: list[Sample] = []

    fireball_names = set()
    detected_fireball_names = set()


    # Run predictions
    for file, image in tqdm(images.items(), desc="running predictions"):
        fireball = file.split(".")[0]

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
            false_negative_samples_list.append(sample)  # Add to false negatives if no boxes predicted
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
            false_negative_samples_list.append(sample)
        
        samples.append(sample)


    false_negative_samples = positive_samples - detected_samples
    recall_individual_samples = detected_samples / positive_samples

    total_fireballs = len(fireball_names)
    detected_fireballs = len(detected_fireball_names)
    false_negative_fireballs = len(fireball_names) - len(detected_fireball_names)
    recall_entire_fireballs = len(detected_fireball_names) / len(fireball_names)
    
    false_positives = total_boxes - true_positives
    precision = true_positives / total_boxes


    return {
        "total_samples": total_samples,
        "positive_samples": positive_samples,
        "negative_samples": negative_samples,
        "detected_samples": detected_samples,
        "false_negative_samples_list": false_negative_samples_list,
        "false_negative_samples": false_negative_samples,
        "recall_individual_samples": recall_individual_samples,
        "total_fireballs": total_fireballs,
        "detected_fireballs": detected_fireballs,
        "false_negative_fireballs": false_negative_fireballs,
        "recall_entire_fireballs": recall_entire_fireballs,
        "total_boxes": total_boxes,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "precision": precision
    }



def analyse_all_splits(args: Args) -> dict:
    splits_stats: list[dict] = []

    for i in range(5):
        args.split = i
        splits_stats.append(analyse_split(args))

    stats = {}

    for stat in splits_stats[0].keys():
        if isinstance(splits_stats[i][stat], list):
            continue

        total = 0
        for i in range(5):
            total += splits_stats[i][stat] 
        
        stats[stat] = total / 5
    
    return stats


def output_stats(args: Args, stats: dict) -> None:
    detected_samples = stats["detected_samples"]
    false_negative_samples = stats["false_negative_samples"]
    recall_individual_samples = stats["recall_individual_samples"]
    total_fireballs = stats["total_fireballs"]
    detected_fireballs = stats["detected_fireballs"]
    false_negative_fireballs = stats["false_negative_fireballs"]
    recall_entire_fireballs = stats["recall_entire_fireballs"]
    total_boxes = stats["total_boxes"]
    true_positives = stats["true_positives"]
    false_positives = stats["false_positives"]
    precision = stats["precision"]

    print()
    print(f"{'Detected samples:':<30} {detected_samples}")
    print(f"{'False negative samples:':<30} {false_negative_samples}")
    print(f"{'Recall on individual samples:':<30} {recall_individual_samples:.5f}")
    print()
    print(f"{'Total fireballs:':<30} {total_fireballs}")
    print(f"{'Detected fireballs:':<30} {detected_fireballs}")
    print(f"{'False negative fireballs:':<30} {false_negative_fireballs}")
    print(f"{'Recall on entire fireballs:':<30} {recall_entire_fireballs:.5f}")
    print()
    print(f"{'Total boxes:':<30} {total_boxes}")
    print(f"{'True positives:':<30} {true_positives}")
    print(f"{'False positives:':<30} {false_positives}")
    print(f"{'Precision:':<30} {precision:.5f}")

    if args.command == "val_all_splits":
        return

    false_negative_samples_list = stats["false_negative_samples_list"]

    if args.save_false_negatives:
        with open(Path(Path(__file__).parents[2], "data", "false_negatives.txt"), 'w') as file:
            file.write("\n".join([i.name for i in false_negative_samples_list]))

    if args.show_false_negatives:
        for sample in false_negative_samples_list:
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


def main():

    # Set up argparse to parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run object detection with YOLO and evaluate results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('command', choices=['val_split', 'val_all_splits'], help="Command to execute")
    parser.add_argument('--split', type=int, help='K-Fold cross-validation split number to use')
    parser.add_argument('--border_size', type=int, default=0, help='Size of the border to add around images')
    parser.add_argument('--samples', choices=['positive', 'negative', 'both'], default='both',
                        help='Specify whether to include positive, negative, or both types of images')
    parser.add_argument('--metric', type=str, choices=['iom', 'iou', 'intersects'], required=True, help='Metric to be used')
    parser.add_argument('--threshold', type=float, help='Threshold value between 0.0 and 1.0')
    parser.add_argument('--show_false_negatives', action='store_true', help='Show plots of false negatives')
    parser.add_argument('--save_false_negatives', action='store_true', help='Save names of false negatives')

    args = Args(**vars(parser.parse_args()))

    if args.command == 'val_split':
        if args.split is None:
            parser.error("--split is required when the command is 'val_split'")
    if args.command == 'val_all_splits':
        if args.split is not None:
            parser.error("--split should not be provided when the command is 'val_all_splits'")


    if args.metric in ['iom', 'iou']:
        if args.threshold is None:
            parser.error(f"--threshold is required when --metric is '{args.metric}'")
        elif not (0.0 <= args.threshold <= 1.0):
            raise ValueError('Threshold must be between 0.0 and 1.0')
    elif args.metric == 'intersects' and args.threshold is not None:
        parser.error("--threshold should not be provided when --metric is 'intersects'")

    print("args:", vars(args))
    print()

    if args.command == 'val_split':
        output_stats(args, analyse_split(args))
    elif args.command == 'val_all_splits':
        output_stats(args, analyse_all_splits(args))
    else:
        parser.error("Invalid command")


if __name__ == "__main__":
    main()