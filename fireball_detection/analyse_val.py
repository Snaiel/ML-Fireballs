import argparse
import os
import shutil
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fireball_detection.detect import intersects
from fireball_detection.val_full_images import get_split_folder
from object_detection.utils import iom, iou


def diagonal_length(box) -> float:
    """
    Calculate the diagonal length of a rectangular box.

    Args:
        box (tuple): A tuple of four elements representing the coordinates of the box 
                     in the format (x1, y1, x2, y2).

    Returns:
        float: The length of the diagonal of the box.
    """
    from math import sqrt
    return sqrt((box[2] - box[0])**2 + (box[3] - box[1])**2)



def main():
    @dataclass
    class Args:
        split: int
        metric: str
        threshold: float

    parser = argparse.ArgumentParser(description='Fireball detection analysis')
    parser.add_argument('--split', type=int, required=True, help='Fold number for the analysis')
    parser.add_argument('--metric', type=str, choices=['iom', 'iou', 'intersects'], required=True, help='Metric to be used')
    parser.add_argument('--threshold', type=float, help='Threshold value between 0.0 and 1.0')
    
    args = Args(**vars(parser.parse_args()))

    if args.metric in ['iom', 'iou']:
        if args.threshold is None:
            parser.error(f"--threshold is required when --metric is '{args.metric}'")
        elif not (0.0 <= args.threshold <= 1.0):
            raise ValueError('Threshold must be between 0.0 and 1.0')
    elif args.metric == 'intersects' and args.threshold is not None:
        parser.error("--threshold should not be provided when --metric is 'intersects'")
    

    SPLIT_FOLDER = get_split_folder(args.split)

    FALSE_NEGATIVES_FOLDER = Path(SPLIT_FOLDER, "false_negatives")
    LONG_FALSE_NEGATIVES_FOLDER = Path(FALSE_NEGATIVES_FOLDER, "long")
    SMALL_FALSE_NEGATIVES_FOLDER = Path(FALSE_NEGATIVES_FOLDER, "small")

    BOXES_FOLDER = Path(SPLIT_FOLDER, "boxes")
    PP_BB_FOLDER = Path(SPLIT_FOLDER, "pp_bb")
    PREDS_FOLDER = Path(SPLIT_FOLDER, "preds")

    total_fireballs = len(os.listdir(BOXES_FOLDER))
    fireballs_detected = 0
    long_fireballs = 0
    long_fireballs_detected = 0
    small_fireballs = 0
    small_fireballs_detected = 0

    total_boxes = 0
    true_positive_boxes = 0
    long_boxes = 0
    long_true_positive_boxes = 0
    small_boxes = 0
    small_true_positive_boxes = 0

    boxes_in_each_file = []

    true_positive_box_sizes = []
    false_positive_box_sizes = []

    true_positive_conf_box_size = []
    false_positive_conf_box_size = []

    if os.path.isdir(FALSE_NEGATIVES_FOLDER):
        shutil.rmtree(FALSE_NEGATIVES_FOLDER)
    os.mkdir(FALSE_NEGATIVES_FOLDER)
    os.mkdir(LONG_FALSE_NEGATIVES_FOLDER)
    os.mkdir(SMALL_FALSE_NEGATIVES_FOLDER)


    for fireball_file in os.listdir(BOXES_FOLDER):
        fireball_name = fireball_file.split(".")[0]
        long = False

        boxes = []
        with open(Path(BOXES_FOLDER, fireball_file)) as file:
            lines = file.readlines()
            for line in lines:
                line = [float(x) for x in line.split(" ")]
                boxes.append((line[0], tuple(line[1:])))
        
        pp_bb = []
        with open(Path(PP_BB_FOLDER, fireball_file)) as file:
            pp_bb = [float(x) for x in file.readline().split(" ")]
            if diagonal_length(pp_bb) >= 400:
                long = True
        
        boxes_in_each_file.append(len(boxes))

        if long:
            long_fireballs += 1
        else:
            small_fireballs += 1

        total_boxes += len(boxes)

        count_as_positive = False
        for box in boxes:
            box_diagonal = diagonal_length(box[1])

            long_box = box_diagonal >= 400
            if long_box:
                long_boxes += 1
            else:
                small_boxes += 1

            if (args.metric == "intersects" and intersects(box[1], pp_bb)) or \
            (args.metric == "iom" and iom(box[1], pp_bb) >= args.threshold) or \
            (args.metric == "iou" and iou(box[1], pp_bb) >= args.threshold):
                count_as_positive = True
                true_positive_boxes += 1
                true_positive_box_sizes.append(box_diagonal)
                true_positive_conf_box_size.append((box[0], box_diagonal))
                if long_box:
                    long_true_positive_boxes += 1
                else:
                    small_true_positive_boxes += 1
            else:
                false_positive_box_sizes.append(box_diagonal)
                false_positive_conf_box_size.append((box[0], box_diagonal))
                

        if count_as_positive:
            fireballs_detected += 1
            if long:
                long_fireballs_detected += 1
            else:
                small_fireballs_detected += 1
        else:
            if long:
                shutil.copy(Path(PREDS_FOLDER, fireball_name + ".jpg"), LONG_FALSE_NEGATIVES_FOLDER)
            else:
                shutil.copy(Path(PREDS_FOLDER, fireball_name + ".jpg"), SMALL_FALSE_NEGATIVES_FOLDER)


    ## Recall

    # Overall
    print("Based on Ground Truth Fireballs")
    print(f"{'Total fireballs:':<25} {total_fireballs}")
    print(f"{'Fireballs detected:':<25} {fireballs_detected}")
    false_negatives = total_fireballs - fireballs_detected
    print(f"{'False negatives:':<25} {false_negatives}")

    recall = fireballs_detected /total_fireballs if total_fireballs > 0 else 0
    print(f"{'Overall Recall:':<25} {recall:.5f}")
    print()

    # Long
    print(f"{'Long total fireballs:':<25} {long_fireballs}")
    print(f"{'Long fireballs detected:':<25} {long_fireballs_detected}")
    long_false_negatives = long_fireballs - long_fireballs_detected
    print(f"{'Long false negatives:':<25} {long_false_negatives}")

    long_recall = long_fireballs_detected / long_fireballs if long_fireballs > 0 else 0
    print(f"{'Long Recall:':<25} {long_recall:.5f}")
    print()

    # Small
    print(f"{'Small total fireballs:':<25} {small_fireballs}")
    print(f"{'Small fireballs detected:':<25} {small_fireballs_detected}")
    small_false_negatives = small_fireballs - small_fireballs_detected
    print(f"{'Small false negatives:':<25} {small_false_negatives}")

    small_recall = small_fireballs_detected / small_fireballs if small_fireballs > 0 else 0
    print(f"{'Small Recall:':<25} {small_recall:.5f}")

    print()
    print()

    ## Precision

    # Overall
    print("Based on Predicted Boxes")
    print(f"{'Total boxes:':<25} {total_boxes}")
    print(f"{'True positives:':<25} {true_positive_boxes}")
    false_positives = total_boxes - true_positive_boxes
    print(f"{'False positives:':<25} {false_positives}")
    box_precision = true_positive_boxes / total_boxes if total_boxes > 0 else 0
    print(f"{'Overall Precision:':<25} {box_precision:.6f}")
    
    print()

    # Long
    print(f"{'Long total boxes:':<25} {long_boxes}")
    print(f"{'Long true positives:':<25} {long_true_positive_boxes}")
    long_false_positives = long_boxes - long_true_positive_boxes
    print(f"{'Long false positives:':<25} {long_false_positives}")
    long_precision = long_true_positive_boxes / long_boxes if long_boxes > 0 else 0
    print(f"{'Long Precision:':<25} {long_precision:.5f}")
    
    print()

    # Small
    print(f"{'Small total boxes:':<25} {small_boxes}")
    print(f"{'Small true positives:':<25} {small_true_positive_boxes}")
    small_false_positives = small_boxes - small_true_positive_boxes
    print(f"{'Small false positives:':<25} {small_false_positives}")
    small_precision = small_true_positive_boxes / small_boxes if small_boxes > 0 else 0
    print(f"{'Small Precision:':<25} {small_precision:.5f}")


    ## Distribution of Boxes in Each File
    
    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(boxes_in_each_file, bins=range(25), alpha=0.7, color='blue')

    # Add titles and labels
    plt.title('Distribution of Boxes in Each File')
    plt.xlabel('Number of Boxes')
    plt.ylabel('Frequency')

    # Show the plot
    plt.grid(True)
    plt.show()

    # Create the histogram
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # Plot the histogram for true positive box sizes
    axes[0].hist(true_positive_box_sizes, bins=np.linspace(0, 1000, 51), alpha=0.7, color='blue')
    axes[0].set_title('Distribution of True Positive Box Sizes')
    axes[0].set_xlabel('Diagonal Length')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True)

    # Plot the histogram for false positive box sizes
    axes[1].hist(false_positive_box_sizes, bins=np.linspace(0, 1000, 51), alpha=0.7, color='red')
    axes[1].set_title('Distribution of False Positive Box Sizes')
    axes[1].set_xlabel('Diagonal Length')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True)

    # Show the plots
    plt.tight_layout()
    plt.show()

    # Unpack data
    tp_conf, tp_box_size = zip(*true_positive_conf_box_size)
    fp_conf, fp_box_size = zip(*false_positive_conf_box_size)

    # Plotting
    plt.scatter(tp_conf, tp_box_size, color='blue', label='True Positive')
    plt.scatter(fp_conf, fp_box_size, color='orange', label='False Positive')

    # Adding labels and title
    plt.xlabel('Confidence')
    plt.ylabel('Box Size')
    plt.title('True Positive and False Positive Dot Plot')
    plt.legend()

    # Show plot
    plt.show()



if __name__ == "__main__":
    main()