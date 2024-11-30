import argparse
import json
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

from fireball_detection.boxes.merge import intersects
from object_detection.dataset import DATA_FOLDER
from object_detection.utils import add_border, iom, iou


discard_fireballs = {
    "24_2015-03-18_140528_DSC_0352" # image requires two bounding boxes. logic below assumes one fireball per image.
}


@dataclass
class Args:
    border_size: int
    data_yaml_path: str | None
    yolo_pt_path: str | None
    samples: str
    metric: str
    threshold: float | None
    show_false_negatives: bool
    save_false_negatives: bool


@dataclass
class FireballSample:
    name: str
    image: np.ndarray
    boxes: list
    ground_truth: list = None


def val_split(args: Args) -> dict:
    
    model_path = Path(args.yolo_pt_path)
    print(model_path, "\n")

    model = YOLO(model_path)

    data_path = Path(args.data_yaml_path)
    val_images_folder = Path(data_path.parent, "images", "val")
    val_labels_folder = Path(data_path.parent, "labels", "val")


    print("kfold folder:", data_path, "\n")


    image_files = os.listdir(val_images_folder)

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


    detected_samples = 0
    total_boxes = 0
    true_positives = 0

    false_negative_samples_list: list[FireballSample] = []

    fireball_names = set()
    detected_fireball_names = set()


    for image_file in tqdm(image_files, desc="running predictions"):
        image = io.imread(Path(val_images_folder, image_file))
        image = add_border(image, args.border_size)

        fireball = image_file.split(".")[0]

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

        sample = FireballSample(fireball, image, boxes)

        # Any boxes in negative tiles count as false positives
        if "negative" in fireball:
            continue
        
        with open(Path(val_labels_folder, fireball + ".txt")) as label_file:
            xyxy = xywhn2xyxy(
                np.array([float(i) for i in label_file.read().split(" ")[1:]]),  # Read and parse the label coordinates
                400,
                400
            )
        sample.ground_truth = xyxy

        # Check for false negatives (no predicted boxes)
        if len(boxes) == 0:
            false_negative_samples_list.append(sample)  # Add to false negatives if no boxes predicted
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

    if not args.data_yaml_path:
        return

    false_negative_samples_list: list[FireballSample] = stats["false_negative_samples_list"]

    if args.save_false_negatives:
        with open(Path(Path(args.data_yaml_path).parent, "false_negatives.txt"), 'w') as file:
            file.write("\n".join([i.name for i in false_negative_samples_list]))

    if args.show_false_negatives:
        for sample in false_negative_samples_list:
            _, ax = plt.subplots(1)
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
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Run object detection with YOLO and evaluate results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Define command-line arguments
    parser.add_argument('--yolo_pt_path', type=str, help='Path to the YOLO model .pt file')
    parser.add_argument('--data_yaml_path', type=str, help='Path to the data YAML file')
    parser.add_argument('--border_size', type=int, default=5, help='Size of the border to add around images')
    parser.add_argument('--samples', choices=['positive', 'negative', 'both'], default='both',
                        help='Specify whether to include positive, negative, or both types of images')
    parser.add_argument('--metric', type=str, choices=['iom', 'iou', 'intersects'], required=True, help='Metric to be used')
    parser.add_argument('--threshold', type=float, help='Threshold value between 0.0 and 1.0')
    parser.add_argument('--show_false_negatives', action='store_true', help='Show plots of false negatives')
    parser.add_argument('--save_false_negatives', action='store_true', help='Save names of false negatives')

    # Parse the arguments
    args = Args(**vars(parser.parse_args()))

    # Validate metric and threshold arguments
    if args.metric in ['iom', 'iou']:
        if args.threshold is None:
            parser.error(f"--threshold is required when --metric is '{args.metric}'")
        elif not (0.0 <= args.threshold <= 1.0):
            raise ValueError('Threshold must be between 0.0 and 1.0')
    elif args.metric == 'intersects' and args.threshold is not None:
        parser.error("--threshold should not be provided when --metric is 'intersects'")

    # Validate the yolo_pt_path and data_yaml_path arguments
    if bool(args.yolo_pt_path) != bool(args.data_yaml_path):
        parser.error("Both --yolo_pt_path and --data_yaml_path must be specified together, or neither.")

    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    # If yolo_pt_path and data_yaml_path are provided, run evaluation directly
    if args.yolo_pt_path and args.data_yaml_path:
        output_stats(args, val_split(args))
        return

    # Dataclass to represent a dataset with optional data_yaml and yolo_model paths
    @dataclass
    class Dataset:
        data_yaml: Path | None = None
        yolo_model: Path | None = None

    datasets: list[Dataset] = []

    # Collect all available datasets from the DATA_FOLDER directory
    for od_folder in sorted([i for i in os.listdir(DATA_FOLDER) if i.startswith("object_detection")]):
        for folder in os.listdir(Path(DATA_FOLDER, od_folder)):
            dataset = Dataset()
            
            # Check for the existence of a data YAML file and YOLO model in each folder
            data_yaml_file = Path(DATA_FOLDER, od_folder, folder, "data.yaml")
            if data_yaml_file.exists():
                dataset.data_yaml = data_yaml_file
                datasets.append(dataset)
            
            yolo_model = Path(DATA_FOLDER, od_folder, folder, "yolo.pt")
            if yolo_model.exists():
                dataset.yolo_model = yolo_model

    # Display available dataset options
    print("Dataset options:\n")

    # NOTE: The folder "all" is used as a placeholder for when the user
    # chooses to evaluate against all the splits, not the folder itself.

    for i, dataset in enumerate(datasets):
        if dataset.data_yaml.parent.name == "all":
            od_folder = dataset.data_yaml.parents[1]
            print(f"  [{i+1}] all splits in {od_folder.relative_to(DATA_FOLDER)}")

            # Display information about the splits
            splits = sorted(
                [
                    d for d in datasets if any(parent == od_folder for parent in d.data_yaml.parents) and
                    d != dataset
                ],
                key=lambda d: str(d.data_yaml)
            )

            for split in splits:
                if split.yolo_model:
                    print(f"        ✅ {split.yolo_model.relative_to(od_folder)}")
                else:
                    print(f"        ❌ no yolo.pt model for {split.data_yaml.parent.relative_to(od_folder)}")

            print()
            continue

        # Display information for datasets with individual YAML and model files
        print(f"  [{i+1}] {dataset.data_yaml.relative_to(DATA_FOLDER)}")
        if dataset.yolo_model:
            print(f"        ✅ {dataset.yolo_model.relative_to(DATA_FOLDER)}")
        else:
            print(f"        ❌ no yolo.pt model found.")
        print()

    # Prompt the user to select a dataset for validation
    while True:
        try:
            user_input = int(input("Enter which dataset to validate: "))

            # Based on the user's choice, get the corresponding dataset
            dataset = datasets[user_input-1]

            if dataset.data_yaml.parent.name == "all":
                od_folder = dataset.data_yaml.parents[1]

                # Ensure all splits have the YOLO model files
                splits = sorted(
                    [
                        d for d in datasets if any(parent == od_folder for parent in d.data_yaml.parents) and
                        d != dataset
                    ],
                    key=lambda d: str(d.data_yaml)
                )

                all_models_present = True

                for split in splits:
                    if not split.yolo_model:
                        print(f"All splits yolo.pt models must be present.\n")
                        all_models_present = False
                        break
                
                if not all_models_present:
                    continue

            elif not (dataset.data_yaml and dataset.yolo_model):
                print("Both data.yaml and yolo.pt must be present.\n")
                continue
            
            # Did not select all splits, set the paths in the args based on user's dataset choice
            args.data_yaml_path = dataset.data_yaml
            args.yolo_pt_path = dataset.yolo_model

            break
        except ValueError:
            print("Invalid input. Please enter an integer.\n")

    # If the user didn't choose all splits, evaluate and output the statistics
    if dataset.data_yaml.parent.name != "all":
        output_stats(args, val_split(args))
        return

    # If got to this section of code, this means the user chose to evaluate all splits
    
    splits = sorted(
        [
            d for d in datasets if any(parent == od_folder for parent in d.data_yaml.parents) and
            d != dataset
        ],
        key=lambda d: str(d.data_yaml)
    )

    splits_stats: list[dict] = []
    
    # Collect statistics for each split
    for d in splits:
        args.data_yaml_path = d.data_yaml
        args.yolo_pt_path = d.yolo_model
        splits_stats.append(val_split(args))

    # Reset paths in args for summary statistics calculation
    args.data_yaml_path = None
    args.yolo_pt_path = None

    stats = {}

    # Calculate and average statistics across all splits
    for stat in splits_stats[0].keys():
        if isinstance(splits_stats[0][stat], list):
            continue

        total = 0
        for i in range(len(splits_stats)):
            total += splits_stats[i][stat] 
        
        stats[stat] = total / len(splits_stats)
    
    # Output the averaged statistics
    output_stats(args, stats)



if __name__ == "__main__":
    main()