import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io as io
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.ops import xywhn2xyxy

from fireball_detection.detect import intersects
from object_detection.utils import add_border, iom, iou
from object_detection.dataset import DATA_FOLDER


def main():

    @dataclass
    class Args:
        split: int
        metric: str
        threshold: float | None

    parser = argparse.ArgumentParser(
        description='For a given split, test recall of border sizes 0-32 inclusive.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--split', type=int, required=True, help='The split number to use for K-Fold cross-validation (0, 1, 2, 3, 4)')
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

    print("args:", vars(args))
    print()

    model = YOLO(Path(Path(__file__).parents[2], "runs", "detect", f"train2{args.split}", "weights", "last.pt"))


    KFOLD_FOLDER = Path(Path(__file__).parents[2], "data", "kfold_object_detection", f"split{args.split}")
    VAL_IMAGES_FOLDER = Path(KFOLD_FOLDER, "images", "val")
    VAL_LABELS_FOLDER = Path(KFOLD_FOLDER, "labels", "val")

    print("kfold folder:", KFOLD_FOLDER)
    print()


    image_files = os.listdir(VAL_IMAGES_FOLDER)
    total_positive_samples = sum(1 for i in image_files if not "negative" in i)


    images = {}
    for i in tqdm(image_files, desc="loading images"):
        image = io.imread(Path(VAL_IMAGES_FOLDER, i))
        images[i] = image

    border_sizes = range(33)

    boxes_list = []
    recalls = []
    precisions = []

    # Process each border size
    for b_size in tqdm(border_sizes, desc="evaluating each border size", position=0):
        
        detected_fireballs = 0
        true_positive_boxes = 0
        total_boxes = 0

        false_negative_files = []

        # Run predictions on images with the current border size
        for file, image in tqdm(images.items(), desc=f"border size {b_size}", position=1):
            
            positive_sample = "negative" not in file

            if b_size > 0:
                bordered_image = add_border(image, b_size)
            else:
                bordered_image = image
            
            fireball = file.split(".")[0]

            # Use the model to predict bounding boxes for the bordered image
            results = model.predict(bordered_image, verbose=False, imgsz=416)
            boxes = results[0].boxes.xyxy.cpu()  # Extract predicted boxes

            total_boxes += len(boxes)  # Update total number of boxes

            if not positive_sample:
                continue

            # Read the corresponding label file and convert xywh to xyxy format
            with open(Path(VAL_LABELS_FOLDER, fireball + ".txt")) as label_file:
                xyxy = xywhn2xyxy(
                    np.array([float(i) for i in label_file.read().split(" ")[1:]]),
                    400,
                    400
                )

            # Check for false negatives (no predicted boxes)
            if len(boxes) == 0:
                false_negative_files.append(fireball)  # Add to false negatives if no boxes predicted

            ack_true_positive = False
            for box in boxes:
                if (args.metric == "intersects" and intersects(xyxy, box)) or \
                (args.metric == "iom" and iom(xyxy, box) >= args.threshold) or \
                (args.metric == "iou" and iou(xyxy, box) >= args.threshold):
                    true_positive_boxes += 1
                    if not ack_true_positive:
                        detected_fireballs += 1
                        ack_true_positive = True


        boxes_list.append(total_boxes)
        
        recall = detected_fireballs / total_positive_samples if total_positive_samples > 0 else 0
        recalls.append(recall)

        precision = true_positive_boxes / total_boxes if total_boxes > 0 else 0
        precisions.append(precision)

        print(f"Border size: {b_size}, Recall: {recall}, Precision: {precision}, Boxes: {total_boxes} ")


    normalised_boxes = [i / boxes_list[0] for i in boxes_list]

    print()
    print("box totals:")
    for i in zip(normalised_boxes, boxes_list):
        print(i)

    print()
    print("recalls:")
    for i in recalls:
        print(i)

    print()
    print("precisions:")
    for i in precisions:
        print(i)

    df = pd.DataFrame({
        'box_totals': boxes_list,
        'recall': recalls,
        'precision': precisions
    })

    csv_path = Path(DATA_FOLDER, f'border_sizes_fold{args.split}.csv')
    df.to_csv(csv_path, index=False)

    print()
    print(f"statistics saved to: \"{csv_path}\"")

    plt.figure(figsize=(10, 5))
    plt.plot(border_sizes, recalls, marker='o', label="Recall")
    plt.plot(border_sizes, precisions, marker='x', label='Precision')
    plt.plot(border_sizes, normalised_boxes, marker='s', label='Normalised Box Totals')
    plt.title(f"Statistics with Different Border Sizes using {vars(args)}")
    plt.xlabel("Border Size")
    plt.ylabel("Percentrage")
    plt.xticks(border_sizes)
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()