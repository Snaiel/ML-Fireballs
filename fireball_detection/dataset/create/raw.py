"""
Dataset that just uses the full sized images.

NOTES: Fireballs are hard to notice. Too small.
"""

from pathlib import Path
import shutil
from dataset.point_pickings_to_bounding_boxes import get_yolov8_label_from_point_pickings_csv
from dataset.prepare_dataset import prepare_dataset, gfo_jpegs, gfo_pickings, dataset_folder

fireball_dataset = prepare_dataset()

for dataset, fireballs in fireball_dataset.items():
    print(f"Creating {dataset} dataset...")
    for fireball in fireballs:
        fireball_name = fireball.split(".")[0]
        
        shutil.copy(
            Path(gfo_jpegs, fireball),
            Path(dataset_folder, "images", dataset, fireball_name + ".jpg")
        )
        
        label = get_yolov8_label_from_point_pickings_csv(Path(gfo_pickings, fireball_name + ".csv"))
        label.insert(0, 0)

        with open(Path(dataset_folder, "labels", dataset, fireball_name + ".txt"), 'x') as label_file:
            label_file.write(" ".join(str(item) for item in label))