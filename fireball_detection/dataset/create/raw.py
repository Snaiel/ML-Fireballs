"""
Dataset that just uses the full sized images.

NOTES: Fireballs are hard to notice. Too small.
"""

from pathlib import Path
import shutil
from dataset import IMAGE_DIM
from dataset.prepare_dataset import prepare_dataset, gfo_jpegs, gfo_pickings, dataset_folder
from dataset.point_pickings import PointPickings

fireball_dataset = prepare_dataset()

for dataset, fireballs in fireball_dataset.items():
    print(f"Creating {dataset} dataset...")
    for fireball in fireballs:
        fireball_name = fireball.split(".")[0]
        
        shutil.copy(
            Path(gfo_jpegs, fireball),
            Path(dataset_folder, "images", dataset, fireball_name + ".jpg")
        )
        
        pp = PointPickings(Path(gfo_pickings, fireball_name + ".csv"))

        norm_bb_centre_x = pp.bb_centre_x / IMAGE_DIM[0]
        norm_bb_centre_y = pp.bb_centre_y / IMAGE_DIM[1]

        norm_bb_width = pp.bounding_box_dim[0] / IMAGE_DIM[0]
        norm_bb_height = pp.bounding_box_dim[1] / IMAGE_DIM[1]

        label = [norm_bb_centre_x, norm_bb_centre_y, norm_bb_width, norm_bb_height]

        label.insert(0, 0)

        with open(Path(dataset_folder, "labels", dataset, fireball_name + ".txt"), 'x') as label_file:
            label_file.write(" ".join(str(item) for item in label))