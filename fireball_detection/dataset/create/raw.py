"""
Dataset that just uses the full sized images.

NOTES: Fireballs are hard to notice. Too small.
"""

import shutil
from pathlib import Path

from dataset import IMAGE_DIM, GFO_JPEGS, DATASET_FOLDER
from dataset.fireball import Fireball
from dataset.utils import create_dataset


class RawFireball(Fireball):
    def __init__(self, fireball_filename: str) -> None:
        super().__init__(fireball_filename)

        norm_bb_centre_x = self.pp.bb_centre_x / IMAGE_DIM[0]
        norm_bb_centre_y = self.pp.bb_centre_y / IMAGE_DIM[1]

        norm_bb_width = self.pp.bounding_box_dim[0] / IMAGE_DIM[0]
        norm_bb_height = self.pp.bounding_box_dim[1] / IMAGE_DIM[1]

        self._label = [norm_bb_centre_x, norm_bb_centre_y, norm_bb_width, norm_bb_height]

        self._image_dimensions = IMAGE_DIM
    

    def save_image(self, dataset: str) -> None:
        shutil.copy(
            Path(GFO_JPEGS, self.fireball_filename),
            Path(DATASET_FOLDER, "images", dataset, self.fireball_name + ".jpg")
        )


if __name__ == "__main__":
    create_dataset(RawFireball)