"""
Dataset that just uses the full sized images.

NOTES: Fireballs are hard to notice. Too small.
"""

from pathlib import Path

import numpy as np
import skimage
from dataset import GFO_JPEGS, GFO_THUMB_EXT, IMAGE_DIM
from dataset.create import MIN_BB_DIM_SIZE, create_dataset
from dataset.fireball import Fireball
from dataset.point_pickings import PointPickings

## NOTE: THIS TAKES FOREVER. CHANGE IT TO USE GPU NEXT TIME.


class DownscaledRawFireball(Fireball):
    max_dim: int = 1280

    def __init__(self, fireball_name: str, point_pickings: PointPickings = None) -> None:
        super().__init__(fireball_name, point_pickings)

        bb_width = max(self.pp.bounding_box_dim[0], MIN_BB_DIM_SIZE)
        bb_height = max(self.pp.bounding_box_dim[1], MIN_BB_DIM_SIZE)

        norm_bb_centre_x = self.pp.bb_centre_x / IMAGE_DIM[0]
        norm_bb_centre_y = self.pp.bb_centre_y / IMAGE_DIM[1]

        norm_bb_width = bb_width / IMAGE_DIM[0]
        norm_bb_height = bb_height / IMAGE_DIM[1]

        image = skimage.io.imread(Path(GFO_JPEGS, fireball_name + GFO_THUMB_EXT))

        # make landscape
        height, width = image.shape[:2]
        if width < height:
            image = skimage.transform.rotate(image, angle=90, resize=True)
        
        # Determine the scaling factor
        if height > width:
            scale_factor = self.max_dim / height
        else:
            scale_factor = self.max_dim / width
        
        # Calculate the new dimensions
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        # Resize the image
        downscaled_image = skimage.transform.resize(image, (new_height, new_width), anti_aliasing=True)
        image_uint8 = (downscaled_image * 255).astype(np.uint8)
        
        self._image = image_uint8
        self._image_dimensions = (new_width, new_height)
        self._label = [norm_bb_centre_x, norm_bb_centre_y, norm_bb_width, norm_bb_height]


if __name__ == "__main__":
    create_dataset(DownscaledRawFireball)