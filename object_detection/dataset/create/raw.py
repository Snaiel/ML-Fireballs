"""
Dataset that just uses the full sized images.

NOTES: Fireballs are hard to notice. Too small.
"""

import shutil
from pathlib import Path

from PIL import ExifTags, Image

from object_detection.dataset import GFO_JPEGS, GFO_THUMB_EXT, IMAGE_DIM
from object_detection.dataset.create import MIN_BB_DIM_SIZE, create_dataset
from object_detection.dataset.fireball import Fireball
from object_detection.dataset.point_pickings import PointPickings


class RawFireball(Fireball):
    def __init__(self, fireball_name: str, point_pickings: PointPickings = None) -> None:
        super().__init__(fireball_name, point_pickings)

        bb_width = max(self.pp.bounding_box_dim[0], MIN_BB_DIM_SIZE)
        bb_height = max(self.pp.bounding_box_dim[1], MIN_BB_DIM_SIZE)

        norm_bb_centre_x = self.pp.bb_centre_x / IMAGE_DIM[0]
        norm_bb_centre_y = self.pp.bb_centre_y / IMAGE_DIM[1]

        norm_bb_width = bb_width / IMAGE_DIM[0]
        norm_bb_height = bb_height / IMAGE_DIM[1]

        self._label = [norm_bb_centre_x, norm_bb_centre_y, norm_bb_width, norm_bb_height]

        self._image_dimensions = IMAGE_DIM
    

    def save_image(self, folder: str) -> None:
        image_path = Path(GFO_JPEGS, self.fireball_name + GFO_THUMB_EXT)
        dest_path = Path(folder, self.fireball_name + ".jpg")


        image = Image.open(image_path)
        exif = image.getexif()
        for tag_id, value in exif.items():
            # Get the tag name from TAGS dictionary
            tag_name = ExifTags.TAGS.get(tag_id, tag_id)
            
            # Check if this is the Orientation tag
            if tag_name == 'Orientation':
                orientation_value = value
        
        
        if orientation_value == 1:
            shutil.copy(
                image_path,
                dest_path
            )
        else:
            image.save(dest_path)



if __name__ == "__main__":
    create_dataset(RawFireball)