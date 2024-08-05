"""
Puts the point pickings in the centre of a 1840x1228 window.
"""

from pathlib import Path

from skimage import io

from object_detection.dataset import GFO_JPEGS, GFO_THUMB_EXT, IMAGE_DIM
from object_detection.dataset.create import MIN_BB_DIM_SIZE
from object_detection.dataset.fireball import Fireball
from object_detection.dataset.point_pickings import PointPickings


class TileCentredFireball(Fireball):

    window_dim: tuple[int, int]

    def __init__(self, fireball_name: str, point_pickings: PointPickings = None) -> None:
        super().__init__(fireball_name, point_pickings)

        # Calculate the x-coordinate of the left edge of the window
        window_x1 = self.pp.bb_centre_x - (self.window_dim[0] / 2)

        # Calculate the x-coordinate of the right edge of the window
        window_x2 = self.pp.bb_centre_x + (self.window_dim[0] / 2)

        # Calculate the distance between the left edge of the window and the left image boundary
        left_off_bounds = (self.window_dim[0] / 2) - self.pp.bb_centre_x

        # Calculate the distance between the right edge of the window and the right image boundary
        right_off_bounds = (self.window_dim[0] / 2) - (IMAGE_DIM[0] - self.pp.bb_centre_x)

        # Adjust window coordinates if it goes off the left or right bounds of the image
        if left_off_bounds > 0:
            window_x1 = 0
            window_x2 += left_off_bounds
        elif right_off_bounds > 0:
            window_x1 -= right_off_bounds
            window_x2 = IMAGE_DIM[0]

        # Calculate the y-coordinate of the top edge of the window
        window_y1 = self.pp.bb_centre_y - (self.window_dim[1] / 2)

        # Calculate the y-coordinate of the bottom edge of the window
        window_y2 = self.pp.bb_centre_y + (self.window_dim[1] / 2)

        # Calculate the distance between the top edge of the window and the top image boundary
        top_off_bounds = (self.window_dim[1] / 2) - self.pp.bb_centre_y

        # Calculate the distance between the bottom edge of the window and the bottom image boundary
        bottom_off_bounds = (self.window_dim[1] / 2) - (IMAGE_DIM[1] - self.pp.bb_centre_y)

        # Adjust window coordinates if it goes off the top or bottom bounds of the image
        if top_off_bounds > 0:
            window_y1 = 0
            window_y2 += top_off_bounds
        elif bottom_off_bounds > 0:
            window_y1 -= bottom_off_bounds
            window_y2 = IMAGE_DIM[1]

        window_x1 = int(window_x1)
        window_x2 = int(window_x2)
        window_y1 = int(window_y1)
        window_y2 = int(window_y2)


        fireball_image = io.imread(Path(GFO_JPEGS, fireball_name + GFO_THUMB_EXT))
        cropped_image = fireball_image[window_y1:window_y2, window_x1:window_x2]
        
        self._image = cropped_image


        # These pixel values are still relative to the original image size
        new_bb_min_x = max(window_x1, self.pp.bb_min_x)
        new_bb_max_x = min(window_x2, self.pp.bb_max_x)

        new_bb_min_y = max(window_y1, self.pp.bb_min_y)
        new_bb_max_y = min(window_y2, self.pp.bb_max_y)

        new_bb_width = max(new_bb_max_x - new_bb_min_x, MIN_BB_DIM_SIZE)
        new_bb_height = max(new_bb_max_y - new_bb_min_y, MIN_BB_DIM_SIZE)

        new_bb_centre_x = (new_bb_min_x + new_bb_max_x) / 2
        new_bb_centre_y = (new_bb_min_y + new_bb_max_y) / 2

        # These new normalised values are now relative to the window
        norm_bb_centre_x = (new_bb_centre_x - window_x1) / self.window_dim[0]
        norm_bb_centre_y = (new_bb_centre_y - window_y1) / self.window_dim[1]

        norm_bb_width = new_bb_width / self.window_dim[0]
        norm_bb_height = new_bb_height / self.window_dim[1]


        # final label
        self._label = [norm_bb_centre_x, norm_bb_centre_y, norm_bb_width, norm_bb_height]

        self._image_dimensions = self.window_dim