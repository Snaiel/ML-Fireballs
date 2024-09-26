"""
csv file in the format

x_image_thumb,y_image_thumb
"""

import pandas as pd
import numpy as np


IMAGE_DIM = (7360, 4912)
PADDING = 0.05

class PointPickings:

    pp: pd.DataFrame
    pp_min_x: int
    pp_max_x: int
    pp_min_y: int
    pp_max_y: int
    point_pickings_dim: int
    bb_min_x: int
    bb_max_x: int
    bb_min_y: int
    bb_max_y: int
    bounding_box_dim: int
    bb_centre_x: int
    bb_centre_y: int

    def __init__(self, csv_file: str) -> None:
        self.pp = pd.read_csv(csv_file)

        self.pp_min_x = self.pp['x_image_thumb'].min()
        self.pp_max_x = self.pp['x_image_thumb'].max()
        self.pp_min_y = self.pp['y_image_thumb'].min()
        self.pp_max_y = self.pp['y_image_thumb'].max()

        self.point_pickings_dim = (self.pp_max_x - self.pp_min_x, self.pp_max_y - self.pp_min_y)
        self.padding = PADDING

        self.bb_min_x = self.pp_min_x - (self.point_pickings_dim[0] * self.padding)
        self.bb_max_x = self.pp_max_x + (self.point_pickings_dim[0] * self.padding)

        self.bb_min_y = self.pp_min_y - (self.point_pickings_dim[1] * self.padding)
        self.bb_max_y = self.pp_max_y + (self.point_pickings_dim[1] * self.padding)

        self.bb_min_x = np.clip(self.bb_min_x, 0, IMAGE_DIM[0])
        self.bb_max_x = np.clip(self.bb_max_x, 0, IMAGE_DIM[0])

        self.bb_min_y = np.clip(self.bb_min_y, 0, IMAGE_DIM[1])
        self.bb_max_y = np.clip(self.bb_max_y, 0, IMAGE_DIM[1])

        self.bounding_box_dim = (self.bb_max_x - self.bb_min_x, self.bb_max_y - self.bb_min_y)

        self.bb_centre_x = (self.bb_min_x + self.bb_max_x) / 2
        self.bb_centre_y = (self.bb_min_y + self.bb_max_y) / 2