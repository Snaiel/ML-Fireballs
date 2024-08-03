import pandas as pd
import numpy as np

IMAGE_DIM = (7360, 4912)
PADDING = 0.05

class PointPickings:
    def __init__(self, csv_file: str) -> None:
        point_pickings_df = pd.read_csv(csv_file)

        self._pp_min_x = point_pickings_df['x_image_thumb'].min()
        self._pp_max_x = point_pickings_df['x_image_thumb'].max()
        self._pp_min_y = point_pickings_df['y_image_thumb'].min()
        self._pp_max_y = point_pickings_df['y_image_thumb'].max()

        self._point_pickings_dim = (self._pp_max_x - self._pp_min_x, self._pp_max_y - self._pp_min_y)
        self._padding = PADDING
        self._image_dim = IMAGE_DIM

        self._bb_min_x = self._pp_min_x - (self._point_pickings_dim[0] * self._padding)
        self._bb_max_x = self._pp_max_x + (self._point_pickings_dim[0] * self._padding)

        self._bb_min_y = self._pp_min_y - (self._point_pickings_dim[1] * self._padding)
        self._bb_max_y = self._pp_max_y + (self._point_pickings_dim[1] * self._padding)

        self._bb_min_x = np.clip(self._bb_min_x, 0, self._image_dim[0])
        self._bb_max_x = np.clip(self._bb_max_x, 0, self._image_dim[0])

        self._bb_min_y = np.clip(self._bb_min_y, 0, self._image_dim[1])
        self._bb_max_y = np.clip(self._bb_max_y, 0, self._image_dim[1])

        self._bounding_box_dim = (self._bb_max_x - self._bb_min_x, self._bb_max_y - self._bb_min_y)

        self._bb_centre_x = (self._bb_min_x + self._bb_max_x) / 2
        self._bb_centre_y = (self._bb_min_y + self._bb_max_y) / 2

    @property
    def pp_min_x(self):
        return self._pp_min_x

    @property
    def pp_max_x(self):
        return self._pp_max_x

    @property
    def pp_min_y(self):
        return self._pp_min_y

    @property
    def pp_max_y(self):
        return self._pp_max_y

    @property
    def point_pickings_dim(self):
        return self._point_pickings_dim

    @property
    def bb_min_x(self):
        return self._bb_min_x

    @property
    def bb_max_x(self):
        return self._bb_max_x

    @property
    def bb_min_y(self):
        return self._bb_min_y

    @property
    def bb_max_y(self):
        return self._bb_max_y

    @property
    def bounding_box_dim(self):
        return self._bounding_box_dim
    
    @property
    def bb_centre_x(self):
        return self._bb_centre_x

    @property
    def bb_centre_y(self):
        return self._bb_centre_y