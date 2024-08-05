from pathlib import Path
from typing import Optional

import numpy as np
import skimage as ski

from object_detection.dataset import GFO_PICKINGS
from object_detection.dataset.point_pickings import PointPickings


class Fireball:

    pp: PointPickings
    image: np.ndarray
    label: list
    image_dimensions: tuple[int, int]

    def __init__(self, fireball_name: str, point_pickings: Optional[PointPickings] = None) -> None:
        self.image: np.ndarray = None
        self.label: list = None
        self.image_dimensions: tuple[int, int] = None

        self.fireball_name = fireball_name

        if point_pickings:
            self.pp = point_pickings
        else:
            self.pp = PointPickings(Path(GFO_PICKINGS, self.fireball_name + ".csv"))


    @property
    def stores_image(self) -> bool:
        return self.image is not None


    def save_image(self, folder: str) -> None:
        ski.io.imsave(
            Path(folder, self.fireball_name + ".jpg"),
            self._image,
            check_contrast=False
        )


    def save_label(self, folder: str) -> None:
        # this is for the class label
        if self.label[0] != 0:
            self.label.insert(0, 0)

        with open(Path(folder, self.fireball_name + ".txt"), 'x') as label_file:
            label_file.write(" ".join(str(item) for item in self.label))