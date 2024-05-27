from pathlib import Path

import numpy as np
import skimage as ski
from dataset import GFO_PICKINGS
from dataset.point_pickings import PointPickings


class Fireball:

    def __init__(self, fireball_name: str, point_pickings: PointPickings = None) -> None:
        self._image: np.ndarray = None
        self._label: list = None
        self._image_dimensions: tuple[int, int] = None

        self.fireball_name = fireball_name

        if point_pickings:
            self.pp = point_pickings
        else:
            self.pp = PointPickings(Path(GFO_PICKINGS, self.fireball_name + ".csv"))


    @property
    def stores_image(self) -> bool:
        return self.image is not None


    @property
    def image(self) -> np.ndarray:
        return self._image


    @property
    def label(self) -> list:
        return self._label
    

    @property
    def image_dimensions(self) -> tuple[int, int]:
        return self._image_dimensions


    def save_image(self, folder: str) -> None:
        ski.io.imsave(
            Path(folder, self.fireball_name + ".jpg"),
            self._image,
            check_contrast=False
        )


    def save_label(self, folder: str) -> None:
        if self.label[0] != 0:
            self.label.insert(0, 0)

        with open(Path(folder, self.fireball_name + ".txt"), 'x') as label_file:
            label_file.write(" ".join(str(item) for item in self.label))