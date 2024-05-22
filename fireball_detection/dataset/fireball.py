from pathlib import Path

import numpy as np
from dataset.point_pickings import PointPickings
from dataset import GFO_PICKINGS, DATASET_FOLDER
from skimage import io

class Fireball:


    def __init__(self, fireball_filename: str) -> None:
        self._image: np.ndarray
        self._label: list
        self._image_dimensions: tuple[int, int]

        self.fireball_filename = fireball_filename
        self.fireball_name = fireball_filename.split(".")[0]
        self.pp = PointPickings(Path(GFO_PICKINGS, self.fireball_name + ".csv"))


    @property
    def stores_image(self) -> bool:
        return self.image.any()


    @property
    def image(self) -> np.ndarray:
        return self._image


    @property
    def label(self) -> list:
        return self._label
    

    @property
    def image_dimensions(self) -> tuple[int, int]:
        return self._image_dimensions


    def save_image(self, dataset: str) -> None:
        io.imsave(
            Path(DATASET_FOLDER, "images", dataset, self.fireball_name + ".jpg"),
            self._image,
            check_contrast=False
        )


    def save_label(self, dataset: str) -> None:
        if self.label[0] != 0:
            self.label.insert(0, 0)

        with open(Path(DATASET_FOLDER, "labels", dataset, self.fireball_name + ".txt"), 'x') as label_file:
            label_file.write(" ".join(str(item) for item in self.label))