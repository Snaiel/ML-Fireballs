from pathlib import Path

import cv2
from included import retrieve_included_coordinates
from numpy import ndarray
from ultralytics import YOLO

from fireball_detection import SQUARE_SIZE


class Tile:
    position: tuple[float, float]
    image: ndarray

    def __init__(self, position: tuple[float, float], image: ndarray) -> None:
        self.position = position
        self.image = image


included_coordinates = retrieve_included_coordinates()
img = cv2.imread("../data/GFO_fireball_object_detection_training_set/jpegs/043_2021-05-12_163859_E_DSC_0946.thumb.jpg")

tiles: list[Tile] = []

for pos in included_coordinates:
    tiles.append(
        Tile(
            pos,
            img[pos[0] : pos[0] + SQUARE_SIZE, pos[1] : pos[1] + SQUARE_SIZE]
        )
    )


model = YOLO(Path(Path(__file__).parents[1], "data", "e14.pt"))
results = model(tiles[26].image)

for r in results:
    r.show()