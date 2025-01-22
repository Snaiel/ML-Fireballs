import numpy as np

from utils.constants import SQUARE_SIZE


def get_image_tile(image: np.ndarray, tile_pos: tuple) -> np.ndarray:
    return image[tile_pos[1] : tile_pos[1] + SQUARE_SIZE, tile_pos[0] : tile_pos[0] + SQUARE_SIZE]