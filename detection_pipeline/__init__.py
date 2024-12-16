import numpy as np


PIXEL_BRIGHTNESS_THRESHOLD = 10
PIXEL_TOTAL_THRESHOLD = 200

MAX_TIME_DIFFERENCE = 180
MIN_DIAGONAL_LENGTH = 60


def check_tile_threshold(tile: np.ndarray) -> bool:
    pixels_over_threshold = np.sum(tile > PIXEL_BRIGHTNESS_THRESHOLD)
    if pixels_over_threshold > PIXEL_TOTAL_THRESHOLD:
        return True
    else:
        return False