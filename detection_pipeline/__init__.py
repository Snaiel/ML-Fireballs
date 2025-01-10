import numpy as np


PIXEL_BRIGHTNESS_THRESHOLD = 10
MIN_PIXEL_TOTAL_THRESHOLD = 200
MAX_PIXEL_TOTAL_THRESHOLD = 50000

MAX_TIME_DIFFERENCE = 180
MIN_DIAGONAL_LENGTH = 60


def check_tile_threshold(tile: np.ndarray) -> bool:
    pixels_over_threshold = np.sum(tile > PIXEL_BRIGHTNESS_THRESHOLD)
    if pixels_over_threshold > MIN_PIXEL_TOTAL_THRESHOLD and pixels_over_threshold < MAX_PIXEL_TOTAL_THRESHOLD:
        return True
    else:
        return False