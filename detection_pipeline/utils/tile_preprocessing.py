import numpy as np

from utils.constants import (MAX_PIXEL_TOTAL_THRESHOLD,
                             MIN_PIXEL_TOTAL_THRESHOLD,
                             PIXEL_BRIGHTNESS_THRESHOLD)


def check_tile_threshold(tile: np.ndarray) -> bool:
    pixels_over_threshold = np.sum(tile > PIXEL_BRIGHTNESS_THRESHOLD)
    if pixels_over_threshold > MIN_PIXEL_TOTAL_THRESHOLD and pixels_over_threshold < MAX_PIXEL_TOTAL_THRESHOLD:
        return True
    else:
        return False