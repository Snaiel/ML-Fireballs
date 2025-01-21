import numpy as np

from utils.constants import (MAX_PIXEL_TOTAL_THRESHOLD,
                             MIN_PIXEL_TOTAL_THRESHOLD,
                             PIXEL_BRIGHTNESS_THRESHOLD)


def satisfies_thresholds(tile: np.ndarray) -> bool:
    pixels_over_threshold = np.sum(tile > PIXEL_BRIGHTNESS_THRESHOLD)
    return MIN_PIXEL_TOTAL_THRESHOLD < pixels_over_threshold < MAX_PIXEL_TOTAL_THRESHOLD