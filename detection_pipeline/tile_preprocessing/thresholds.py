from dataclasses import dataclass

import numpy as np

from utils.constants import (MAX_PIXEL_TOTAL_THRESHOLD,
                             MIN_PIXEL_TOTAL_THRESHOLD,
                             PIXEL_BRIGHTNESS_THRESHOLD, VARIANCE_THRESHOLD)


@dataclass
class TilePreprocessingThresholds:
    pixel_threshold: int = MAX_PIXEL_TOTAL_THRESHOLD
    min_pixel: int = MIN_PIXEL_TOTAL_THRESHOLD
    max_pixel: int = PIXEL_BRIGHTNESS_THRESHOLD
    variance_threshold: float = VARIANCE_THRESHOLD


def satisfies_thresholds(
        tile: np.ndarray,
        thresholds: TilePreprocessingThresholds = None
    ) -> bool:
    
    if not thresholds:
        thresholds = TilePreprocessingThresholds()

    pixels_over_threshold = np.sum(tile > thresholds.pixel_threshold)
    
    if pixels_over_threshold < thresholds.min_pixel:
        return False
    
    if pixels_over_threshold > thresholds.max_pixel and np.var(tile) < thresholds.variance_threshold:
        return False

    return True