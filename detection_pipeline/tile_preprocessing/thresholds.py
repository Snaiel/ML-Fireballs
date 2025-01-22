import numpy as np
from dataclasses import dataclass
from utils.constants import (MAX_PIXEL_TOTAL_THRESHOLD,
                             MIN_PIXEL_TOTAL_THRESHOLD,
                             PIXEL_BRIGHTNESS_THRESHOLD)


@dataclass
class TilePreprocessingThresholds:
    pixel_threshold: int
    min_pixel: int
    max_pixel: int
    variance_threshold: float


def satisfies_thresholds(
        tile: np.ndarray,
        thresholds: TilePreprocessingThresholds = None
    ) -> bool:
    
    if not thresholds:
        thresholds = TilePreprocessingThresholds(
            PIXEL_BRIGHTNESS_THRESHOLD,
            MIN_PIXEL_TOTAL_THRESHOLD,
            MAX_PIXEL_TOTAL_THRESHOLD,
            1.0
        )

    pixels_over_threshold = np.sum(tile > thresholds.pixel_threshold)
    
    if pixels_over_threshold < thresholds.min_pixel:
        return False
    
    if pixels_over_threshold > thresholds.max_pixel and np.var(tile) < thresholds.variance_threshold:
        return False

    return True