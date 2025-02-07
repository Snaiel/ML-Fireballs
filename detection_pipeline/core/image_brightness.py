from pathlib import Path

import cv2
import numpy as np

from utils.constants import (MAX_THRESHOLD_MEDIAN_BRIGHTNESS,
                             MIN_THRESHOLD_MAX_BRIGHTNESS)


def check_image_brightness(image_path: Path) -> None | str:
    """
    returns None if image is good otherwise returns reason it is bad.
    """
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    median = np.median(image)
    if median > MAX_THRESHOLD_MEDIAN_BRIGHTNESS:
        return f"median pixel brightness too bright ({median} > {MAX_THRESHOLD_MEDIAN_BRIGHTNESS})"
    
    max = np.max(image)
    if max < MIN_THRESHOLD_MAX_BRIGHTNESS:
        return f"max pixel brightness too low ({max} < {MIN_THRESHOLD_MAX_BRIGHTNESS})"
    
    return None