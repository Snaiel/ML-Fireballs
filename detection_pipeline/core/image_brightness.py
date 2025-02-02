from pathlib import Path

import numpy as np
import skimage.io as io
import cv2


def check_image_brightness(image_path: Path) -> None | str:
    """
    returns None if image is good otherwise returns reason it is bad.
    """
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    median = np.median(image)
    median_max_threshold = 165
    if median > median_max_threshold:
        return f"median pixel brightness too bright ({median:.2f} > {median_max_threshold})"
    
    max = np.max(image)
    max_min_threshold =40
    if max < max_min_threshold:
        return f"max pixel brightness too low ({max:.2f} < {max_min_threshold})"
    
    return None