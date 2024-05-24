"""
    Crop fireball image using the corresponding thumbnail tile.

    Usage:
        Import module and call load_cropped_image

        Run as script to view cropped fireballs. While in `points_processing` folder, run:

        python3 thumb_tile/crop_using_thumb_tile.py
"""


import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski


# Matches: 025_2021-09-09_053629_E_DSC_0379.thumb.4800.1600.tile
TILE_FILE_PATTERN = re.compile(r'^(.*?)\.thumb\.(\d+)\.(\d+)\.tile', re.IGNORECASE)


def load_cropped_image(event_folder_path: str, crop_size: int = 2500) -> np.ndarray | None:
    """
        Load and Crop Image from Event Folder

        This function loads an image from a specified event folder, searches for the thumb tile,
        and crops the image to a predefined square size around specified coordinates extracted from the file name.

        Parameters:
        - event_folder (str): The path to the event folder containing the image and tile files.
        - crop_size (int): The length of a side of the square that will be cropped.

        Returns:
        - np.ndarray | None: The cropped image as a NumPy array if a matching file is found; otherwise, None.
    """

    match = None
    for file in os.listdir(event_folder_path):
        match = re.search(TILE_FILE_PATTERN, file)
        if match:
            break
    
    if match is None:
        return None

    event_name = match.group(1)
    x = match.group(2)
    y = match.group(3)
    print(event_name, x, y)

    image = ski.io.imread(Path(event_folder_path, event_name + "-G.jpeg"))

    # Calculate the top-left corner coordinates for cropping
    tl_x_coord = max (1, int(x) - crop_size // 2)
    tl_y_coord = max(1, int(y) - crop_size // 2)

    br_x_coord = min(tl_x_coord + crop_size, image.shape[1])
    br_y_coord = min(tl_y_coord + crop_size, image.shape[0])

    # Perform cropping
    cropped_image = image[tl_y_coord:br_y_coord, tl_x_coord:br_x_coord]

    return cropped_image


def main():
    dfn_highlights_folder = Path(Path(__file__).parents[2], 'data', 'dfn_highlights')
    for event_folder in os.listdir(dfn_highlights_folder):
        cropped_image = load_cropped_image(Path(dfn_highlights_folder, event_folder))

        # Plot the original image with fitted curve
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(event_folder)
        ax.imshow(cropped_image, cmap='gray')

        # plt.title(event_folder)
        plt.tight_layout()
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    main()