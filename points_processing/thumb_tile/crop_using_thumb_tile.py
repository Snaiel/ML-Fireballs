
import os, re
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import skimage as ski
from skimage.feature import blob_dog

file_path = Path(__file__)
dfn_highlights_folder = Path(file_path.parents[1], 'dfn_highlights')

# Define regular expression pattern to capture thumb, numbers, and tile
pattern = re.compile(r'^(.*?)\.thumb\.(\d+)\.(\d+)\.tile', re.IGNORECASE)

for event_folder in os.listdir(dfn_highlights_folder):
    print(event_folder)

    for file in os.listdir(os.path.join(dfn_highlights_folder, event_folder)):
        match = re.search(pattern, file)
        if match:
            event_name = match.group(1)
            x = match.group(2)
            y = match.group(3)
            print(event_name, x, y)

            image = ski.io.imread(os.path.join(dfn_highlights_folder, event_folder, event_name + "-G.jpeg"))

            crop_size = 2500  # Size of the square crop

            # Calculate the top-left corner coordinates for cropping
            tl_x_coord = int(x) - crop_size // 2
            tl_x_coord = tl_x_coord if tl_x_coord >= 0 else 1
            tl_y_coord = int(y) - crop_size // 2
            tl_y_coord = tl_y_coord if tl_y_coord >= 0 else 1

            br_x_coord = tl_x_coord + crop_size
            br_x_coord = br_x_coord if br_x_coord <= image.shape[1] else image.shape[1]
            br_y_coord = tl_y_coord + crop_size
            br_y_coord = br_y_coord if br_y_coord <= image.shape[0] else image.shape[0]

            # Perform cropping
            cropped_image = image[tl_y_coord:br_y_coord, tl_x_coord:br_x_coord]

            # Return (y-coord, x-coord, radius)
            blobs_dog = blob_dog(cropped_image, max_sigma=20, threshold=.025)
            # Compute radii in the 3rd column.
            blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

            # Plot the original image with fitted curve
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_title(event_folder)
            ax.imshow(cropped_image, cmap='gray')

            # Plot pink circles around inlier points
            for blob in blobs_dog:
                y, x, r = blob
                c = plt.Circle((x, y), r, color='pink', linewidth=2, fill=False)
                ax.add_patch(c)

            # plt.title(event_folder)
            plt.tight_layout()
            plt.axis('off')
            plt.show()
