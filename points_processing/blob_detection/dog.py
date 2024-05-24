"""
    Blob Detection Script using Difference of Gaussian (DoG)

    This script performs blob detection on a grayscale image using the Difference of Gaussian (DoG) method from the scikit-image library.
    The script includes the following main functionalities:
    - Loading a grayscale image from a specified file path.
    - Detecting blobs in the image using the DoG method.
    - Visualizing the detected blobs on the original image using matplotlib.

    Dependencies:
    - scikit-image

    Usage:
        Run the script to perform blob detection on the predefined sample image and display the detected blobs.

        Whilst in `points_processing/`, run:

        python3 blob_detection/dog.py
"""


from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import skimage as ski
from skimage.feature import blob_dog


image_path = Path(Path(__file__).parents[2], 'data', 'fireball_images', 'cropped', '051_2021-12-11_182730_E_DSC_0884-G_cropped.jpeg')

image = ski.io.imread(image_path)

image_gray = image

blobs_dog = blob_dog(image_gray, max_sigma=20, threshold=.025)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

fig, ax = plt.subplots(figsize=(6, 6))

ax.set_title('Difference of Gaussian')
ax.imshow(image, cmap='gray')

for blob in blobs_dog:
    y, x, r = blob
    c = plt.Circle((x, y), r, color='lime', linewidth=1, fill=False)
    ax.add_patch(c)

plt.axis('off')
plt.tight_layout()
plt.show()