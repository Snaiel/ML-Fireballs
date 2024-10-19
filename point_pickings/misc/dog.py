from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import skimage as ski
from skimage.feature import blob_dog
import os


def show_dog(image_path: str) -> None:
    image = ski.io.imread(image_path, True)

    blobs_dog = blob_dog(image, max_sigma=20, threshold=.1)
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


def main():
    image_path = Path(Path(__file__).parents[2], 'data', 'fireball_highlights', 'cropped', '051_2021-12-11_182730_E_DSC_0884-G_cropped.jpeg')
    show_dog(image_path)

if __name__ == "__main__":
    main()