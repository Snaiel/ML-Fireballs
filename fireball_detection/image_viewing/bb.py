import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dataset import GFO_JPEGS
from dataset.create.raw import RawFireball
from dataset.create.tile_centred import TileCentredFireball
from dataset.fireball import Fireball
from matplotlib.patches import Rectangle
from skimage import io


def plot_fireball_bb(image: np.ndarray, label: list, image_dimensions: tuple = None) -> None:
    # Display the image using matplotlib
    fig, ax = plt.subplots()
    ax.imshow(image)

    dim = image.shape
    if not image_dimensions:
        image_dimensions = (dim[1], dim[0], *dim[2:])

    # Define the rectangle parameters: (x, y, width, height)
    c_x, c_y, rect_width, rect_height = label
    c_x *= image_dimensions[0]
    c_y *= image_dimensions[1]
    rect_width *= image_dimensions[0]
    rect_height *= image_dimensions[1]

    rect_x = c_x - rect_width / 2
    rect_y = c_y - rect_height / 2

    # Create a rectangle patch
    rect = Rectangle((rect_x, rect_y), rect_width, rect_height,
                    linewidth=2, edgecolor='r', facecolor='none')

    # Add the rectangle to the plot
    ax.add_patch(rect)

    # Show the plot with the image and rectangle
    plt.show()


def show_fireball_bb(fireball_type: Fireball) -> None:
    FIREBALL_FILENAME = "10_2018-09-10_203029_S_DSC_0312.thumb.jpg"
    FIREBALL_IMAGE_PATH = Path(GFO_JPEGS, FIREBALL_FILENAME)
    
    fireball: Fireball = fireball_type(FIREBALL_FILENAME)

    label = fireball.label
    print(f"Label info: {label}")

    # Load the image using skimage
    image = fireball.image if fireball.stores_image else io.imread(FIREBALL_IMAGE_PATH)

    plot_fireball_bb(image, label, fireball.image_dimensions)



def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='View image and bounding box of fireball')

    # Add positional arguments
    parser.add_argument('fireball', type=str, help='Type of fireball dataset to view: raw, tile_centred')

    # Parse the command-line arguments
    args = parser.parse_args()

    fireball_types = {
        "raw": RawFireball,
        "tile_centred": TileCentredFireball
    }

    if args.fireball not in fireball_types:
        parser.error("No fireball type exists for: " + args.fireball)

    show_fireball_bb(fireball_types[args.fireball])


if __name__ == "__main__":
    main()