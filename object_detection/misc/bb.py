import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    if label:
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


def main():

    @dataclass
    class Args:
        data_yaml_path: str
        split: str

    parser = argparse.ArgumentParser(
        description="View bounding boxes of tiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--data_yaml_path', type=str, help='path to data.yaml')
    parser.add_argument('--split', type=str, help='train or val')

    args = Args(**vars(parser.parse_args()))
    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    
    images_folder = Path(Path(args.data_yaml_path).parent, "images", args.split)
    labels_folder = Path(Path(args.data_yaml_path).parent, "labels", args.split)

    images = sorted(os.listdir(images_folder))

    for image_filename in images:
        if "negative" in image_filename:
            continue

        image_file = Path(images_folder, image_filename)

        fireball_name = image_filename.split(".")[0]

        with open(Path(labels_folder, fireball_name + ".txt"), "r") as label_file:
            label = label_file.read().split(" ")
            label = [float(i) for i in label[1:]]

        image = io.imread(image_file)

        plot_fireball_bb(image, label)


if __name__ == "__main__":
    main()