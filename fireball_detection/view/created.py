from dataset import DATASET_FOLDER
from pathlib import Path
import os
from view.bb import plot_fireball_bb
from skimage import io
import argparse


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='View image and bounding box of fireball')

    # Add positional arguments
    parser.add_argument('dataset', type=str, help='Which dataset to view: train, val, test')

    # Parse the command-line arguments
    args = parser.parse_args()
    
    images_folder = Path(DATASET_FOLDER, "images", args.dataset)
    labels_folder = Path(DATASET_FOLDER, "labels", args.dataset)

    images = os.listdir(images_folder)

    for image_filename in images:
        image_file = Path(images_folder, image_filename)

        fireball_name = image_filename.split(".")[0]

        with open(Path(labels_folder, fireball_name + ".txt"), "r") as label_file:
            label = label_file.read().split(" ")
            label = [float(i) for i in label[1:]]

        image = io.imread(image_file)

        plot_fireball_bb(image, label)


if __name__ == "__main__":
    main()