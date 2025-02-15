import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from skimage import io

from detection_pipeline.tile_preprocessing import satisfies_thresholds
from fireball_detection import FireballBox, Tile
from fireball_detection.boxes.fireball_boxes import get_absolute_fireball_boxes
from fireball_detection.boxes.merge import merge_bboxes
from fireball_detection.tiling import (get_image_tile,
                                       retrieve_included_coordinates)
from object_detection.detectors import Detector, get_detector
from object_detection.utils import add_border, diagonal_length
from utils.constants import MIN_DIAGONAL_LENGTH


INCLUDED_COORDINATES = retrieve_included_coordinates()


def detect_tiles_common(detector: Detector, border_size: int, tiles: list[Tile]) -> list[Tile]:
    """
    Common function to detect objects on tiles of an image.

    Args:
        image (np.ndarray): The input image to process.
        model (YOLO): A pre-trained YOLO model used for detection.
        border_size (int): The width of the border to add around each tile before detection.

    Returns:
        list[Tile]: A list of Tile objects that contain detected objects.
    """
    
    detected_tiles: list[Tile] = []
    for tile in tiles:
        input_image = add_border(tile.image, border_size)
        boxes, confidences, labels = detector.detect(input_image)
        if len(boxes) > 0:
            tile.boxes = boxes
            tile.confidences = confidences
            detected_tiles.append(tile)
    
    return detected_tiles

    
def detect_standalone_tiles(image: np.ndarray, detector: Detector, border_size: int) -> list[Tile]:
    tiles: list[Tile] = [
        Tile(pos, get_image_tile(image, pos))
        for pos in INCLUDED_COORDINATES
    ]
    return detect_tiles_common(detector, border_size, tiles)


def detect_differenced_tiles(image: np.ndarray, detector: Detector, border_size: int) -> list[Tile]:
    tiles: list[Tile] = [
        Tile(pos, np.stack((tile_image,) * 3, axis=-1))
        for pos in INCLUDED_COORDINATES
        if (tile_image := get_image_tile(image, pos)).any()
        and satisfies_thresholds(tile_image)
    ]
    return detect_tiles_common(detector, border_size, tiles)


def detect_differenced_norm_tiles(image: np.ndarray, detector: Detector, border_size: int) -> list[Tile]:
    max_value = np.max(image)
    norm_differenced_image = image
    if max_value > 0:
        norm_differenced_image = (norm_differenced_image / max_value) * 255
    norm_differenced_image = norm_differenced_image.astype(np.uint8)

    tiles: list[Tile] = [
        Tile(pos, np.stack((tile_image,) * 3, axis=-1))
        for pos in INCLUDED_COORDINATES
        if (tile_image := get_image_tile(norm_differenced_image, pos)).any()
        and satisfies_thresholds(tile_image)
    ]
    
    return detect_tiles_common(detector, border_size, tiles)


def detect_differenced_tiles_norm(image: np.ndarray, detector: Detector, border_size: int) -> list[Tile]:
    tiles: list[Tile] = []

    for pos in INCLUDED_COORDINATES:
        tile_image = get_image_tile(image, pos)

        if not satisfies_thresholds(tile_image):
            continue
        
        max_value = np.max(tile_image)
        if max_value <= 0:
            continue
        
        tile_image: np.ndarray = (tile_image / max_value) * 255
        tile_image = tile_image.astype(np.uint8)
        tiles.append(Tile(pos, np.stack((tile_image,) * 3, axis=-1)))

    return detect_tiles_common(detector, border_size, tiles)


def detect_fireballs(image: np.ndarray, detector: Detector, border_size: int = 5) -> list[FireballBox]:
    """
    Detects fireballs within an image using a YOLO model.

    This function takes an input image, divides it into smaller tiles based on predefined
    coordinates, and uses a YOLO model to detect fireballs within each tile. Detected fireball
    bounding boxes are then merged and returned.

    Parameters:
        image : np.ndarray
            The input image in which to detect fireballs.
        model : YOLO, optional
            A trained YOLO model to use for detection. If not provided, a default model located at 
            "data/e15.pt" relative to the script's directory will be used.
    
    Returns:
        list[FireballBox]
            A list of FireballBox objects representing the bounding boxes around detected fireballs.
    
    Example:
    ```python
    from fireball_detection.detect import detect_fireballs
    from skimage import io
    image = io.imread('path/to/image.jpg')
    fireballs = detect_fireballs(image)
    for fireball in fireballs:
        print(fireball)
    ```
    """

    if len(image.shape) == 2:
        detected_tiles = detect_differenced_tiles(image, detector, border_size)
    else:
        detected_tiles = detect_standalone_tiles(image, detector, border_size)
    
    fireball_boxes = get_absolute_fireball_boxes(detected_tiles)
    
    detected_fireballs = merge_bboxes(fireball_boxes)
    detected_fireballs = [f for f in detected_fireballs if diagonal_length(f.box) > MIN_DIAGONAL_LENGTH]

    return detected_fireballs


def plot_boxes(image: np.ndarray, fireballs: list[FireballBox]) -> tuple[Figure, Axes]:
    """
    Plots detected fireball bounding boxes on an image.

    Args:
        image (np.ndarray): The input image as a NumPy array on which fireball boxes will be plotted.
        fireballs (list[FireballBox]): A list of FireballBox objects, where each object contains
                                       the bounding box (as a tuple of coordinates) and confidence score.

    Returns:
        tuple[Figure, Axes]: A tuple containing the Matplotlib Figure and Axes objects.
                             The Figure contains the Axes with the plotted image and fireball boxes.
    """

    fig: Figure
    ax: Axes

    fig, ax = plt.subplots()
    ax.imshow(image)

    for fireball in fireballs:
        ax.add_patch(
            patches.Rectangle(
                (fireball.box[0], fireball.box[1]),
                fireball.box[2] - fireball.box[0],
                fireball.box[3] - fireball.box[1],
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            )
        )
        ax.text(
            fireball.box[0],
            fireball.box[1] - 10 if fireball.box[1] > 50 else fireball.box[3] + 25,
            f"{fireball.conf:.2f}",
            color='r',
            fontsize=18,
            va='bottom' if fireball.box[1] > 20 else 'top'
        )

    ax.axis('off')
    fig.tight_layout()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    return fig, ax


def main():
    """
    Main function for loading an image, detecting fireballs, and plotting bounding boxes.

    This function performs the following steps:
    1. Loads a fireball image from a specified path.
    2. Detects fireballs within the image.
    3. Prints the time taken for loading the image and detecting fireballs.
    4. Prints the detected fireball information.
    5. Plots the fireball bounding boxes on the image.
    6. Displays the resulting image with plotted bounding boxes.
    """

    @dataclass
    class Args:
        image_path: str
        model_path: str
        detector: str
        border_size: int
        verbose: bool
        plot: bool

    parser = argparse.ArgumentParser(
        description="Detect fireballs in an image and plot bounding boxes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--image_path", type=str, required=True, help="Path to the fireball image file.")
    parser.add_argument('--detector', type=str, choices=['Ultralytics', 'ONNX'], default='Ultralytics', help='The type of detector to use.')
    parser.add_argument("--model_path", type=str, help="Path to the YOLO model file.")
    parser.add_argument("--border_size", type=str, default=5, help="Border size.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--plot", action="store_true", help="Plot and display the bounding boxes on the image.")
    
    args = Args(**vars(parser.parse_args()))

    if args.verbose:
        print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    detector = get_detector(args.detector, args.model_path)
    if not detector: return

    t0 = time.time()
    image = io.imread(Path(args.image_path))
    t1 = time.time()
    fireballs = detect_fireballs(image, detector, args.border_size)
    t2 = time.time()

    if args.verbose:
        print(f"{'Load Time':<15} {'Detect Time':<15} {'Total Time':<15}")
        print(f"{(t1 - t0):<15.5f} {(t2 - t1):<15.5f} {(t2 - t0):<15.5f}")

    if args.verbose: print("\nFireballs:")
    for fireball in fireballs:
        print(repr(fireball))

    if args.plot:
        plot_boxes(image, fireballs)
        plt.show()


if __name__ == "__main__":
    main()