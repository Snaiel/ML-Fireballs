import time
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from skimage import io
from ultralytics import YOLO

from fireball_detection import SQUARE_SIZE, Tile, FireballBox
from fireball_detection.discard.included import retrieve_included_coordinates
from fireball_detection.boxes.fireball_boxes import get_absolute_fireball_boxes
from object_detection.utils import add_border
from fireball_detection.boxes.merge import merge_bboxes


INCLUDED_COORDINATES = retrieve_included_coordinates()


def detect_tiles(image: ndarray, model: YOLO | None = None, border_size: int = 0) -> list[Tile]:
    """
    Detects and returns a list of tiles containing detected objects from an image.

    This function tiles an input image and runs YOLO model on them. Detected tiles 
    contain bounding boxes and confidence scores for each detected fireball.

    Args:
        image (ndarray): The input image to process.
        model (YOLO | None, optional): A pre-trained YOLO model used for detection. 
            If None, a default model is loaded from "data/e15.pt".
        border_size (int, optional): The width of the border to add around each tile
            before detection. Defaults to 0, meaning no border is added.

    Returns:
        list[Tile]: A list of Tile objects that contain detected objects.
    """

    if model is None:
        model = YOLO(Path(Path(__file__).parents[1], "data", "e15.pt"))
    
    tiles: list[Tile] = []
    for pos in INCLUDED_COORDINATES:
        tiles.append(
            Tile(
                pos,
                image[pos[1] : pos[1] + SQUARE_SIZE, pos[0] : pos[0] + SQUARE_SIZE]
            )
        )

    detected_tiles: list[Tile] = []
    for tile in tiles:
        input_image = tile.image if border_size == 0 else add_border(tile.image, border_size)
        results = model.predict(
            input_image,
            verbose=False
        )
        result = results[0]
        if len(result.boxes.conf) > 0:
            tile.boxes = list(result.boxes.xyxy.cpu())
            tile.confidences = list(result.boxes.conf.cpu())
            detected_tiles.append(tile)
    
    return detected_tiles


def detect_fireballs(image: ndarray, model: YOLO | None = None, border_size: int = 0) -> list[FireballBox]:
    """
    Detects fireballs within an image using a YOLO model.

    This function takes an input image, divides it into smaller tiles based on predefined
    coordinates, and uses a YOLO model to detect fireballs within each tile. Detected fireball
    bounding boxes are then merged and returned.

    Parameters:
        image : ndarray
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

    detected_tiles = detect_tiles(image, model, border_size)
    fireball_boxes = get_absolute_fireball_boxes(detected_tiles)
    detected_fireballs = merge_bboxes(fireball_boxes)

    return detected_fireballs


def plot_boxes(image: ndarray, fireballs: list[FireballBox]) -> tuple[Figure, Axes]:
    """
    Plots detected fireball bounding boxes on an image.

    Args:
        image (ndarray): The input image as a NumPy array on which fireball boxes will be plotted.
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
                linewidth=4,
                edgecolor='r',
                facecolor='none'
            )
        )
        ax.text(
            fireball.box[0],
            fireball.box[1] - 10 if fireball.box[1] > 20 else fireball.box[3] + 25,
            f"{fireball.conf:.2f}",
            color='r',
            fontsize=24,
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

    fireball_image = "data/GFO_fireball_object_detection_training_set/jpegs/57_2016-06-17_182058_S_DSC_1189.thumb.jpg"

    t0 = time.time()
    image = io.imread(Path(Path(__file__).parents[1], fireball_image))
    t1 = time.time()
    fireballs = detect_fireballs(image)
    t2 = time.time()

    print(f"{'Load Time':<15} {'Detect Time':<15} {'Total Time':<15}")
    print(f"{(t1 - t0):<15.5f} {(t2 - t1):<15.5f} {(t2 - t0):<15.5f}")


    print("\nFireballs:")
    for fireball in fireballs:
        print(fireball)

    fig, ax = plot_boxes(image, fireballs)
    plt.tight_layout()
    # fig.savefig("detect.png", bbox_inches='tight', pad_inches=0, dpi=600)
    plt.show()


if __name__ == "__main__":
    main()