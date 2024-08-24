import copy
import time
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from skimage import io
from ultralytics import YOLO
from ultralytics.engine.results import Boxes as YOLOBoxes

from fireball_detection import SQUARE_SIZE
from fireball_detection.included import retrieve_included_coordinates


class Tile:
    position: tuple[float, float] = None
    image: ndarray = None
    boxes: YOLOBoxes = None

    def __init__(self, position: tuple[float, float], image: ndarray) -> None:
        self.position = position
        self.image = image


class FireballBox:
    box: tuple[float, float, float, float] #xyxy
    conf: float

    def __init__(self, box: tuple[float, float, float, float], conf: float) -> None:
        self.box = box
        self.conf = conf
    
    def __repr__(self) -> str:
        return f"{self.conf} {' '.join(map(str, self.box))}"
    
    def __str__(self) -> str:
        return f"<{self.conf:.2f} ({', '.join(f'{i:.2f}' for i in self.box)})>"


INCLUDED_COORDINATES = retrieve_included_coordinates()


def intersect(bbox: tuple[float, float, float, float], bbox_: tuple[float, float, float, float]):
    """
    https://stackoverflow.com/questions/40795709/checking-whether-two-rectangles-overlap-in-python-using-two-bottom-left-corners

    Arguments:
        - bbox | list | bounding box of float_values [xmin, ymin, xmax, ymax]
        - bbox_ | list | bounding box of float_values [xmin, ymin, xmax, ymax]
    
    Returns:
        - boolean | true if the bboxes intersect
    """
    return not (
        bbox[0] > bbox_[2]
        or bbox[2] < bbox_[0]
        or bbox[1] > bbox_[3]
        or bbox[3] < bbox_[1]
    )


def merge_bboxes(fireballs: list[FireballBox], margin: float = 0.1) -> list[FireballBox]:
    """
    https://gist.github.com/YaYaB/39f9df9d481d784b786ad88eea8533e8

    Combines intersecting boxes, taking the maximum confidence

    Arguments:
        - fireballs | list | list of FireballBox objects
        - margin | float | margin taken in width to merge
    
    Returns:
        - list[FireballBox] | list of merged fireballs
    """

    # Sort fireballs by ymin
    fireballs = sorted(fireballs, key=lambda x: x.box[1])

    tmp_fireball = None
    while True:
        nb_merge = 0
        used = [] # a list of indexes that have already been considered
        new_fireballs: list[FireballBox] = []
        # Loop over fireballs
        for i, fb in enumerate(fireballs):
            for j, fb_ in enumerate(fireballs):
                # If the bbox has already been used just continue
                if i in used or j <= i:
                    continue
                
                # Compute the fireballs with a margin
                b = fb.box
                b_ = fb_.box
                bmargin = [
                    b[0] - (b[2] - b[0]) * margin,
                    b[1] - (b[3] - b[1]) * margin,
                    b[2] + (b[2] - b[0]) * margin,
                    b[3] + (b[3] - b[1]) * margin
                ]
                b_margin = [
                    b_[0] - (b_[2] - b_[0]) * margin,
                    b_[1] - (b_[3] - b_[1]) * margin,
                    b_[2] + (b_[2] - b_[0]) * margin,
                    b_[3] + (b_[3] - b_[1]) * margin
                ]
                
                # Merge fireballs if fireballs with margin have an intersection
                # Check if one of the corner is in the other bbox
                # We must verify the other side away in case one bounding box is inside the other
                if intersect(bmargin, b_margin) or intersect(b_margin, bmargin):
                    tmp_fireball = FireballBox(
                        (
                            min(b[0], b_[0]),
                            min(b[1], b_[1]),
                            max(b[2], b_[2]),
                            max(b[3], b_[3])
                        ),
                        max(fb.conf, fb_.conf)
                    )
                    used.append(j)
                    nb_merge += 1
                
                if tmp_fireball:
                    fb = tmp_fireball
            
            if tmp_fireball:
                new_fireballs.append(tmp_fireball)
            elif i not in used:
                new_fireballs.append(fb)
            
            used.append(i)
            tmp_fireball = None
        
        # If no merge were done, that means all bboxes were already merged
        if nb_merge == 0:
            break
        
        fireballs = copy.deepcopy(new_fireballs)

    return new_fireballs


def detect_fireballs(image: ndarray, model: YOLO | None = None) -> list[FireballBox]:
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
        results = model.predict(tile.image, verbose=False)
        if len(results[0].boxes.conf) > 0:
            tile.boxes = results[0].boxes
            detected_tiles.append(tile)

    detected_fireballs = []
    for tile in detected_tiles:
        for box, conf in zip(tile.boxes.xyxy, tile.boxes.conf):
            box = box.cpu()
            detected_fireballs.append(
                FireballBox(
                    (
                        float(tile.position[0] + box[0]),
                        float(tile.position[1] + box[1]),
                        float(tile.position[0] + box[2]),
                        float(tile.position[1] + box[3])
                    ),
                    conf.cpu()
                )
            )

    detected_fireballs = merge_bboxes(detected_fireballs)
    return detected_fireballs


def plot_boxes(image: ndarray, fireballs: list[FireballBox]) -> tuple[Figure, Axes]:
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
            fireball.box[1] - 10 if fireball.box[1] > 20 else fireball.box[3] + 25,
            f"{fireball.conf:.2f}",
            color='r',
            fontsize=8,
            va='bottom' if fireball.box[1] > 20 else 'top'
        )

    ax.axis('off')
    fig.tight_layout()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    return fig, ax


def main():
    fireball_image = "data/GFO_fireball_object_detection_training_set/jpegs/03_2020-02-27_131328_K_DSC_4670.thumb.jpg"

    t0 = time.time()
    image = io.imread(Path(Path(__file__).parents[1], fireball_image))
    t1 = time.time()
    fireballs = detect_fireballs(image)
    t2 = time.time()

    print("Load Image, Detect, Total")
    print(t1 - t0, t2 - t1, t2 - t0)

    for fireball in fireballs:
        print(fireball)

    fig, ax = plot_boxes(image, fireballs)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.tight_layout()
    # fig.savefig("yeah.png", bbox_inches='tight', pad_inches=0, dpi=600)
    plt.show()


if __name__ == "__main__":
    main()