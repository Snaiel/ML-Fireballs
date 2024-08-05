import copy
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from numpy import ndarray
from skimage import io
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

from fireball_detection import SQUARE_SIZE
from fireball_detection.included import retrieve_included_coordinates


class Tile:
    position: tuple[float, float] = None
    image: ndarray = None
    boxes: Boxes = None

    def __init__(self, position: tuple[float, float], image: ndarray) -> None:
        self.position = position
        self.image = image


included_coordinates = retrieve_included_coordinates()
img = io.imread(Path(Path(__file__).parents[1], "data/GFO_fireball_object_detection_training_set/jpegs/005_2019-06-01_104929_E_DSC_0119.thumb.jpg"))

tiles: list[Tile] = []

for pos in included_coordinates:
    tiles.append(
        Tile(
            pos,
            img[pos[1] : pos[1] + SQUARE_SIZE, pos[0] : pos[0] + SQUARE_SIZE]
        )
    )


model = YOLO(Path(Path(__file__).parents[1], "data", "e15.pt"))

detected_tiles: list[Tile] = []

for tile in tiles:
    results = model(tile.image)
    if len(results[0].boxes.conf) > 0:
        tile.boxes = results[0].boxes
        detected_tiles.append(tile)

boxes = []

for tile in detected_tiles:
    for box in tile.boxes.xyxy:
        box = box.cpu()
        boxes.append(
            [
                tile.position[0] + box[0],
                tile.position[1] + box[1],
                tile.position[0] + box[2],
                tile.position[1] + box[3]
            ]
        )


def merge_bboxes(bboxes, delta_x=0.1, delta_y=0.1):
    """
    https://gist.github.com/YaYaB/39f9df9d481d784b786ad88eea8533e8

    Arguments:
        bboxes {list} -- list of bounding boxes with each bounding box is a list [xmin, ymin, xmax, ymax]
        delta_x {float} -- margin taken in width to merge
        detlta_y {float} -- margin taken in height to merge
    Returns:
        {list} -- list of bounding boxes merged
    """

    def intersect(bbox, bbox_):
        """
        Arguments:
            bbox {list} -- bounding box of float_values [xmin, ymin, xmax, ymax]
            bbox_ {list} -- bounding box of float_values [xmin, ymin, xmax, ymax]
        Returns:
            {boolean} -- true if the bboxes intersect
        """
        # https://stackoverflow.com/questions/40795709/checking-whether-two-rectangles-overlap-in-python-using-two-bottom-left-corners
        return not (
            bbox[0] > bbox_[2]
            or bbox[2] < bbox_[0]
            or bbox[1] > bbox_[3]
            or bbox[3] < bbox_[1]
        )

    # Sort bboxes by ymin
    bboxes = sorted(bboxes, key=lambda x: x[1])

    tmp_bbox = None
    while True:
        nb_merge = 0
        used = []
        new_bboxes = []
        # Loop over bboxes
        for i, b in enumerate(bboxes):
            for j, b_ in enumerate(bboxes):
                # If the bbox has already been used just continue
                if i in used or j <= i:
                    continue
                # Compute the bboxes with a margin
                bmargin = [
                    b[0] - (b[2] - b[0]) * delta_x, b[1] - (b[3] - b[1]) * delta_y,
                    b[2] + (b[2] - b[0]) * delta_x, b[3] + (b[3] - b[1]) * delta_y
                ]
                b_margin = [
                    b_[0] - (b_[2] - b_[0]) * delta_x, b_[1] - (b[3] - b[1]) * delta_y,
                    b_[2] + (b_[2] - b_[0]) * delta_x, b_[3] + (b_[3] - b_[1]) * delta_y
                ]
                # Merge bboxes if bboxes with margin have an intersection
                # Check if one of the corner is in the other bbox
                # We must verify the other side away in case one bounding box is inside the other
                if intersect(bmargin, b_margin) or intersect(b_margin, bmargin):
                    tmp_bbox = [min(b[0], b_[0]), min(b[1], b_[1]), max(b_[2], b[2]), max(b[3], b_[3])]
                    used.append(j)
                    # print(bmargin, b_margin, 'done')
                    nb_merge += 1
                if tmp_bbox:
                    b = tmp_bbox
            if tmp_bbox:
                new_bboxes.append(tmp_bbox)
            elif i not in used:
                new_bboxes.append(b)
            used.append(i)
            tmp_bbox = None
        # If no merge were done, that means all bboxes were already merged
        if nb_merge == 0:
            break
        bboxes = copy.deepcopy(new_bboxes)

    return new_bboxes


boxes = merge_bboxes(boxes)

fig, ax = plt.subplots()
ax.imshow(img)

for box in boxes:
    rect = patches.Rectangle(
        (box[0], box[1]),
        box[2] - box[0],
        box[3] - box[1],
        linewidth=2,
        edgecolor='r',
        facecolor='none'
    )
    ax.add_patch(rect)

plt.tight_layout()
plt.show()