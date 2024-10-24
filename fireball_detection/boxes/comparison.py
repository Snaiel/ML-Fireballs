"""
script to compare between intersecting, nms, wbf
for merging bounding boxes.

python3 -m fireball_detection.boxes.comparison
"""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from numpy import ndarray
from skimage import io

from fireball_detection import IMAGE_DIMENSIONS, FireballBox
from fireball_detection.boxes.fireball_boxes import (
    get_absolute_fireball_boxes, get_normalised_fireball_boxes)
from fireball_detection.boxes.merge import merge_bboxes
from fireball_detection.boxes.nms import nms
from fireball_detection.boxes.wbf import weighted_boxes_fusion
from fireball_detection.detect import detect_tiles


def _merge_boxes(method: callable, noramlised_fireball_boxes: list[FireballBox], threshold: float) -> list[FireballBox]:
    final_boxes, final_scores, _ = method(
        [[i.box for i in noramlised_fireball_boxes]],
        [[i.conf for i in noramlised_fireball_boxes]],
        [[0 for _ in range(len(noramlised_fireball_boxes))]],
        iou_thr=threshold
    )

    return [
        FireballBox(
            (
                box[0] * IMAGE_DIMENSIONS[0],
                box[1] * IMAGE_DIMENSIONS[1],
                box[2] * IMAGE_DIMENSIONS[0],
                box[3] * IMAGE_DIMENSIONS[1]
            ),
            score
        ) for box, score in zip(final_boxes, final_scores)
    ]


def merge_with_wbf(noramlised_fireball_boxes: list[FireballBox], threshold: float = 0.25) -> list[FireballBox]:
    return _merge_boxes(weighted_boxes_fusion, noramlised_fireball_boxes, threshold)


def merge_with_nms(noramlised_fireball_boxes: list[FireballBox], threshold: float = 0.25) -> list[FireballBox]:
    return _merge_boxes(nms, noramlised_fireball_boxes, threshold)


def plot_boxes_on_image(ax: Axes, image: ndarray, fireball_boxes: list[FireballBox], color: str, title: str) -> None:
    ax.imshow(image)
    for fireball in fireball_boxes:
        ax.add_patch(
            patches.Rectangle(
                (fireball.box[0], fireball.box[1]),
                fireball.box[2] - fireball.box[0],
                fireball.box[3] - fireball.box[1],
                linewidth=4,
                edgecolor=color,
                facecolor='none'
            )
        )
        ax.text(
            fireball.box[0],
            fireball.box[1] - 10 if fireball.box[1] > 20 else fireball.box[3] + 25,
            f"{fireball.conf:.2f}",
            color=color,
            fontsize=24,
            va='bottom' if fireball.box[1] > 20 else 'top'
        )

    ax.set_title(title)
    ax.axis('off')
    ax.set_xlim(4380, 5380)
    ax.set_ylim(2160, 1160)


def main() -> None:
    fireball_image = "data/GFO_fireball_object_detection_training_set/jpegs/50_2017-02-22_172331_S_DSC_7698.thumb.jpg"

    image = io.imread(Path(Path(__file__).parents[2], fireball_image))
    detected_tiles = detect_tiles(image, border_size=5)

    absolute_fireball_boxes = get_absolute_fireball_boxes(detected_tiles)
    normalised_fireball_boxes = get_normalised_fireball_boxes(detected_tiles)

    intersect_fireball_boxes = merge_bboxes(absolute_fireball_boxes)
    
    threshold = 0.25
    nms_fireball_boxes = merge_with_nms(normalised_fireball_boxes, threshold)
    wbf_fireball_boxes = merge_with_wbf(normalised_fireball_boxes, threshold)

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    plot_boxes_on_image(axs[0, 0], image, absolute_fireball_boxes, 'r', 'Boxes')
    plot_boxes_on_image(axs[0, 1], image, intersect_fireball_boxes, 'g', 'Intersect Boxes')
    plot_boxes_on_image(axs[1, 0], image, nms_fireball_boxes, 'b', f'NMS Boxes IoU≥{threshold}')
    plot_boxes_on_image(axs[1, 1], image, wbf_fireball_boxes, 'y', f'WBF Boxes IoU≥{threshold}')

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()