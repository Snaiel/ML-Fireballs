import os
from pathlib import Path

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tqdm import tqdm
from ultralytics.utils.ops import xywhn2xyxy


with open(Path(Path(__file__).parents[2], "data", "false_negatives.txt")) as file:
    false_negative_fireballs = [i.strip() for i in file.readlines()]

TEST_IMAGES_FOLDER = Path(Path(__file__).parents[2], "data", "kfold_object_detection", "fold0", "images", "val")
TEST_LABELS_FOLDER = Path(Path(__file__).parents[2], "data", "kfold_object_detection", "fold0", "labels", "val")

FALSE_NEGATIVE_IMAGES_FOLDER = Path(Path(__file__).parents[2], "data", "false_negative_images")
if not FALSE_NEGATIVE_IMAGES_FOLDER.exists():
    os.mkdir(FALSE_NEGATIVE_IMAGES_FOLDER)

matplotlib.use("agg")

for fireball in tqdm(false_negative_fireballs, desc="generating plots"):
    image = io.imread(Path(TEST_IMAGES_FOLDER, fireball + ".jpg"))

    with open(Path(TEST_LABELS_FOLDER, fireball + ".txt")) as label_file:
        xyxy = xywhn2xyxy(
            np.array([float(i) for i in label_file.read().split(" ")[1:]]), #xywh
            400,
            400
        )

    fig: Figure
    ax: Axes

    fig, ax = plt.subplots()
    ax.imshow(image)

    ax.add_patch(
        patches.Rectangle(
            (xyxy[0], xyxy[1]),
            xyxy[2] - xyxy[0],
            xyxy[3] - xyxy[1],
            linewidth=1,
            edgecolor='lime',
            facecolor='none'
        )
    )

    ax.axis('off')
    fig.tight_layout()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    fig.savefig(Path(FALSE_NEGATIVE_IMAGES_FOLDER, fireball + ".jpg"), bbox_inches='tight', pad_inches=0, dpi=600)

    plt.cla()
    plt.clf()
    plt.close('all')