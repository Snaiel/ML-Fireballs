import copy
import math
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from skimage import io
from ultralytics import YOLO

from fireball_detection import SQUARE_SIZE
from fireball_detection.detect import FireballBox, Tile, intersects
from fireball_detection.discard.included import retrieve_included_coordinates
from object_detection.utils import add_border


def get_merge_bboxes_iterations(fireballs: list[FireballBox], margin: float = 0.1) -> list[FireballBox]:
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

    iterations = []

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
                if intersects(bmargin, b_margin) or intersects(b_margin, bmargin):
                    tmp_fireball = FireballBox(
                        (
                            min(b[0], b_[0]),
                            min(b[1], b_[1]),
                            max(b[2], b_[2]),
                            max(b[3], b_[3])
                        ),
                        max(fb.conf, fb_.conf)
                    )
                    
                    iterations.append(tmp_fireball)

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

    return iterations


def is_box_in_box(box1, box2):
    return (
        box2[0] <= box1[0] <= box2[2] and
        box2[0] <= box1[2] <= box2[2] and
        box2[1] <= box1[1] <= box2[3] and
        box2[1] <= box1[3] <= box2[3]
    )


# Sample list of rectangle coordinates (top-left corner (x, y), width, height)
coordinates = sorted(retrieve_included_coordinates(), key=lambda x: x[1])
coordinates = [i for i in coordinates if 3700 <= i[0] and i[0] <= 4700 and 2800 <= i[1] and i[1] <= 3900]

# Load the background image
image = io.imread(Path("data/GFO_fireball_object_detection_training_set/jpegs/15_2017-06-12_111729_S_DSC_2102.thumb.jpg"))

model = YOLO(Path(Path(__file__).parents[1], "data", "e15.pt"))

tiles: list[Tile] = []
for pos in coordinates:
    tiles.append(
        Tile(
            pos,
            image[pos[1] : pos[1] + SQUARE_SIZE, pos[0] : pos[0] + SQUARE_SIZE]
        )
    )

detected_tiles: list[Tile] = []
for tile in tiles:
    results = model.predict(
        add_border(tile.image, 1),
        verbose=False
    )
    if len(results[0].boxes.conf) > 0:
        tile.boxes = results[0].boxes
        detected_tiles.append(tile)

detected_fireballs = []
for tile in detected_tiles:
    for box, conf in zip(tile.boxes.xyxy, tile.boxes.conf):
        box = box.cpu()
        detected_fireballs.append(
            (
                tile,
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
        )


fireballs = [i[1] for i in detected_fireballs]
merge_iterations = get_merge_bboxes_iterations(fireballs)


ax: Axes

# Create the plot
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')

ax.axis('off')
fig.tight_layout()
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

# List to keep track of rectangles
stuff = []

# Initial rectangle
initial_rect = patches.Rectangle((coordinates[0][0], coordinates[0][1]), 
                         SQUARE_SIZE, SQUARE_SIZE, 
                         linewidth=2, edgecolor='r', facecolor='none')

stuff.append(initial_rect)
ax.add_patch(initial_rect)

confidences_text = ax.text(2000, 1000, "box confidences:", color="red", va="top")
stuff.append(confidences_text)

rects = []

confidences = []

# Update function for animation
def update(frame):
    global confidences

    if frame == len(coordinates):
        initial_rect.remove()
        return stuff

    if frame < len(coordinates):
        x, y = coordinates[frame]
        initial_rect.set_xy((x, y))

        for detected in detected_fireballs:
            # print(coordinates[frame], detected[0])
            tile_pos = detected[0].position
            if not coordinates[frame] == tile_pos:
                continue
            fireball_box = detected[1].box
            new_rect = patches.Rectangle(
                (fireball_box[0], fireball_box[1]),
                fireball_box[2] - fireball_box[0],
                fireball_box[3] - fireball_box[1],
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            )
            stuff.append(new_rect)
            rects.append(new_rect)
            ax.add_patch(new_rect)

            confidences.append((new_rect, detected[1]))

    if frame > len(coordinates) + 4 and frame < len(coordinates) + 5 + len(merge_iterations):
        merged_fireball = merge_iterations[frame - len(coordinates) - 5]

        fireball_box = merged_fireball.box
        new_rect = patches.Rectangle(
            (fireball_box[0], fireball_box[1]),
            fireball_box[2] - fireball_box[0],
            fireball_box[3] - fireball_box[1],
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )

        stuff.append(new_rect)
        ax.add_patch(new_rect)

        for rect in rects:
            rect: patches.Rectangle
            if is_box_in_box(
                [rect.get_x(), rect.get_y(), rect.get_x() + rect.get_width(), rect.get_y() + rect.get_height()],
                merged_fireball.box
            ):
                try:
                    confidences = [i for i in confidences if i[0] != rect and not math.isclose(i[1].conf, merge_iterations[-1].conf)]
                    rect.remove()
                    rects.remove(rect)
                except ValueError:
                    pass
        
        rects.append(new_rect)
        confidences.append((new_rect, merged_fireball))

    confidences_text.set_text(
        "box confidences:\n" + "\n".join([f"{i[1].conf:.2f}" for i in confidences])
    )

    return stuff


# Create animation
ani = FuncAnimation(
    fig,
    update,
    frames=len(coordinates) + 5 + len(merge_iterations) + 20,
    interval=200,
    blit=True,
    repeat=False,
)


ani.save(
    "anim_tile_detections.gif",
    progress_callback = lambda i, n: print(f'Saving frame {i}/{n}'),
    savefig_kwargs={'pad_inches': 0},
    dpi=200
)

# Show the plot
# plt.show()
