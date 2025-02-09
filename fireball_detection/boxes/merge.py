import copy
from typing import Callable

from fireball_detection import FireballBox
from object_detection.utils import iom


def intersects(bbox: tuple[float, float, float, float], bbox_: tuple[float, float, float, float]):
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


def find_groups_of_boxes(
    boxes: list[FireballBox], 
    criteria: Callable[[FireballBox, FireballBox], bool] # function that takes two boxes and returns whether to group them
) -> list[list[FireballBox]]:
    groups = []  # List of sets

    for box in boxes:
        merged_sets = []  # Keep track of sets that need to be merged
        new_set = {box}  # Start with a new set containing only this box

        for group in groups:
            group: set[FireballBox]
            if any(criteria(existing_box, box) for existing_box in group):
                merged_sets.append(group)

        # Merge all found sets with the new set
        for group in merged_sets:
            new_set.update(group)
            groups.remove(group)  # Remove merged sets

        # Add the new or merged set to the list of groups
        groups.append(new_set)

    return groups


def smallest_enclosing_box(fboxes: list[FireballBox]) -> tuple:
    if not fboxes:
        return None

    min_x1 = min(fbox.box[0] for fbox in fboxes)
    min_y1 = min(fbox.box[1] for fbox in fboxes)
    max_x2 = max(fbox.box[2] for fbox in fboxes)
    max_y2 = max(fbox.box[3] for fbox in fboxes)

    return (min_x1, min_y1, max_x2, max_y2)


def merge_groups_of_boxes(
    fboxes: list[FireballBox],
    criteria: Callable[[FireballBox, FireballBox], bool]
) -> list[FireballBox]:

    groups = find_groups_of_boxes(fboxes, criteria)

    merged_fireball_boxes: list[FireballBox] = []

    for group in groups:
        max_conf = max(fbox.conf for fbox in group)
        new_box = smallest_enclosing_box(group)
        merged_fireball_boxes.append(FireballBox(new_box, max_conf))
    
    return merged_fireball_boxes


def main():
    tile_detections = [
        {
        "position": [
            3400,
            3000
        ],
        "detections": [
            {
            "box": [
                361.0283508300781,
                48.08544921875,
                394.4242248535156,
                350.9884033203125
            ],
            "confidence": 0.34214717149734497
            }
        ]
        },
        {
        "position": [
            3400,
            3200
        ],
        "detections": [
            {
            "box": [
                369.6860656738281,
                5.819679260253906,
                392.6781311035156,
                138.92807006835938
            ],
            "confidence": 0.376398503780365
            }
        ]
        },
        {
        "position": [
            3600,
            2400
        ],
        "detections": [
            {
            "box": [
                22.907333374023438,
                221.24472045898438,
                228.42678833007812,
                408.5757751464844
            ],
            "confidence": 0.3854653239250183
            }
        ]
        },
        {
        "position": [
            3600,
            2600
        ],
        "detections": [
            {
            "box": [
                26.669387817382812,
                15.78814697265625,
                396.71636962890625,
                368.7706298828125
            ],
            "confidence": 0.2996208369731903
            }
        ]
        },
        {
        "position": [
            3600,
            2800
        ],
        "detections": [
            {
            "box": [
                219.5765380859375,
                1.7720108032226562,
                406.71514892578125,
                176.25967407226562
            ],
            "confidence": 0.4078142046928406
            },
            {
            "box": [
                150.46495056152344,
                162.17794799804688,
                182.46815490722656,
                406.7033386230469
            ],
            "confidence": 0.25499874353408813
            }
        ]
        },
        {
        "position": [
            3600,
            3000
        ],
        "detections": [
            {
            "box": [
                160.71487426757812,
                48.92567443847656,
                193.05490112304688,
                350.05224609375
            ],
            "confidence": 0.3448764383792877
            }
        ]
        },
        {
        "position": [
            3600,
            3200
        ],
        "detections": [
            {
            "box": [
                171.30772399902344,
                4.4071197509765625,
                189.53492736816406,
                134.95230102539062
            ],
            "confidence": 0.38606521487236023
            }
        ]
        },
        {
        "position": [
            3800,
            2600
        ],
        "detections": [
            {
            "box": [
                1.2056198120117188,
                179.95977783203125,
                251.9345703125,
                408.9610595703125
            ],
            "confidence": 0.39970993995666504
            }
        ]
        },
        {
        "position": [
            3800,
            2800
        ],
        "detections": [
            {
            "box": [
                39.06144714355469,
                16.359619140625,
                388.118408203125,
                322.55523681640625
            ],
            "confidence": 0.35804086923599243
            }
        ]
        }
    ]

    fireball_boxes = []

    for t in tile_detections:
        x, y = t["position"]
        for d in t["detections"]:
            x1, y1, x2, y2 = d["box"]
            conf = d["confidence"]
            fireball_boxes.append(
                FireballBox(
                    (
                        x + x1,
                        y + y1,
                        x + x2,
                        y + y2
                    ),
                    conf
                )
            )
    
    for i in fireball_boxes:
        print(i)

    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np
    import skimage.io as io
    from matplotlib.axes import Axes

    def plot_boxes(ax: Axes, image: np.ndarray, boxes: list[FireballBox], title: str) -> None:
        ax.imshow(image, cmap="gray", aspect="equal")
        for fbox in boxes:
            x1, y1, x2, y2 = fbox.box
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)

    new_fireball_boxes = merge_groups_of_boxes(
        fireball_boxes,
        lambda x, y: iom(x.box, y.box) >= 0.25
    )

    print()
    for fbox in new_fireball_boxes:
        print(fbox)

    image = io.imread("data/dfn-2015-01-01/dfn-2015-01/dfn-l0-20150101/DFNSMALL27/27_2015-01-01_184757_DSC_0935/27_2015-01-01_184757_DSC_0935.thumb.jpg")
    
    ax1: Axes
    ax2: Axes

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)
    plot_boxes(ax1, image, fireball_boxes, "Unmerged Fireball Detections")
    plot_boxes(ax2, image, new_fireball_boxes, "Merged Fireball Detections")

    ax1.set_xlim(0, image.shape[1])
    ax1.set_ylim(image.shape[0], 0)
    ax2.set_xlim(0, image.shape[1])
    ax2.set_ylim(image.shape[0], 0)

    plt.show()


if __name__ == "__main__":
    main()