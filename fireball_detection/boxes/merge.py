import copy
from fireball_detection import FireballBox


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


def find_groups_of_intersecting_boxes(boxes: list[FireballBox]) -> list[list[FireballBox]]:
    groups = []  # List of sets

    for box in boxes:
        merged_sets = []  # Keep track of sets that need to be merged
        new_set = {box}  # Start with a new set containing only this box

        for group in groups:
            group: set[FireballBox]
            if any(intersects(existing_box.box, box.box) for existing_box in group):
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
