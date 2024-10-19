from dataclasses import dataclass

from point_pickings.core import DE_BRUIJN_SEQUENCE
from point_pickings.core.sequence import FireballAlignment
from point_pickings.core.blobs import FireballBlobs


@dataclass
class FireballPoint:
    x: float
    y: float
    de_bruijn_pos: int
    de_bruijn_val: str


def backtrack_odd_number_of_1s(fireball_points: list[FireballPoint], de_bruijn_pos: int) -> None:
    """
    Every group of small gaps should have an even number of 1 values.
    In total a group of small gaps should have an odd number of nodes.
    The last node would have a big gap which would make it a 0.

    For example (going from left to right):

    `.   . . . . . . .   .`            \n
    `0   1 1 1 1 1 1 0   ?`

    6 '1s' (even) + 1 '0' = 7 (odd)


    Usually at the ends of the fireball, the algorithm does not get the
    correct number of 1s. This function is called when you assume there
    will be two 1s to label but you only label one and come across a 0
    as the next node.

    This function backtracks and labels the 1s from this point so that
    the unpaired 1 is at the beginning of the small gap group.

    NOTE: Needs more consideration for what happens in the middle of the
    fireball. Maybe count the number of '1' nodes in the small gap group
    and check distances for missing nodes?
    """

    # the current point we are going to change
    current_point = fireball_points[-1]
    # the position of the current point
    current_pos = len(fireball_points) - 1
    # how many points have we assigned a de bruijn position.
    # resets when it reaches two to start counting for the next
    # de bruijn position
    current_de_bruijn_pos_count = 0
    # the current de bruijn position we will assign to the
    # currrent node
    backtrack_de_bruijn_pos = de_bruijn_pos

    while current_pos >= 0 and current_point.de_bruijn_val == '1':
        # assign de bruijn position to point
        current_point.de_bruijn_pos = backtrack_de_bruijn_pos
        current_de_bruijn_pos_count += 1

        # paired labelled, move on to previous de bruijn position
        if current_de_bruijn_pos_count == 2:
            current_de_bruijn_pos_count = 0
            backtrack_de_bruijn_pos -= 1
        
        # go to previous fireball point
        current_pos -= 1
        current_point = fireball_points[current_pos]


def assign_labels_to_blobs(fireball_blobs: FireballBlobs, distance_labels: list[int], alignment: FireballAlignment) -> list[FireballPoint]:
    """
        Assigns de Bruijn positions and labels to the fireball blobs
        using the distance labels and alignment.

        Mistakes found in the alignment result in offending blobs being discarded.

        ### Parameters
        | Name            | Type             | Description                                |
        |-----------------|------------------|--------------------------------------------|
        | fireball_blobs  | FireballBlobs    | List of fireball blobs (x, y, r).          |
        | distance_labels | list[int]        | List of distance labels 1s or 0s.          |
        | alignment       | FireballAlignment| Alignment data for the fireball.           |

        ### Returns
        | Type                  | Description                                 |
        |-----------------------|---------------------------------------------|
        | list[FireballPoint]   | List of fireball points (x, y, pos, val).   |
    """
    
    fireball_points: list[FireballPoint] = []
    fireball_blobs_queue = list(fireball_blobs.copy())
    fireball_labels_queue = list(distance_labels.copy())

    def consume_node(de_bruijn_pos: int, de_bruijn_val: str):
        node = fireball_blobs_queue.pop(0)
        fireball_points.append(
            FireballPoint(
                node[0],
                node[1],
                de_bruijn_pos,
                de_bruijn_val
            )
        )
        fireball_labels_queue.pop(0)
    
    def discard_node():
        fireball_blobs_queue.pop(0)
        fireball_labels_queue.pop(0)

    de_bruijn_start_index = alignment.start_index
    de_bruijn_segment = alignment.de_bruijn_segment
    fireball_sequence = alignment.fireball_sequence

    # Assign labels to nodes
    for de_bruijn_pos, de_bruijn_val, fireball_val in zip(
        range(
            de_bruijn_start_index,
            de_bruijn_start_index + len(de_bruijn_segment)
        ), 
        de_bruijn_segment,
        fireball_sequence
    ):

        if len(fireball_labels_queue) == 0:
            break

        if de_bruijn_val == "0" and fireball_val == "0":
            consume_node(de_bruijn_pos, de_bruijn_val)
        elif de_bruijn_val == "1" and fireball_val == "1":
            # first blob in pair
            consume_node(de_bruijn_pos, de_bruijn_val)
            if len(fireball_labels_queue) == 0:
                break
            
            # check for second blob in pair
            if fireball_labels_queue[0] == 1:
                consume_node(de_bruijn_pos, de_bruijn_val)
            else:
                # the next distance label was not 1, shift labels
                backtrack_odd_number_of_1s(fireball_points, de_bruijn_pos)

        elif de_bruijn_val == "0" and fireball_val == "1":
            # Two false positives, ignore both
            discard_node()
            discard_node()
        elif de_bruijn_val == "1" and fireball_val == "0":
            # Missing blob, ignore the singular blob without partner
            discard_node()
        elif de_bruijn_val == "0" and fireball_val == "-":
            pass
        elif de_bruijn_val == "1" and fireball_val == "-":
            pass


    # Assign label to the last fireball node
    last_node_de_bruijn_pos = -1

    last_point, second_last_point = fireball_points[-1], fireball_points[-2]
    if last_point.de_bruijn_val == '0':
        # last point had de Bruijn valie of '0', allow anything to come next
        last_node_de_bruijn_pos = alignment.start_index + len(alignment.de_bruijn_segment) # next de bruijn
    else: # the last point had de Bruijn value of '1'
        if second_last_point.de_bruijn_val == '0':
            # second last point was '0', this new one should be a '1'
            last_node_de_bruijn_pos = alignment.start_index + len(alignment.de_bruijn_segment) - 1 # current de bruijn
        else: # second last point was '1'
            if last_point.de_bruijn_pos == second_last_point.de_bruijn_pos:
                # last and second last were a pair
                last_node_de_bruijn_pos = alignment.start_index + len(alignment.de_bruijn_segment) # next de bruijn
            else:
                # become a pair with the last point
                last_node_de_bruijn_pos = alignment.start_index + len(alignment.de_bruijn_segment) - 1 # current de bruijn
    
    last_node = fireball_blobs_queue.pop(0)
    fireball_points.append(
        FireballPoint(
            last_node[0],
            last_node[1],
            last_node_de_bruijn_pos,
            DE_BRUIJN_SEQUENCE[last_node_de_bruijn_pos]
        )
    )

    print("Final Fireball Points:")
    print("[x, y, de bruijn pos, 0 or 1]")
    for i in fireball_points:
        print(i)
    

    return fireball_points