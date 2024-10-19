import Levenshtein
from Bio import Align
from point_pickings.core import DE_BRUIJN_SEQUENCE
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class FireballAlignment:
    left_to_right: bool
    levenshtein_ratio: float
    start_index: int
    de_bruijn_segment: str
    fireball_sequence: str


def get_left_to_right_sequence_from_distance_labels(distance_labels: list[int]) -> str:
    """
        Creates a left-to-right sequence from the provided distance labels.
        Applies the DFN encoding system where two 1s equal a singular 1. One zero is a singular zero.

        ### Parameters
        | Name             | Type       | Description                                  |
        |------------------|------------|----------------------------------------------|
        | distance_labels  | list[int]  | The 1 or 0 label for the distances.          |

        ### Returns
        | Type             | Description                                                             |
        |------------------|-------------------------------------------------------------------------|
        | str              | The left-to-right sequence of 1s or 0s made from the distance labels.   |
    """
    lr_sequence = ""
    skip_value = False

    for i in distance_labels:
        if i == 0:
            lr_sequence += str(i)
            skip_value = False
        elif i == 1:
            if skip_value:
                skip_value = False
            else:
                lr_sequence += str(i)
                skip_value = True

    return lr_sequence


def cyclic_slice(string: str, start: int, end: int) -> str:
    """
        Slices the input string cyclically, meaning it wraps around if the end index is before the start index.

        ### Parameters
        | Name   | Type     | Description                                      |
        |--------|----------|--------------------------------------------------|
        | string | str      | The input string to be cyclically sliced.        |
        | start  | int      | The starting index of the slice.                 |
        | end    | int      | The ending index of the slice.                   |

        ### Returns
        | Type   | Description                                          |
        |--------|------------------------------------------------------|
        | str    | The cyclically sliced substring of the input string. |
    """
    length = len(string)
    if length == 0:
        return ''

    # Adjust start and end indices to be within the length of the string
    start %= length
    end %= length

    # If start is negative, adjust it to count from the end of the string
    if start < 0:
        start += length

    # If start comes after end due to modulo, adjust end
    if start > end:
        return string[start:] + string[:end]
    else:
        return string[start:end]


def perform_alignment(fireball_sequence: str, left_to_right: bool) -> FireballAlignment:
    """
        ### Description
        Attempts to align the given sequence to the de Bruijn sequence

        ### Parameters
        | Name               | Type     | Description                                       |
        |--------------------|----------|---------------------------------------------------|
        | fireball_sequence  | str      | The sequence of 1s and 0s that represent the      |
        |                    |          | fireball as a segment of the de Bruijn sequence.  |
        | left_to_right      | bool     | Boolean indicating whether the sequence goes      |
        |                    |          | from left-to-right or not.                        |

        ### Returns
        | Type               | Description                                        |
        |--------------------|----------------------------------------------------|
        | FireballAlignment  | An object representing the alignment of fireballs. |
    """
    aligner = Align.PairwiseAligner()

    # Higher match score encourages aligning similar characters.
    aligner.match_score = 2

    # Higher negative mismatch score discourages aligning dissimilar characters.
    aligner.mismatch_score = -3

    # Higher negative open gap score discourages introducing gaps.
    aligner.open_gap_score = -5

    # Higher negative extend gap score discourages extending existing gaps.
    aligner.extend_gap_score = -1


    aligner.mode = "local"

    print("Alignment Algorithm Used:", aligner.algorithm, "\n")

    sequence_length = len(fireball_sequence)
    
    # Locally align fireball with de bruijn sequence
    alignments = aligner.align(DE_BRUIJN_SEQUENCE, fireball_sequence)

    print("No. Possible Alignments:", len(alignments))
    print("Top Alignment Score:", alignments[0].score)
    alignments = sorted(alignments)
    alignment = alignments[0]
    print(alignment)

    indices = alignment.indices
    print(indices)
    start_index = indices[0][0] - indices[1][0]
    # plus one for the last fireball node
    end_index = indices[0][-1] + (sequence_length - indices[1][-1])
    print("De Bruijn Start Index:", start_index)
    print("De Bruijn End   Index:", end_index)

    de_bruijn_segment = cyclic_slice(DE_BRUIJN_SEQUENCE, start_index, end_index)

    # gaps found from local alignment represented with -1.
    # add these gaps into the fireball sequence for similarity analysis
    gaps = np.where(indices[1] == -1)[0]
    for gap in gaps:
        fireball_sequence = fireball_sequence[:gap] + '-' + fireball_sequence[gap:]

    print("De Bruijn Segment:", de_bruijn_segment)
    print("Fireball Sequence:", fireball_sequence)

    # Retrieve ratio of edit distance between the de bruijn segment
    # and the fireball sequence. from 0 being completely different
    # to 1 being exactly the same
    levenshtein_ratio = Levenshtein.ratio(de_bruijn_segment, fireball_sequence)
    print(f"Levenshtein Ratio: {levenshtein_ratio:.4f}\n\n")

    fireball_alignment = FireballAlignment(
        left_to_right,
        levenshtein_ratio,
        start_index,
        de_bruijn_segment,
        fireball_sequence
    )

    return fireball_alignment


def get_best_alignment(distance_labels: list[list]) -> FireballAlignment:
    """
        Attempts to find the best alignment between the given sequence of distance
        labels and the de Bruijn sequence. It first generates left-to-right and
        right-to-left sequences from the distance labels, performs alignments for
        each sequence, and compares the alignment scores to determine the best 
        alignment. Finally, it returns the best alignment found.

        ### Parameters
        | Name             | Type       | Description                                                  |
        |------------------|------------|--------------------------------------------------------------|
        | distance_labels  | list[list] | List of lists containing cluster labels for each data point. |

        ### Returns
        | Type             | Description                                        |
        |------------------|----------------------------------------------------|
        | FireballAlignment | The best alignment found between the sequences.   |
    """
    lr_sequence = get_left_to_right_sequence_from_distance_labels(distance_labels)

    print("Left to right sequence:\n", lr_sequence)
    lr_alignment = perform_alignment(lr_sequence, True)

    rl_sequence = lr_sequence[::-1]
    print("Right to left sequence:\n", rl_sequence)
    rl_alignment = perform_alignment(rl_sequence, False)

    alignment = lr_alignment

    if rl_alignment.levenshtein_ratio > lr_alignment.levenshtein_ratio:
        alignment = rl_alignment

    print("Left to right?", alignment.left_to_right)
    print()

    print("De Bruijn Segment:", alignment.de_bruijn_segment)
    print("Fireball Sequence:", alignment.fireball_sequence)
    print("Distance Labels  :", "".join([str(i) for i in distance_labels]))
    print()

    return alignment