from dataclasses import dataclass
from pathlib import Path, PosixPath

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.axes import Axes

from point_pickings.core.assign import FireballPoint, assign_labels_to_blobs
from point_pickings.core.blobs import (FireballBlobs, get_blob_brightnesses,
                                       get_false_positives_based_on_blobs,
                                       get_fireball_blobs, sort_fireball_blobs)
from point_pickings.core.distances import (
    get_distance_groups, get_distance_labels_using_k_means,
    get_distances_between_blobs, get_false_positives_based_on_distance,
    get_indices_of_unusual_distances, get_unusually_small_distances,
    recalculate_distances)
from point_pickings.core.sequence import FireballAlignment, get_best_alignment
from point_pickings.core.utils import make_image_landscape


@dataclass
class FullAutoFireball:
    image: np.ndarray
    rotated: bool
    fireball_blobs: FireballBlobs
    brightnesses: list[float]
    removed_blobs_size_brightness: FireballBlobs
    distances: np.ndarray[float]
    removed_blobs_unusually_small_distances: FireballBlobs
    distance_groups: list[np.ndarray[float]]
    distance_labels: list[int]
    alignment: FireballAlignment
    fireball_points: list[FireballPoint]


def retrieve_fireball(image: np.ndarray | PosixPath | str, **kwargs) -> FullAutoFireball:
    """
        Retrieves fireball information from the given image. Processes the image to make
        it landscape orientation, detects fireball blobs, calculates distances between nodes,
        removes unusually small blobs and distances, performs K-Means clustering, removes
        false positives based on distance, realigns the sequence, assigns labels to blobs,
        and returns a FullAutoFireball object containing all relevant information.

        ### Parameters
        | Name          | Type                           | Description                          |
        |---------------|--------------------------------|--------------------------------------|
        | image         | np.ndarray or PosixPath or str | The image file or ndarray image to   |
        |               |                                | retrieve fireball information from.  |

        ### Returns
        | Type               | Description                                                    |
        |--------------------|----------------------------------------------------------------|
        | FullAutoFireball   | An object containing information about the retrieved fireball. |
    """

    if type(image) in [str, PosixPath]:
        image = ski.io.imread(image)
    elif type(image) is np.ndarray:
        image = image
    else:
        raise Exception("image must be file path or ndarray image")
    
    print("Making image landscape...")
    rotated_image = make_image_landscape(image)
    rotated = rotated_image.shape != image.shape
    image = rotated_image

    ## Blob Detection
    print("Retrieving nodes...")
    fireball_blobs: FireballBlobs
    fireball_blobs = get_fireball_blobs(image, **kwargs)
    fireball_blobs = sort_fireball_blobs(fireball_blobs)

    ## Brightnesses
    brightnesses = get_blob_brightnesses(image, fireball_blobs)
    print("Brightnesses:")
    for brightness in brightnesses:
        print(brightness)
    print()

    ## Remove false positives based on blob size and brightness
    mean_percent_difference, false_positive_indices_size_brightess = get_false_positives_based_on_blobs(image, fireball_blobs)
    removed_blobs_size_brightness = fireball_blobs[false_positive_indices_size_brightess]
    fireball_blobs = np.delete(
        fireball_blobs,
        false_positive_indices_size_brightess,
        axis=0
    )
    print("Size and brightness mean percent difference from respective moving averages:")
    print(mean_percent_difference)
    print("Fireball blob indices to remove based on size and brightness:")
    print(false_positive_indices_size_brightess)
    print()

    ## Distances Between Nodes                       
    distances = get_distances_between_blobs(fireball_blobs[:, :2])
    print("Distances:\n", distances, "\n")

    ## Remove unusual distances
    ## NOTE: THIS DOESN"T ACTUALLY WORK PROPERLY
    ## THE DISTANCES ARE REMOVED BUT THE FIREBALL NODES STILL EXIST.
    unusual_distances = get_indices_of_unusual_distances(distances)
    print("Unusual distances:", unusual_distances)
    distances = np.delete(
        distances,
        unusual_distances,
        axis=0
    )
    print("Distances with outliers based on distances removed:\n", distances)
    print("Number of distances:", len(distances), "\n")

    ## Distance Groups
    distance_groups = get_distance_groups(distances)

    ## Initial K-Means Cluster
    distance_labels = get_distance_labels_using_k_means(distance_groups)
    print("Fireball distance labels after initial k means cluster:\n", distance_labels,"\n")

    ## Remove Small Distances
    unusually_small_distance = get_unusually_small_distances(
        distances,
        distance_labels
    )
    fireball_indices_to_remove, distance_labels_indices_to_remove = get_false_positives_based_on_distance(
        fireball_blobs,
        unusually_small_distance
    )

    print("Indicies to remove based on unusually small distance:")
    print("Fireball indices:", fireball_indices_to_remove)
    print("Distance indices:", distance_labels_indices_to_remove, "\n")

    print("No. of Distances, No. of Distance Labels, No. of Fireballs")
    print("Before deleting unusually small distances:\n", len(distances), len(distance_labels), len(fireball_blobs))

    distances = np.delete(distances, distance_labels_indices_to_remove)

    print("After deleting unusually small distances:\n", len(distances), len(distance_labels), len(fireball_blobs))

    distances = recalculate_distances(fireball_blobs, distances, fireball_indices_to_remove)

    print("After inserting distances skipping false fireball nodes:\n", len(distances), len(distance_labels), len(fireball_blobs))

    removed_blobs_unusually_small_distances = fireball_blobs[fireball_indices_to_remove]
    fireball_blobs = np.delete(fireball_blobs, fireball_indices_to_remove, axis=0)
    print("After removing false fireball node:\n", len(distances), len(distance_labels), len(fireball_blobs))
    print()

    print("Final Fireballs Nodes:\n", fireball_blobs, "\n")


    ## K-Means Again                  
    # Retrieve distance groups and distance labels again
    # now that we removed the assumed false positives
    # based on unusually small distances
    distance_groups = get_distance_groups(distances)
    distance_labels = get_distance_labels_using_k_means(distance_groups)

    print("Final Distance Labels:\n", distance_labels, "\n")
    print("Final No.\n", len(distances), len(distance_labels), len(fireball_blobs), "\n\n\n")

    ## Sequence Alignment
    alignment = get_best_alignment(distance_labels)
    if not alignment.left_to_right:
        fireball_blobs = np.flip(fireball_blobs, 0)
        print(fireball_blobs)
        distance_labels = np.flip(distance_labels, 0)

    print()

    ## Assign labels to blobs
    fireball_points = assign_labels_to_blobs(
        fireball_blobs,
        distance_labels,
        alignment
    )

    original_blobs = sort_fireball_blobs(np.concatenate((fireball_blobs, removed_blobs_size_brightness, removed_blobs_unusually_small_distances)))

    fireball = FullAutoFireball(
        image,
        rotated,
        original_blobs,
        brightnesses,
        removed_blobs_size_brightness,
        distances,
        removed_blobs_unusually_small_distances,
        distance_groups,
        distance_labels,
        alignment,
        fireball_points
    )

    return fireball


def visualise_fireball(fireball: FullAutoFireball):
    """
        Visualizes the fireball using matplotlib. It plots the fireball image with
        fireball nodes in lime color, distance groups in pink color, and distance
        labels between fireball nodes.
        
        ### Parameters
        | Name          | Type               | Description                                                            |
        |---------------|--------------------|------------------------------------------------------------------------|
        | fireball      | FullAutoFireball   | An object containing information about the fireball to be visualized.  |
    """
    
    ax: Axes
    _, ax = plt.subplots()
    ax.imshow(fireball.image, cmap='gray', aspect='equal')

    # Plot fireball nodes in lime
    for node in fireball.fireball_blobs:
        x, y, r = node
        c = plt.Circle((x, y), r, color='lime', linewidth=2, fill=False)
        d = plt.Circle((x, y), 0.5, color='lime', fill=True)
        ax.add_patch(c)
        ax.add_patch(d)

    # Plot removed blobs based on size and brightness
    for node in fireball.removed_blobs_size_brightness:
        x, y, r = node
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        ax.add_patch(c)

    # Distance Groups
    cumulative_node_count = 0
    for i in range(len(fireball.distance_groups)):
        node = fireball.fireball_blobs[cumulative_node_count]
        x, y, r = node
        c = plt.Circle((x, y), r, color='pink', linewidth=2, fill=False)
        ax.add_patch(c)

        cumulative_node_count += len(fireball.distance_groups[i])

    # Plot blobs removed from unusually small distances
    for node in fireball.removed_blobs_unusually_small_distances:
        x, y, r = node
        c = plt.Circle((x, y), r, color='orange', linewidth=2, fill=False)
        ax.add_patch(c)

    # Plot Distance Labels
    for i in range(len(fireball.distance_labels)):
        if i + 1 == len(fireball.fireball_blobs):
            break
        x1, y1, _ = fireball.fireball_blobs[i]
        x2, y2, _ = fireball.fireball_blobs[i + 1]
        ax.text(((x1 + x2) / 2) - 7, ((y1 + y2) / 2), fireball.distance_labels[i], color="white")

    plt.show()



def main():
    image_path = Path(Path(__file__).parents[1], "data", "fireball_highlights", "cropped", "025_2021-09-09_053629_E_DSC_0379-G_cropped.jpeg")
    fireball = retrieve_fireball(image_path)
    visualise_fireball(fireball)


if __name__ == "__main__":
    main()