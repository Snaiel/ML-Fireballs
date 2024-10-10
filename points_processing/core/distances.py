import numpy as np
from core import RANDOM_STATE
from core.blobs import FireballBlobs
from sklearn.cluster import KMeans


def get_distances_between_blobs(positions: np.ndarray[float, float]) -> np.ndarray[float]:
    """
        Make sure positions is sorted by x value beforehand.

        ### Parameters
        | Name      | Type                      | Description                                     |
        |-----------|---------------------------|-------------------------------------------------|
        | positions | np.ndarray[float, float]  | The list of sorted x and y values for the blobs |

        ### Returns
        | Type                   | Description                          |
        |------------------------|--------------------------------------|
        | np.ndarray[float]      | The distances between nodes          |
    """
    
    diffs = np.diff(positions[:, :2], axis=0)
    distances = np.linalg.norm(diffs, axis=1)

    return distances


def get_indices_of_unusual_distances(distances: np.ndarray[float]) -> np.ndarray[int]:
    """
        Get the indices of unusual distances in the given array.


        ### Parameters
        | Name      | Type                   | Description                |
        |-----------|------------------------|----------------------------|
        | distances | np.ndarray[float]      | The list of distances.     |

        ### Returns
        | Type              | Description                              |
        |-------------------|------------------------------------------|
        | np.ndarray[int]   | The indices of the unusual distances.    |
    """

    q25, q75 = np.percentile(distances, [25, 75])
    iqr = q75 - q25
    lower_bound = q25 - (1.5 * iqr)
    upper_bound = q75 + (8 * iqr)
    
    unusual_indices = np.nonzero((distances < lower_bound) | (distances > upper_bound))
    
    print(
        np.column_stack(
            (
                ["q25", "q75", "iqr", "lower_bound", "upper_bound"],
                [q25, q75, iqr, lower_bound, upper_bound]
            )
        ),
        "\n"
    )

    return unusual_indices


def get_distance_groups(distances: np.ndarray[float], min_group_number: int = 20) -> list[np.ndarray[float]]:
    """
        Retrieves a list of distance groups to localise analysis.

        ### Parameters
        | Name                  | Type                    | Description                                            |
        |-----------------------|-------------------------|--------------------------------------------------------|
        | distances             | np.ndarray[float]       | The list of distances.                                 |
        | min_distances_amount  | int                     | If below this, a list of one group is returned.        |
        | min_group_number      | int                     | The minimum number of distances per group.             |

        ### Returns
        | Type                     | Description                                                             |
        |--------------------------|-------------------------------------------------------------------------|
        | list[np.ndarray[float]]  | A list of arrays containing distances. Each array is a distance group   |
    """
    
    distance_groups = []
    number_of_distances = len(distances)

    if number_of_distances < min_group_number * 2:
        # Return one group of distances.
        return [distances]

    # Only split if the splits would have above the minimum group number

    number_of_groups = number_of_distances // min_group_number
    remainder_from_20 = number_of_distances % min_group_number
    number_extra_from_remainder = remainder_from_20 // number_of_groups

    base_group_size = min_group_number + number_extra_from_remainder
    print("Base group size:", base_group_size)

    for i in range(number_of_groups):
        if i < number_of_groups - 1:
            # Every group except the last group must have a group size of base_group_size
            group = distances[i * base_group_size: (i + 1) * base_group_size]
        else:
            # Last group gets the leftovers. May or may not have a group size of base_group_size.
            group = distances[i * base_group_size: number_of_distances]
        distance_groups.append(group)

    print("Distance groups:\n", distance_groups)

    return distance_groups


def k_means_distances(distances: np.ndarray[float]) -> tuple[list[float], list[int]]:
    """
        Performs K-Means clustering on the provided distances to classify
        them as either a 1 (short) or a 0 (long).

        The provided distances should be a single distance group so that the analysis
        is done locally. Changes throughout the fireball may cause a long gap on one
        end to be classified as a short gap at the other end. So using distance groups
        helps localise clustering on a small part of the fireball.

        ### Parameters
        | Name       | Type                    | Description                    |
        |------------|-------------------------|--------------------------------|
        | distances  | np.ndarray[float]       | The list of distances.        |

        ### Returns
        | Type                            | Description                                           |
        |---------------------------------|-------------------------------------------------------|
        | tuple[list[float], list[int]]   | A tuple of a list of the two cluster centres and      |
        |                                 | a list of the corresponding labels of the distances   |
    """
    reshaped_distances = np.array(distances).reshape(-1, 1)
    kmeans = KMeans(2, random_state=RANDOM_STATE)
    kmeans.fit(reshaped_distances)

    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_[:,0]
    # Get cluster labels for each data point
    labels = kmeans.labels_

    # make it so that the 'longer' distances get labelled with a 1
    # and 'shorter' distances get labelled with a 2

    node_distances_labels = labels
    # cluster labels are either 0 or 1 depending on which label goes first.
    # so check if the shorter labels came first and to do a bitwise XOR on them (flip it).
    if cluster_centers[0] < cluster_centers[1]:
        node_distances_labels ^= 1
    else:
        cluster_centers[0], cluster_centers[1] = cluster_centers[1], cluster_centers[0]

    print("Cluster centers:\n", cluster_centers)
    # 1 = long, 0 = short
    print("Distance labels:\n", node_distances_labels, "\n")

    return (cluster_centers, node_distances_labels)


def get_distance_labels_using_k_means(distance_groups: list[np.ndarray[float]]) -> list[int]:
    """
        Given a list of distance groups, retrieve the distance labels for all distances.

        ### Parameters
        | Name            | Type                          | Description                 |
        |-----------------|-------------------------------|-----------------------------|
        | distance_groups | list[np.ndarray[float]]       | A list of distance groups.  |

        ### Returns
        | Type            | Description                                      |
        |-----------------|--------------------------------------------------|
        | list[int]       | The distance label for each distance. Flattened  |
    """

    # Retrieve the k means results for each of the distance groups
    k_means_groups = []
    for i in range(len(distance_groups)):
        print(f"Group {i+1}")
        k_means_groups.append(k_means_distances(distance_groups[i]))

    distance_labels = []
    # put all labels into one array
    for _, labels in k_means_groups:
        distance_labels.extend(labels)
    print("Fireball distance labels after initial k means cluster:\n", distance_labels,"\n")

    return distance_labels


def get_unusually_small_distances(distances: np.ndarray[float], distance_labels: list[int]) -> list[int]:
    """
        Based on the small gaps found using k means cluster,
        check every gap to see if it's unusually small.
        Assume an unusually small gap means that one of
        the nodes that created this distance is a false positive.

        NOTE: What if there are many false positives in a row?
        Currently not taken into account.

        ### Parameters
        | Name             | Type                    | Description                              |
        |------------------|-------------------------|------------------------------------------|
        | distances        | np.ndarray[float]       | The list of distances.                   |
        | distance_labels  | list[int]               | The 0 or 1 labels for the distances.     |

        ### Returns
        | Type             | Description                               |
        |------------------|-------------------------------------------|
        | list[int]        | The indices of unusually small distances. |
    """

    # threshold used to check if current distance is
    # unusually small compared to the average of its
    # neighbours
    SIZE_THRESHOLD = 0.7

    small_gaps_mask_array = np.array(distance_labels, dtype=bool)
    small_gaps = distances[small_gaps_mask_array]
    
    print("Small gaps to check:\n", small_gaps, "\n")

    small_gaps_length = len(small_gaps)

    original_indices = [i for i, m in enumerate(distance_labels) if m]

    # consider 6 neighbouring distances to check if the current
    # distance is unusually small
    window_size = 7

    unusually_small_distance_indices = []

    for i, gap in enumerate(small_gaps):
        left_idx = i - (window_size // 2)
        right_idx = i + (window_size // 2) + 1

        # ensure a window of 7 is retrieved even if on the ends
        if left_idx < 0:
            right_idx += abs(left_idx)
            left_idx = 0
        
        if right_idx > small_gaps_length + 1:
            left_idx -= (right_idx - small_gaps_length)
            right_idx = small_gaps_length
        
        # retrieve window of gaps, get mean
        window = small_gaps[left_idx:right_idx]
        window_mean = np.mean(window)

        if gap < window_mean * SIZE_THRESHOLD:
            unusually_small_distance_indices.append(original_indices[i])

    return unusually_small_distance_indices


def get_false_positives_based_on_distance(fireball_blobs: FireballBlobs, small_distance_indices: list) -> tuple[list[int], list[int]]:
    """
        Retrieves the indices of the false positive blobs
        and their corresponding distances.

        ### Parameters
        | Name                   | Type                      | Description                                  |
        |------------------------|---------------------------|----------------------------------------------|
        | fireball_blobs         | FireballBlobs             | The list of fireball blobs (x, y, r).        |
        | small_distance_indices | list                      | The indices of unusually small distances.    |

        ### Returns
        | Type                          | Description                                     |
        |-------------------------------|-------------------------------------------------|
        | tuple[list[int], list[int]]   | A tuple containing two lists. The first being   |
        |                               | the indices of fireballs to remove, the second  |
        |                               | being the list of distances to remove.          |
    """

    fireball_indices_to_remove = []
    new_distance_labels_indices_to_remove = []

    for ii in small_distance_indices:

        if ii not in new_distance_labels_indices_to_remove:
            new_distance_labels_indices_to_remove.append(ii)
        
        fireball_index_to_remove = -1
        
        # a distance value has a left node and right node.
        if fireball_blobs[ii][2] < fireball_blobs[ii + 1][2]:
            # left node smaller than right node.
            # treat left node as false positive
            fireball_index_to_remove = ii
            if (ii - 1) not in new_distance_labels_indices_to_remove:
                new_distance_labels_indices_to_remove.append(ii - 1)
        else:
            # right node smaller than left node
            # treat right node as false positive
            fireball_index_to_remove = ii + 1
            if (ii + 1) not in new_distance_labels_indices_to_remove:
                new_distance_labels_indices_to_remove.append(ii + 1)

        if fireball_index_to_remove not in fireball_indices_to_remove:
            fireball_indices_to_remove.append(fireball_index_to_remove)
        
    return fireball_indices_to_remove, new_distance_labels_indices_to_remove


def recalculate_distances(fireball_blobs: FireballBlobs, distances: np.ndarray[float], removed_blobs: list[int]) -> np.ndarray[float]:
    """
        Inserts newly calculated distances between fireball blobs since the blobs
        specified by removed_blobs left a gap in the distances array.

        ### Parameters
        | Name               | Type                 | Description                                              |
        |--------------------|----------------------|----------------------------------------------------------|
        | fireball_blobs     | FireballBlobs        | The list of fireball blobs (x, y, r).                    |
        | distances          | np.ndarray[float]    | The list of distances (Current has missing distances).   |
        | removed_blobs      | list[int]            | Indices of fireball blobs that were removed              |

        ### Returns
        | Type                   | Description                                         |
        |------------------------|-----------------------------------------------------|
        | np.ndarray[float]      | Distances with the newly calculated insertions.     |
    """
    
    for blob_index in removed_blobs:
        distances = np.insert(
            distances,
            blob_index,
            np.sqrt(
                np.sum(
                    (
                        fireball_blobs[blob_index + 1][:2] - fireball_blobs[blob_index - 1][:2]
                    )**2
                )
            )
        )
    
    return distances