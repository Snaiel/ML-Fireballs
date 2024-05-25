import numpy as np
from core import RANDOM_STATE
from skimage.feature import blob_dog
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt


FireballBlobs = np.ndarray[tuple[float, float, float]]


def get_fireball_blobs(image: np.ndarray, min_radius: float = 3, max_radius: float = 30, threshold: float = 0.025) -> FireballBlobs:
    """
        Retrieves the fireball blobs in an image in the following steps:

        1. Blob detection using Difference of Gaussian
        2. Fitting a quadratic curve to the points
        3. Using RANSAC to only select the fireball blobs
        
        ### Parameters
        | Parameter  | Type       | Description                                    |
        |------------|------------|------------------------------------------------|
        | image      | np.ndarray |                                                |
        | min_radius | int        | minimum pixel radius of blob                   |
        | max_radius | int        | maximum pixel radius of blob                   |
        | threshold  | float      | blob intensity threshold (brightness I think?) |

        ### Returns
        | Type           | Description                          |
        |----------------|--------------------------------------|
        | FireballBlobs  | List of fireball blobs (x, y, r)     |
    """
    blobs = blob_dog(
        image,
        min_sigma=min_radius // sqrt(2),
        max_sigma=max_radius // sqrt(2),
        threshold=threshold
    )
    blobs[:, 2] = blobs[:, 2] * sqrt(2)
    print(len(blobs))

    # Extract x and y coordinates from blobs_dog
    y_coords = blobs[:, 0]
    x_coords = blobs[:, 1]

    # Fit a polynomial curve using RANSAC
    model = make_pipeline(PolynomialFeatures(2), RANSACRegressor(residual_threshold=10, max_trials=100, random_state=RANDOM_STATE))
    model.fit(x_coords.reshape(-1, 1), y_coords)

    inlier_mask = model.named_steps['ransacregressor'].inlier_mask_

    # apply inlier mask to blobs to retrieve fireball nodes
    fireball_blobs = np.compress(inlier_mask, blobs, axis=0)

    # Swpan columns so that x comes before y
    fireball_blobs[:, [0, 1]] = fireball_blobs[:, [1, 0]]

    return fireball_blobs


def sort_fireball_blobs(fireball_blobs: FireballBlobs) -> FireballBlobs:
    """
        ### Parameters
        | Name           | Type                                    | Description                               |
        |----------------|-----------------------------------------|-------------------------------------------|
        | fireball_blobs | np.ndarray[tuple[float, float, float]]  | The list of fireball blobs (x, y, r)      |


        ### Returns
        | Type                                    | Description                                              |
        |-----------------------------------------|----------------------------------------------------------|
        | np.ndarray[tuple[float, float, float]]  | The fireball blobs sorted in ascending order based on x value |
    """
    sorted_indices = np.argsort(fireball_blobs[:, 0])
    fireball_blobs = fireball_blobs[sorted_indices]

    print("Fireball nodes (x, y, r):\n", fireball_blobs, "\n")
    
    return fireball_blobs


def get_indices_of_unusually_small_blobs(blobs_sizes: np.ndarray[float]) -> list[int]:
    """
        Going from the second blob to the second last blob, check average
        radius of direct neighbours. Will be considered small if current
        blob is smaller than 40% of the neighbour average.

        ### Parameters
        | Name  | Type         | Description                   |
        |-------|--------------|-------------------------------|
        | blobs | list[float]  | List of blob radii in order.  |

        ### Returns
        | Type       | Description                               |
        |------------|-------------------------------------------|
        | list[int]  | List of indices that are unusually small. |
    """

    indices_small_blobs = []
    for i in range(1, len(blobs_sizes) - 1):
        previous = blobs_sizes[i-1]
        current = blobs_sizes[i]
        next = blobs_sizes[i+1]

        average_radius_neighbours = (previous + next) / 2

        if current < 0.40 * average_radius_neighbours:
            indices_small_blobs.append(i)
    
    return indices_small_blobs