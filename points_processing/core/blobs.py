from math import sqrt

import numpy as np
import pandas as pd
from core import RANDOM_STATE
from skimage.feature import blob_dog
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


FireballBlobs = np.ndarray[tuple[float, float, float]]


def get_fireball_blobs(image: np.ndarray, min_radius: float = 3, max_radius: float = 30, threshold: float = 0.025, **kwargs) -> FireballBlobs:
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
    for key, value in kwargs.items():
        if key == 'min_radius':
            min_radius = value
        elif key == 'max_radius':
            max_radius = value
        elif key == 'threshold':
            threshold = value
    
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


def get_circle_brightness(image: np.ndarray, x: int, y: int, r: float) -> float:
    """
        Calculates the average brightness of a circle in a grayscale image.
        
        ### Parameters
        | Name  | Type         | Description                                |
        |-------|--------------|--------------------------------------------|
        | image | np.ndarray   | 2D array representing the grayscale image. |
        | x     | int          | x-coordinate of the circle's center.       |
        | y     | int          | y-coordinate of the circle's center.       |
        | r     | float        | radius of the circle.                      |

        ### Returns
        | Type   | Description                           |
        |--------|---------------------------------------|
        | float  | average brightness within the circle. |
    """

    rows, cols = image.shape
    Y, X = np.ogrid[:rows, :cols]
    
    # Create a boolean mask for the circle
    dist_from_center = np.sqrt((X - x)**2 + (Y - y)**2)
    
    # Extract pixel values within the circle
    circle_pixels = image[dist_from_center <= r]
    
    # Calculate and return the average brightness
    return circle_pixels.mean()


def get_blob_brightnesses(image: np.ndarray, blobs: FireballBlobs) -> list[float]:
    """
        Calculates the average brightness within each blob in a grayscale image.

        ### Parameters
        | Name   | Type            | Description                                          |
        |--------|-----------------|------------------------------------------------------|
        | image  | np.ndarray      | 2D array representing the grayscale image.           |
        | blobs  | FireballBlobs   | list of blobs (x, y, r).                             |

        ### Returns
        | Type        | Description                                  |
        |-------------|----------------------------------------------|
        | list[float] | List of average brightness of each blob.     |
    """

    brightnesses = []
    for x, y, r in blobs:
        brightnesses.append(get_circle_brightness(image, x, y, r))
    return brightnesses


def get_false_positives_based_on_blobs(image: np.ndarray, blobs: FireballBlobs, threshold: float = -20) -> tuple[pd.Series, list[int]]:
    """
        Identifies false positives based on the characteristics of detected blobs in a grayscale image.

        It computes the size and brightness of each blob. It also computes the moving average of the
        size and brightness. It then takes the percentage difference of each value with its corresponding
        moving average. Being a certain percent difference below the threshold constitutes being a false positive.

        ### Parameters
        | Name       | Type                 | Description                                                |
        |------------|----------------------|------------------------------------------------------------|
        | image      | np.ndarray           | 2D array representing the grayscale image.                 |
        | blobs      | FireballBlobs        | Object containing information about detected blobs.        |
        | threshold  | float (default: -20) | Threshold value for determining false positives.           |

        ### Returns
        | Type            | Description                                                       |
        |-----------------|-------------------------------------------------------------------|
        | pd.Series       | Series containing the mean percent difference for each blob.      |
        | list[int]       | List of indices corresponding to blobs identified as false positives. |
    """
    brightnesses = get_blob_brightnesses(image, blobs)

    brightness_series = pd.Series(brightnesses)
    brightness_moving_avg = brightness_series.rolling(window=5, center=True).mean()
    brightness_percent_difference = ((brightness_series - brightness_moving_avg) / brightness_moving_avg) * 100

    size_series = pd.Series(blobs[:, 2])
    size_moving_avg = size_series.rolling(window=5, center=True).mean()
    size_percent_difference = ((size_series - size_moving_avg) / size_moving_avg) * 100

    mean_percent_difference = (brightness_percent_difference + size_percent_difference) / 2
    indices = mean_percent_difference.index[mean_percent_difference < threshold].to_list()

    return mean_percent_difference, indices