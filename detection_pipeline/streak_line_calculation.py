"""
    Blob Detection and Polynomial Curve Fitting Script

    This script performs blob detection on a set of grayscale images using the Difference of Gaussian (DoG) method from the scikit-image library.
    It fits polynomial curves to the detected blobs using both standard linear regression and RANSAC (RANdom SAmple Consensus).
    The results, including inliers and outliers detected by RANSAC, are visualized on the original images using matplotlib.

    The script includes the following functionalities:
    - Loading grayscale images from a specified directory.
    - Rotating images to landscape orientation if necessary.
    - Detecting blobs in the images using the DoG method.
    - Fitting polynomial curves to the detected blobs using both standard linear regression and RANSAC.
    - Visualizing the detected blobs and fitted curves on the original images.
    - Using check buttons to toggle the visibility of various elements in the plot.

    Dependencies:
    - scikit-image (image processing)
    - scikit-learn (regression and RANSAC)

    Usage:
        Run the script to perform blob detection and polynomial curve fitting on a predefined set of sample images and display the results.

        Whilst in `point_pickings/`, run:

        python3 blob_detection/view_blobs.py
"""


import os
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from skimage.feature import blob_dog
from sklearn.linear_model import LinearRegression, RANSACRegressor


def main():

    image = ski.io.imread(
        "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL51/51_2015-11-01_101728_DSC_0050/51_2015-11-01_101728_DSC_0050_62_4945-3624-5023-3807.differenced.jpg",
        as_gray=True
    )

    image_gray = image

    # Detect blobs using Difference of Gaussian (DoG)
    blobs_dog = blob_dog(image_gray, min_sigma=2, max_sigma=10, threshold=0.01)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)  # Compute radii in the 3rd column

    print("Blobs:", len(blobs_dog))
    for b in blobs_dog:
        print(b)

    # Extract x and y coordinates from blobs
    y_coords = blobs_dog[:, 0]
    x_coords = blobs_dog[:, 1]

    domain: np.ndarray = np.linspace(0, image_gray.shape[1], 1000)

    _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_gray, cmap='gray', aspect='equal')

    gradient: float
    intercept: float

    # Fit a linear regression model using RANSAC, fallback to simple Linear Regression if samples are less than 3
    if len(x_coords) >= 3:
        ax.set_title("Linear Regression with RANSAC")

        model = RANSACRegressor(estimator=LinearRegression(), residual_threshold=10, max_trials=100)
        model.fit(x_coords.reshape(-1, 1), y_coords)

        # Predict y values using the fitted model
        y_values = model.predict(domain.reshape(-1, 1))

        # Line parameters
        estimator: LinearRegression = model.estimator_
        gradient = estimator.coef_[0]
        intercept = estimator.intercept_

        # Get the inlier indices
        inlier_indices = model.inlier_mask_

        # Plot inlier blobs
        for idx in np.where(inlier_indices)[0]:
            blob = blobs_dog[idx]
            y, x, r = blob
            c = plt.Circle((x, y), r, color='lime', linewidth=2, fill=False)
            ax.add_patch(c)

        # Plot outlier blobs
        for idx in np.where(inlier_indices == False)[0]:
            blob = blobs_dog[idx]
            y, x, r = blob
            c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
            ax.add_patch(c)

    else:
        ax.set_title("Linear Regression without RANSAC")

        # Fallback to simple Linear Regression
        model = LinearRegression()
        model.fit(x_coords, y_coords)

        # Predict y values using the fitted model
        y_values = model.predict(domain.reshape(-1, 1))
        
        # Line parameters
        gradient = model.coef_[0]
        intercept = model.intercept_

        # Plot the blobs
        for blob in blobs_dog:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='lime', linewidth=2, fill=False)
            ax.add_patch(c)

    # Check if the predicted y-values fall within the image dimensions
    valid_indices = np.where((y_values >= 0) & (y_values < image_gray.shape[0]))

    # Plot the Linear Regression line
    if len(valid_indices[0]) > 0:
        ax.plot(
            domain[valid_indices],
            y_values[valid_indices],
            color='orange' if len(x_coords) >= 3 else 'blue',
            linestyle='-',
            linewidth=2
        )

    print(f"Gradient: {gradient}")
    print(f"Intercept: {intercept}")

    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()