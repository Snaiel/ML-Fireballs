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
from sklearn.pipeline import make_pipeline


def main():

    image = ski.io.imread(
        "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL48/48_2015-11-01_110359_DSC_9000/48_2015-11-01_110359_DSC_9000_57_3319-91-3607-165.differenced.jpg",
        as_gray=True
    )

    image_gray = image

    # Detect blobs using Difference of Gaussian (DoG)
    blobs_dog = blob_dog(image_gray, min_sigma=2, max_sigma=15, threshold=0.01)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)  # Compute radii in the 3rd column

    print(len(blobs_dog))
    for b in blobs_dog:
        print(b)

    # Extract x and y coordinates from blobs
    y_coords = blobs_dog[:, 0]
    x_coords = blobs_dog[:, 1]

    # Fit a linear regression model using RANSAC, fallback to simple Linear Regression if samples are less than 3
    if len(x_coords) >= 3:
        model_ransac = make_pipeline(RANSACRegressor(estimator=LinearRegression(), residual_threshold=10, max_trials=100))
        model_ransac.fit(x_coords.reshape(-1, 1), y_coords)

        # Predict y values using the fitted model
        domain = np.linspace(0, image_gray.shape[1], 1000)
        y_values_ransac = model_ransac.predict(domain.reshape(-1, 1))

        # Check if the predicted y-values fall within the image dimensions
        valid_indices_ransac = np.where((y_values_ransac >= 0) & (y_values_ransac < image_gray.shape[0]))

        # Plot the original image with the fitted line
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Linear Regression with RANSAC")
        ax.imshow(image_gray, cmap='gray', aspect='equal')

        # Plot the RANSAC regression line
        if len(valid_indices_ransac[0]) > 0:
            ax.plot(domain[valid_indices_ransac], y_values_ransac[valid_indices_ransac], color='orange', linestyle='-', linewidth=2)

        # Get the inlier indices
        inlier_indices = model_ransac.named_steps['ransacregressor'].inlier_mask_

        # Plot inlier points
        for idx in np.where(inlier_indices)[0]:
            blob = blobs_dog[idx]
            y, x, r = blob
            c = plt.Circle((x, y), r, color='lime', linewidth=2, fill=False)
            ax.add_patch(c)

        # Plot outlier points
        for idx in np.where(inlier_indices == False)[0]:
            blob = blobs_dog[idx]
            y, x, r = blob
            c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
            ax.add_patch(c)

        # Output line parameters
        ransac_model = model_ransac.named_steps['ransacregressor']
        print(f"Gradient: {ransac_model.estimator_.coef_[0]}")
        print(f"Intercept: {ransac_model.estimator_.intercept_}")

    else:
        # Fallback to simple Linear Regression
        model_linear = LinearRegression()
        model_linear.fit(x_coords, y_coords)

        # Predict y values using the fitted model
        domain = np.linspace(0, image_gray.shape[1], 1000)
        y_values_linear = model_linear.predict(domain.reshape(-1, 1))

        # Check if the predicted y-values fall within the image dimensions
        valid_indices_linear = np.where((y_values_linear >= 0) & (y_values_linear < image_gray.shape[0]))

        # Plot the original image with the fitted line
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Linear Regression without RANSAC")
        ax.imshow(image_gray, cmap='gray', aspect='equal')

        # Plot the Linear Regression line
        if len(valid_indices_linear[0]) > 0:
            ax.plot(domain[valid_indices_linear], y_values_linear[valid_indices_linear], color='blue', linestyle='-', linewidth=2)
        
        # Plot the blobs
        for blob in blobs_dog:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='lime', linewidth=2, fill=False)
            ax.add_patch(c)

        # Output line parameters
        linear_model = model_linear.named_steps['linearregression']
        print(f"Gradient: {linear_model.coef_}")
        print(f"Intercept: {linear_model.intercept_}")

    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()