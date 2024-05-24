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
"""


import os
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.widgets import CheckButtons
from skimage.feature import blob_dog
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def main():
    cropped_images_folder = Path(Path(__file__).parents[2], 'data', 'fireball_images', 'cropped')

    for image_file in os.listdir(cropped_images_folder):

        image = ski.io.imread(Path(cropped_images_folder, image_file))
        image_gray = image

        height, width = image_gray.shape[:2]
        if width < height:
            # Rotate the image to landscape orientation
            image_gray = ski.transform.rotate(image_gray, angle=90, resize=True)

        # Return (y-coord, x-coord, radius)
        blobs_dog = blob_dog(image_gray, max_sigma=20, threshold=.025)
        # Compute radii in the 3rd column.
        blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

        # Extract x and y coordinates from blobs_dog
        y_coords = blobs_dog[:, 0]
        x_coords = blobs_dog[:, 1]


        x_values = np.linspace(0, image_gray.shape[1], 1000)


        # Fit a polynomial curve without RANSAC
        model_poly = make_pipeline(PolynomialFeatures(2), LinearRegression())
        model_poly.fit(x_coords.reshape(-1, 1), y_coords)

        # Predict y values using the fitted model
        y_values_poly = model_poly.predict(x_values.reshape(-1, 1))

        # Plot the original image with both fitted curves
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f'{image_file}\nDifference of Gaussian, Polynomial 2 Curve')
        ax.imshow(image_gray, cmap='gray', aspect='equal')

        plt.tight_layout()
        plt.axis('off')

        # Check if the predicted y-values fall within the image dimensions
        valid_indices_poly = np.where((y_values_poly >= 0) & (y_values_poly < image_gray.shape[0]))

        # Plot the polynomial regression curve without RANSAC
        if len(valid_indices_poly[0]) > 0:
            line_poly, = ax.plot(x_values[valid_indices_poly], y_values_poly[valid_indices_poly], color='orange', linestyle='-', linewidth=2)

        # Fit a polynomial curve using RANSAC
        model_ransac = make_pipeline(PolynomialFeatures(2), RANSACRegressor(residual_threshold=10, max_trials=100))
        model_ransac.fit(x_coords.reshape(-1, 1), y_coords)

        # Predict y values using the fitted model
        y_values_ransac = model_ransac.predict(x_values.reshape(-1, 1))

        # Check if the predicted y-values fall within the image dimensions
        valid_indices_ransac = np.where((y_values_ransac >= 0) & (y_values_ransac < image_gray.shape[0]))

        # Plot the curve using RANSAC only if it stays within the image bounds
        if len(valid_indices_ransac[0]) > 0:
            line_ransac, = ax.plot(x_values[valid_indices_ransac], y_values_ransac[valid_indices_ransac], color='orange', linestyle='-', linewidth=2)

        # Get the inlier indices
        inlier_indices = model_ransac.named_steps['ransacregressor'].inlier_mask_

        # Plot inlier points
        inlier_circles = []
        for idx in np.where(inlier_indices)[0]:
            blob = blobs_dog[idx]
            y, x, r = blob
            c = plt.Circle((x, y), r, color='lime', linewidth=2, fill=False)
            inlier_circles.append(c)
            ax.add_patch(c)
        
        # Plot outlier points
        outlier_circles = []
        for idx in np.where(inlier_indices == False)[0]:
            blob = blobs_dog[idx]
            y, x, r = blob
            c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
            outlier_circles.append(c)
            ax.add_patch(c)

        # Plot blobs_dog points
        all_blobs = []
        for blob in blobs_dog:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='lime', linewidth=2, fill=False)
            all_blobs.append(c)
            ax.add_patch(c)

        # Define radio button click actions
        def toggle_visibility(label):

            if label == 'None':
                line_poly.set_visible(False)
                line_ransac.set_visible(False)
                for circle in inlier_circles:
                    circle.set_visible(False)
                for circle in outlier_circles:
                    circle.set_visible(False)
                for circle in all_blobs:
                    circle.set_visible(False)

            if label == 'Blobs':
                for circle in all_blobs:
                    circle.set_visible(check.get_status()[1])

            if label == 'Line Poly':
                line_poly.set_visible(check.get_status()[2])

            if label == 'Line RANSAC':
                line_ransac.set_visible(check.get_status()[3])

            if label == 'Inliers':
                for circle in inlier_circles:
                    circle.set_visible(check.get_status()[4])

            if label == 'Outliers':
                for circle in outlier_circles:
                    circle.set_visible(check.get_status()[5])
            
            plt.draw()


        toggle_visibility('None')

        # Add radio buttons
        ax_check = plt.axes([0.01, 0.5, 0.1, 0.15])
        check = CheckButtons(ax_check, ('None', 'Blobs', 'Line Poly', 'Line RANSAC', 'Inliers', 'Outliers'))
        check.on_clicked(toggle_visibility)

        # plt.title(event_folder)
        plt.show()


if __name__ == "__main__":
    main()