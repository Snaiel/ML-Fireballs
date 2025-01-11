from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from scipy.interpolate import BSpline
from skimage.feature import blob_dog
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer


def main():

    image_path = Path(
        "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL51/51_2015-11-01_110428_DSC_0144/51_2015-11-01_110428_DSC_0144_56_4032-1-4266-51.differenced.jpg"
    )

    image = ski.io.imread(
        image_path,
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

    if len(x_coords) < 3:
        print("Less than 3 blobs, not performing line calculation.")
        return

    ax.set_title("Fitting a Degree 2 Spline to Streak Blobs Using RANSAC and Linear Regression")

    spline_transformer = SplineTransformer(n_knots=2, degree=2, extrapolation="continue")
    ransac = RANSACRegressor(residual_threshold=5, max_trials=100)

    model = make_pipeline(
        spline_transformer,
        ransac
    )

    model.fit(x_coords.reshape(-1, 1), y_coords)

    # Get the inlier indices
    inlier_indices = ransac.inlier_mask_

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

    # Predict y values using the fitted model
    y_values = model.predict(domain.reshape(-1, 1))

    # Check if the predicted y-values fall within the image dimensions
    valid_indices = np.where((y_values >= 0) & (y_values < image_gray.shape[0]))

    # Plot the Linear Regression line
    if len(valid_indices[0]) > 0:
        ax.plot(
            domain[valid_indices],
            y_values[valid_indices],
            color='orange',
            linestyle='-',
            linewidth=2
        )

    # Access the B-spline representation
    bspline: BSpline = spline_transformer.bsplines_[0]  # Get the first B-spline (one feature case)

    # Midpoint of the data
    x_mid = (x_coords.min() + x_coords.max()) / 2

    # Calculate the derivatives of the basis functions at the point
    basis_derivatives = bspline.derivative()(x_mid)

    # Retrieve coefficients
    estimator: LinearRegression = ransac.estimator_
    coefficients = estimator.coef_

    # Compute the actual gradient (weighted sum of basis derivatives)
    gradient = np.dot(basis_derivatives, coefficients)

    print(f"Gradient (slope) of the spline at x={x_mid} is {gradient}")

    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()