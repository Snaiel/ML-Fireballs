from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from scipy.interpolate import BSpline
from skimage.feature import blob_dog
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import SplineTransformer


class StreakSpline:

    _blobs: np.ndarray
    _model: Pipeline
    _spline_transformer: SplineTransformer
    _ransac: RANSACRegressor

    def __init__(self, image: str | Path | np.ndarray):
        if not type(image) is np.ndarray:
            image = ski.io.imread(image, as_gray=True)

        blobs_dog = blob_dog(image, min_sigma=2, max_sigma=10, threshold=0.02)
        blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)  # Compute radii in the 3rd column
        self._blobs = blobs_dog

        if not self.is_valid:
            return

        self._y_coords = blobs_dog[:, 0]
        self._x_coords = blobs_dog[:, 1]

        self._spline_transformer = SplineTransformer(n_knots=2, degree=2, extrapolation="continue")
        self._ransac = RANSACRegressor(residual_threshold=5, max_trials=100)

        self._model = make_pipeline(self._spline_transformer, self._ransac)
        self._model.fit(self._x_coords.reshape(-1, 1), self._y_coords)


    @property
    def blobs(self) -> np.ndarray:
        return self._blobs


    @property
    def is_valid(self) -> bool:
        return len(self._blobs) >= 3


    @property
    def inlier_indices(self) -> np.ndarray:
        return self._ransac.inlier_mask_


    @property
    def midpoint(self) -> tuple:
        x_mid = (self._x_coords.min() + self._x_coords.max()) / 2
        return (
            float(x_mid),
            float(self.predict(np.array([[x_mid]]))[0])
        )


    def predict(self, *args, **kwargs):
        return self._model.predict(*args, **kwargs)


    def gradient(self, x: float) -> float:
        bspline: BSpline = self._spline_transformer.bsplines_[0]
        basis_derivatives = bspline.derivative()(x)
        estimator: LinearRegression = self._ransac.estimator_
        gradient = np.dot(basis_derivatives, estimator.coef_)
        return gradient


def main():
    image_path = Path(
        "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL48/48_2015-11-01_110159_DSC_8996/48_2015-11-01_110159_DSC_8996_67_1764-797-2071-1054.differenced.jpg"
    )

    streak_spline = StreakSpline(image_path)

    image = ski.io.imread(
        image_path,
        as_gray=True
    )

    _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray', aspect='equal')

    if not streak_spline.is_valid:
        print("Less than 3 blobs, not performing line calculation.")
        return

    ax.set_title("Fitting a Degree 2 Spline to Streak Blobs Using RANSAC and Linear Regression")

    # Plot inlier blobs
    for idx in np.where(streak_spline.inlier_indices)[0]:
        blob = streak_spline.blobs[idx]
        y, x, r = blob
        c = plt.Circle((x, y), r, color='lime', linewidth=2, fill=False)
        ax.add_patch(c)

    # Plot outlier blobs
    for idx in np.where(streak_spline.inlier_indices == False)[0]:
        blob = streak_spline.blobs[idx]
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        ax.add_patch(c)

    # Predict y values using the fitted model
    x_values: np.ndarray = np.linspace(0, image.shape[1], 1000)
    y_values = streak_spline.predict(x_values.reshape(-1, 1))

    # Check if the predicted y-values fall within the image dimensions
    valid_indices = np.where((y_values >= 0) & (y_values < image.shape[0]))

    # Plot the Linear Regression line
    if len(valid_indices[0]) > 0:
        ax.plot(
            x_values[valid_indices],
            y_values[valid_indices],
            color='orange',
            linestyle='-',
            linewidth=2
        )

    x_mid = streak_spline.midpoint[0]
    print("Spline midpoint:", streak_spline.midpoint)
    gradient_mid = streak_spline.gradient(x_mid)
    print(f"Gradient (slope) of the spline at x={x_mid} is {gradient_mid}")

    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()