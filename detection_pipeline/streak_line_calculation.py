from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from skimage.feature import blob_dog
from sklearn.linear_model import LinearRegression, RANSACRegressor


class StreakLine:
    
    _blobs: np.ndarray
    _ransac: RANSACRegressor


    def __init__(self, image: str | Path | np.ndarray):
        if not isinstance(image, np.ndarray):
            image = ski.io.imread(image, as_gray=True)

        # Detect blobs in the image
        blobs_dog = blob_dog(image, min_sigma=2, max_sigma=10, threshold=0.015)
        blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)  # Compute radii in the 3rd column
        self._blobs = blobs_dog

        if not self.is_valid:
            return

        self._y_coords = blobs_dog[:, 0]
        self._x_coords = blobs_dog[:, 1]

        # RANSAC with Linear Regression
        self._ransac = RANSACRegressor(estimator=LinearRegression(), residual_threshold=5, max_trials=100)
        self._ransac.fit(self._x_coords.reshape(-1, 1), self._y_coords)


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
            float(self.predict(np.array([[x_mid]]))[0]),
        )


    @property
    def gradient(self) -> float:
        return self._ransac.estimator_.coef_[0]


    def predict(self, *args, **kwargs):
        return self._ransac.predict(*args, **kwargs)



def main():
    image_path = Path(
        "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL48/48_2015-11-01_110359_DSC_9000/48_2015-11-01_110359_DSC_9000_57_3319-91-3607-165.differenced.jpg"
    )

    streak_line = StreakLine(image_path)

    image = ski.io.imread(image_path, as_gray=True)

    _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray", aspect="equal")

    if not streak_line.is_valid:
        print("Less than 3 blobs, not performing line calculation.")
        return

    ax.set_title("Fitting a Straight Line to Streak Blobs Using RANSAC and Linear Regression")

    # Plot inlier blobs
    for idx in np.where(streak_line.inlier_indices)[0]:
        blob = streak_line.blobs[idx]
        y, x, r = blob
        c = plt.Circle((x, y), r, color="lime", linewidth=2, fill=False)
        ax.add_patch(c)

    # Plot outlier blobs
    for idx in np.where(~streak_line.inlier_indices)[0]:
        blob = streak_line.blobs[idx]
        y, x, r = blob
        c = plt.Circle((x, y), r, color="red", linewidth=2, fill=False)
        ax.add_patch(c)

    # Predict y values using the fitted model
    x_values = np.linspace(0, image.shape[1], 1000).reshape(-1, 1)
    y_values = streak_line.predict(x_values)

    # Check if the predicted y-values fall within the image dimensions
    valid_indices = np.where((y_values >= 0) & (y_values < image.shape[0]))

    # Plot the fitted line
    if len(valid_indices[0]) > 0:
        ax.plot(
            x_values[valid_indices],
            y_values[valid_indices],
            color="orange",
            linestyle="-",
            linewidth=2,
        )

    print("Line midpoint:", streak_line.midpoint)
    print(f"Gradient (slope) of the line is {streak_line.gradient}")

    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
