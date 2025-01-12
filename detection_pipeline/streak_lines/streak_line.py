from math import sqrt

import numpy as np
import skimage as ski
from skimage.feature import blob_dog
from sklearn.linear_model import LinearRegression, RANSACRegressor


class StreakLine:

    _blobs: np.ndarray
    _ransac: RANSACRegressor
    _y_coords: np.ndarray
    _x_coords: np.ndarray
    _coords: tuple


    def __init__(self, image: str):
        self._coords = [float(i) for i in image.split("_")[-1].removesuffix(".differenced.jpg").split("-")]
        print(self._coords)

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
        self._ransac = RANSACRegressor(residual_threshold=5, max_trials=100)
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
    def start_point(self) -> tuple:
        x1 = float(self._x_coords.min()) 
        y1 = float(self.predict(np.array([[x1]]))[0])
        return (x1 + self._coords[0], y1  + self._coords[1])


    @property
    def end_point(self) -> tuple:
        x2 = float(self._x_coords.max())
        y2 = float(self.predict(np.array([[x2]]))[0])
        return (x2 + self._coords[0], y2 + self._coords[1])


    @property
    def midpoint(self) -> tuple:
        x_mid = (self._x_coords.min() + self._x_coords.max()) / 2
        return (
            float(x_mid) + self._coords[0],
            float(self.predict(np.array([[x_mid]]))[0]) + self._coords[1],
        )


    @property
    def gradient(self) -> float:
        estimator: LinearRegression = self._ransac.estimator_
        return estimator.coef_[0]


    @property
    def coords(self) -> tuple:
        return self._coords


    def predict(self, *args, **kwargs):
        return self._ransac.predict(*args, **kwargs)