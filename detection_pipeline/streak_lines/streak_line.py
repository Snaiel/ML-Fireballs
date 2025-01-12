from __future__ import annotations

import math

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
        print(image)
        self._coords = [float(i) for i in image.split("_")[-1].removesuffix(".differenced.jpg").split("-")]
        print(self._coords)

        if not isinstance(image, np.ndarray):
            image = ski.io.imread(image, as_gray=True)

        # Detect blobs in the image
        blobs_dog = blob_dog(image, min_sigma=2, max_sigma=10, threshold=0.015)
        blobs_dog[:, 2] = blobs_dog[:, 2] * math.sqrt(2)  # Compute radii in the 3rd column
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
    def startpoint_cropped(self) -> tuple:
        x = float(self._x_coords.min())
        y = float(self.compute_y_cropped(np.array([[x]]))[0])
        return (x, y)


    @property
    def startpoint(self) -> tuple:
        xy = self.startpoint_cropped
        return (xy[0] + self._coords[0], xy[1]  + self._coords[1])


    @property
    def endpoint_cropped(self) -> tuple:
        x = float(self._x_coords.max())
        y = float(self.compute_y_cropped(np.array([[x]]))[0])
        return (x, y)


    @property
    def endpoint(self) -> tuple:
        xy = self.endpoint_cropped
        return (xy[0] + self._coords[0], xy[1] + self._coords[1])


    @property
    def midpoint_cropped(self) -> tuple:
        x_mid = (self._x_coords.min() + self._x_coords.max()) / 2
        return (
            float(x_mid),
            float(self.compute_y_cropped(np.array([[x_mid]]))[0]),
        )


    @property
    def midpoint(self) -> tuple:
        xy = self.midpoint_cropped
        return (
            xy[0] + self._coords[0],
            xy[1] + self._coords[1]
        )


    @property
    def gradient(self) -> float:
        estimator: LinearRegression = self._ransac.estimator_
        return estimator.coef_[0]


    @property
    def angle(self) -> float:
        angle_radians = math.atan(self.gradient)
        angle_degrees = math.degrees(angle_radians)
        return angle_degrees


    @property
    def coords(self) -> tuple:
        return self._coords


    def compute_y_cropped(self, *args, **kwargs):
        return self._ransac.predict(*args, **kwargs)


    def compute_y(self, *args, **kwargs):
        return self.compute_y_cropped(*args, **kwargs)
    

    def distance(self, point1: tuple, point2: tuple) -> float:
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


    def midpoint_to_point(self, point: tuple) -> float:
        return self.distance(self.midpoint, point)


    def midpoint_to_midpoint(self, streak_line: StreakLine) -> float:
        return self.midpoint_to_point(streak_line.midpoint)