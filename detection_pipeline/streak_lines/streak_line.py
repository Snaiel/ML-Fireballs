from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import skimage as ski
from skimage.feature import blob_dog
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LinearRegression, RANSACRegressor

from utils.constants import RANDOM_SEED


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class StreakLine:

    _blobs: np.ndarray
    _is_valid: bool
    _ransac: RANSACRegressor
    _y_coords: np.ndarray
    _x_coords: np.ndarray
    _coords: tuple
    _number: int


    def __init__(self, image_path: str | Path):
        image_path = str(image_path)
        self._coords = [float(i) for i in image_path.split("_")[-1].removesuffix(".differenced.jpg").split("-")]
        self._number = int(image_path.split("_")[-3])

        self._is_valid = True

        image = ski.io.imread(image_path, as_gray=True)

        blobs_dog = blob_dog(
            image,
            min_sigma=1,
            max_sigma=10, 
            sigma_ratio=5.0,
            threshold_rel=0.2,
            threshold=None,
            overlap=0.5
        )
        
        # Compute radii in the 3rd column
        blobs_dog[:, 2] = blobs_dog[:, 2] * math.sqrt(2) 
        self._blobs = blobs_dog

        if len(self._blobs) < 3:
            self._is_valid = False
            return

        self._y_coords = blobs_dog[:, 0]
        self._x_coords = blobs_dog[:, 1]

        # RANSAC with Linear Regression
        try:
            self._ransac = RANSACRegressor(residual_threshold=5, max_trials=100, random_state=RANDOM_SEED)
            self._ransac.fit(self._x_coords.reshape(-1, 1), self._y_coords)
            if self._ransac.inlier_mask_.sum() < 5:
                self._is_valid = False
        except:
            self._is_valid = False


    @property
    def blobs(self) -> np.ndarray:
        return self._blobs


    @property
    def is_valid(self) -> bool:
        return self._is_valid


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
    def length(self) -> float:
        return self.distance(self.startpoint, self.endpoint)


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


    @property
    def number(self) -> int:
        return self._number


    def compute_y_cropped(self, *args, **kwargs):
        return self._ransac.predict(*args, **kwargs)


    def compute_y(self, *args, **kwargs):
        return self.compute_y_cropped(*args, **kwargs)
    

    def angle_between(self, streak_line: StreakLine) -> float:
        angle_radians = math.atan(
            abs(
                (self.gradient - streak_line.gradient) /
                (1 + (self.gradient * streak_line.gradient))
            )
        )
        angle_degrees = math.degrees(angle_radians)
        return angle_degrees


    def distance(self, point1: tuple, point2: tuple) -> float:
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


    def midpoint_to_point(self, point: tuple) -> float:
        return self.distance(self.midpoint, point)


    def midpoint_to_midpoint(self, streak_line: StreakLine) -> float:
        return self.midpoint_to_point(streak_line.midpoint)
    

    def similar_line(self, streak_line: StreakLine) -> bool:
        
        if self.angle_between(streak_line) > 20:
            return False
        
        longer_streak, shorter_streak = (self, streak_line) if self.length > streak_line.length else (streak_line, self)

        if longer_streak.midpoint_to_midpoint(shorter_streak) > longer_streak.length * 0.3:
            return False
        
        if shorter_streak.length < 0.5 * longer_streak.length:
            return False

        return True


    def same_trajectory(self, streak_line: StreakLine) -> bool:

        offset = abs(self.number - streak_line.number)

        angle_between = self.angle_between(streak_line)

        # print("Difference:", offset, angle_between)

        if angle_between > 30 * offset:
            return False

        estimated_point_1 = (
            self.midpoint[0] + ((1000 * offset) * math.cos(math.radians(self.angle))),
            self.midpoint[1] + ((1000 * offset) * math.sin(math.radians(self.angle)))
        )

        estimated_point_2 = (
            self.midpoint[0] - ((1000 * offset) * math.cos(math.radians(self.angle))),
            self.midpoint[1] - ((1000 * offset) * math.sin(math.radians(self.angle)))
        )

        # print("Estimated points:", estimated_point_1, estimated_point_2)

        dist_to_p1 = self.distance(estimated_point_1, streak_line.midpoint)
        dist_to_p2 = self.distance(estimated_point_2, streak_line.midpoint)

        # print("Distance to estimated points:", dist_to_p1, dist_to_p2)

        if dist_to_p1 > 1000 * offset:
            if dist_to_p2 > 1000 * offset:
                return False

        return True