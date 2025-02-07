from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import skimage as ski
from skimage.feature import blob_dog
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LinearRegression, RANSACRegressor

from utils.constants import (SAME_TRAJECTORY_MAX_ANGLE_DIFFERENCE,
                             SAME_TRAJECTORY_MAX_PARALLEL_DISTANCE,
                             SAME_TRAJECTORY_MAX_PERPENDICULAR_DISTANCE,
                             SIMILAR_LINES_MAX_ANGLE_DIFFERENCE,
                             SIMILAR_LINES_MAX_MIDPOINT_DISTANCE_RATIO,
                             SIMILAR_LINES_MIN_LENGTH_RATIO,
                             STREAK_LINE_BLOB_DETECTION_KWARGS,
                             STREAK_LINE_MIN_BLOBS, STREAK_LINE_MIN_INLIERS,
                             STREAK_LINE_RANSAC_KWARGS)

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

        blobs_dog = blob_dog(image, **STREAK_LINE_BLOB_DETECTION_KWARGS)
        
        # Compute radii in the 3rd column
        blobs_dog[:, 2] = blobs_dog[:, 2] * math.sqrt(2) 
        self._blobs = blobs_dog

        if len(self._blobs) < STREAK_LINE_MIN_BLOBS:
            self._is_valid = False
            return

        self._y_coords = blobs_dog[:, 0]
        self._x_coords = blobs_dog[:, 1]

        # RANSAC with Linear Regression
        try:
            self._ransac = RANSACRegressor(**STREAK_LINE_RANSAC_KWARGS)
            self._ransac.fit(self._x_coords.reshape(-1, 1), self._y_coords)
            if self._ransac.inlier_mask_.sum() < STREAK_LINE_MIN_INLIERS:
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
        """
        Calculates the angle (in degrees) between the current line and another streak line.

        Args:
            streak_line (StreakLine): Another line object to compare with.

        Returns:
            float: The angle between the two lines in degrees.
        """

        # Calculate the angle in radians using the arctangent (atan) function.
        # This formula is derived from the tangent of the angle between two lines:
        #     tan(θ) = |(m1 - m2) / (1 + m1 * m2)|
        # Where:
        # - m1 = self.gradient (slope of the current line)
        # - m2 = streak_line.gradient (slope of the other line)
        # The absolute value ensures the angle is always positive.
        angle_radians = math.atan(
            abs(
                (self.gradient - streak_line.gradient) /
                (1 + (self.gradient * streak_line.gradient))
            )
        )

        # Return the final angle in degrees.
        return math.degrees(angle_radians)


    def distance(self, point1: tuple, point2: tuple) -> float:
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


    def midpoint_to_point(self, point: tuple) -> float:
        return self.distance(self.midpoint, point)


    def midpoint_to_midpoint(self, streak_line: StreakLine) -> float:
        return self.midpoint_to_point(streak_line.midpoint)
    

    def project_point(self, point: tuple):
        """
        Projects a given point onto the line defined by this streak.

        Args:
            point (tuple): The (x, y) coordinates of the point to be projected.

        Returns:
            tuple: The (x, y) coordinates of the projected point.
        """

        # Convert input points to NumPy arrays for easy vector operations
        p = np.array(point)        # The point to be projected
        p0 = np.array(self.midpoint)  # A known point on the line

        # Create the direction vector of the line based on the gradient
        # This represents the change in x and y along the line: (dx, dy) = (1, m)
        d = np.array([1, self.gradient])  

        # Calculate the projection using the vector projection formula:
        # Projection = p0 + [( (p - p0) ⋅ d ) / ( d ⋅ d )] * d
        # - (p - p0) is the vector from the known point to the point to project
        # - The dot product (⋅) measures how much of (p - p0) aligns with d
        # - The scaling factor adjusts d to reach the closest point on the line
        projection: np.ndarray = p0 + (np.dot(p - p0, d) / np.dot(d, d)) * d

        # Convert the NumPy array to a tuple of native Python floats for clean output
        return tuple(map(float, projection))


    def similar_line(self, streak_line: StreakLine) -> bool:
        
        if self.angle_between(streak_line) > SIMILAR_LINES_MAX_ANGLE_DIFFERENCE:
            return False
        
        longer_streak, shorter_streak = (self, streak_line) if self.length > streak_line.length else (streak_line, self)

        if longer_streak.midpoint_to_midpoint(shorter_streak) > SIMILAR_LINES_MAX_MIDPOINT_DISTANCE_RATIO * longer_streak.length:
            return False
        
        if shorter_streak.length < SIMILAR_LINES_MIN_LENGTH_RATIO * longer_streak.length:
            return False

        return True


    def same_trajectory(self, streak_line: StreakLine) -> bool:

        offset = abs(self.number - streak_line.number)
        angle_between = self.angle_between(streak_line)

        if angle_between > SAME_TRAJECTORY_MAX_ANGLE_DIFFERENCE * offset:
            return False

        projected_point = self.project_point(streak_line.midpoint)

        if self.midpoint_to_point(projected_point) > SAME_TRAJECTORY_MAX_PARALLEL_DISTANCE * offset:
            return False

        if streak_line.midpoint_to_point(projected_point) > SAME_TRAJECTORY_MAX_PERPENDICULAR_DISTANCE * offset:
            return False

        return True
