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



def plot_streak_image(image_path: str) -> None:
    streak_line = StreakLine(image_path)
    image: np.ndarray = ski.io.imread(image_path)

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


def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)


def main():
    # original_image = "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL48/48_2015-11-01_110129_DSC_8995/48_2015-11-01_110129_DSC_8995.thumb.jpg"

    # streak_images = [
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL48/48_2015-11-01_110129_DSC_8995/48_2015-11-01_110129_DSC_8995_32_1605-1076-1741-1204.differenced.jpg",
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL48/48_2015-11-01_110159_DSC_8996/48_2015-11-01_110159_DSC_8996_67_1764-797-2071-1054.differenced.jpg",
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL48/48_2015-11-01_110229_DSC_8997/48_2015-11-01_110229_DSC_8997_51_2084-586-2368-783.differenced.jpg",
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL48/48_2015-11-01_110259_DSC_8998/48_2015-11-01_110259_DSC_8998_62_2476-329-2858-520.differenced.jpg",
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL48/48_2015-11-01_110329_DSC_8999/48_2015-11-01_110329_DSC_8999_68_2897-174-3275-317.differenced.jpg",
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL48/48_2015-11-01_110359_DSC_9000/48_2015-11-01_110359_DSC_9000_57_3319-91-3607-165.differenced.jpg",
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL48/48_2015-11-01_110429_DSC_9001/48_2015-11-01_110429_DSC_9001_46_3741-24-4039-72.differenced.jpg"
    # ]

    original_image = "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL41/41_2015-11-01_110059_DSC_0205/41_2015-11-01_110059_DSC_0205.thumb.jpg"

    streak_images = [
        "/home/snaiel/Dropbox/Curtin/Year 3/NPSC3000 Research, Leadership and Entrepreneurship in Science 2/Fireballs/ML-Fireballs/data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL41/41_2015-11-01_110059_DSC_0205/41_2015-11-01_110059_DSC_0205_45_1237-2122-1364-2426.differenced.jpg",
        "/home/snaiel/Dropbox/Curtin/Year 3/NPSC3000 Research, Leadership and Entrepreneurship in Science 2/Fireballs/ML-Fireballs/data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL41/41_2015-11-01_110129_DSC_0206/41_2015-11-01_110129_DSC_0206_37_1367-1748-1551-2088.differenced.jpg",
        "/home/snaiel/Dropbox/Curtin/Year 3/NPSC3000 Research, Leadership and Entrepreneurship in Science 2/Fireballs/ML-Fireballs/data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL41/41_2015-11-01_110159_DSC_0207/41_2015-11-01_110159_DSC_0207_66_1586-1357-1803-1681.differenced.jpg",
        "/home/snaiel/Dropbox/Curtin/Year 3/NPSC3000 Research, Leadership and Entrepreneurship in Science 2/Fireballs/ML-Fireballs/data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL41/41_2015-11-01_110229_DSC_0208/41_2015-11-01_110229_DSC_0208_53_1822-981-2113-1324.differenced.jpg",
        "/home/snaiel/Dropbox/Curtin/Year 3/NPSC3000 Research, Leadership and Entrepreneurship in Science 2/Fireballs/ML-Fireballs/data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL41/41_2015-11-01_110259_DSC_0209/41_2015-11-01_110259_DSC_0209_79_2144-651-2474-964.differenced.jpg",
        "/home/snaiel/Dropbox/Curtin/Year 3/NPSC3000 Research, Leadership and Entrepreneurship in Science 2/Fireballs/ML-Fireballs/data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL41/41_2015-11-01_110329_DSC_0210/41_2015-11-01_110329_DSC_0210_75_2503-386-2842-624.differenced.jpg"
    ]

    # original_image = "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL25/25_2015-11-01_101629_DSC_0035/25_2015-11-01_101629_DSC_0035.thumb.jpg"

    # streak_images = [
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL25/25_2015-11-01_101629_DSC_0035/25_2015-11-01_101629_DSC_0035_47_4557-3224-4639-3403.differenced.jpg",
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL25/25_2015-11-01_101758_DSC_0038/25_2015-11-01_101758_DSC_0038_45_3993-4231-4114-4407.differenced.jpg"
    # ]

    # original_image = "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL25/25_2015-11-01_103259_DSC_0068/25_2015-11-01_103259_DSC_0068.thumb.jpg"

    # streak_images = [
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL25/25_2015-11-01_103259_DSC_0068/25_2015-11-01_103259_DSC_0068_55_3274-3204-3439-3646.differenced.jpg",
    #     "data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL25/25_2015-11-01_103329_DSC_0069/25_2015-11-01_103329_DSC_0069_67_3083-3735-3244-4181.differenced.jpg"
    # ]

    original = ski.io.imread(original_image)

    # Initialize the plot
    _, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(original, cmap="gray", aspect="equal")
    ax.set_title("Start and End Points of Streaks on Original Image")

    colors = ["blue", "yellow", "orange", "green", "red", "purple", "cyan"]  # Colors for different streaks

    streak_lines = [StreakLine(s) for s in streak_images]

    for idx, streak in enumerate(streak_lines):

        print(streak.midpoint, streak.gradient)

        if idx > 0:
            prev_streak = streak_lines[idx - 1]
            print(distance(streak.midpoint, prev_streak.midpoint), abs(streak.gradient - prev_streak.gradient))

        if not streak.is_valid:
            print(f"Streak image {idx + 1} has less than 3 blobs. Skipping.")
            continue

        # Get start and end points
        start_point = streak.start_point
        end_point = streak.end_point

        # Draw a line connecting the start and end points
        ax.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            linestyle="-",
            color=colors[idx % len(colors)],
            label=f"Streak {idx + 1} Line",
        )

    # Configure plot
    ax.legend(loc="upper right")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    


if __name__ == "__main__":
    main()
