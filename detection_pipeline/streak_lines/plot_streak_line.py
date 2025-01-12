import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

from detection_pipeline.streak_lines import StreakLine


def main():
    image_path = "/home/snaiel/Dropbox/Curtin/Year 3/NPSC3000 Research, Leadership and Entrepreneurship in Science 2/Fireballs/ML-Fireballs/data/detections_dfn-l0-20151101/dfn-l0-20151101/DFNSMALL41/41_2015-11-01_110059_DSC_0205/41_2015-11-01_110059_DSC_0205_45_1237-2122-1364-2426.differenced.jpg",

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


if __name__ == "__main__":
    main()