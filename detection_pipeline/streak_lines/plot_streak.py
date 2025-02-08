import argparse
import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.axes import Axes

from detection_pipeline.streak_lines import StreakLine


def main():
    @dataclass
    class Args:
        image_path: str
    
    parser = argparse.ArgumentParser(
        description="Plot a .differenced.jpg detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("image_path", type=str, help="Path to the .differenced.jpg detection")

    args = Args(**vars(parser.parse_args()))
    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    streak_line = StreakLine(args.image_path)
    image: np.ndarray = ski.io.imread(args.image_path)

    ax1: Axes
    ax2: Axes
    ax3: Axes

    # Create three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)  # Shared zoom

    # --- Plot 1: Original Image ---
    ax1.imshow(image, cmap="gray", aspect="equal")
    ax1.set_title("Original Image")
    ax1.axis("off")

    # --- Plot 2: Sample Weights (Brightness-Based Scatter) ---
    ax2.scatter(
        streak_line._x_coords, streak_line._y_coords,
        c=streak_line.brightnesses, cmap="viridis", edgecolor='black'
    )
    ax2.set_title("Sample Weights")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.invert_yaxis()

    # --- Plot 3: RANSAC Fit ---
    ax3.imshow(image, cmap="gray", aspect="equal")
    ax3.set_title("Weighted RANSAC Line Fit")

    # Plot inlier blobs
    for idx in np.where(streak_line.inlier_indices)[0]:
        blob = streak_line.blobs[idx]
        x, y, r = blob
        c = plt.Circle((x, y), r, color="lime", linewidth=2, fill=False)
        ax3.add_patch(c)

    # Plot outlier blobs
    for idx in np.where(~streak_line.inlier_indices)[0]:
        blob = streak_line.blobs[idx]
        x, y, r = blob
        c = plt.Circle((x, y), r, color="red", linewidth=2, fill=False)
        ax3.add_patch(c)

    # Predict y values using the fitted model
    x_values: np.ndarray = np.linspace(0, image.shape[1], 1000)
    x_values = x_values.reshape(-1, 1)
    y_values = streak_line.compute_y(x_values)

    # Check if the predicted y-values fall within the image dimensions
    valid_indices = np.where((y_values >= 0) & (y_values < image.shape[0]))

    # Plot the fitted line
    if len(valid_indices[0]) > 0:
        ax3.plot(
            x_values[valid_indices],
            y_values[valid_indices],
            color="orange",
            linestyle="-",
            linewidth=2,
        )

    ax3.axis("off")

    # --- Ensure All Axes Share the Same View ---
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.set_aspect("equal")

    # Adjust layout
    plt.tight_layout()
    plt.show()

    print("Line midpoint:", streak_line.midpoint)
    print(f"Gradient (slope) of the line is {streak_line.gradient}")


if __name__ == "__main__":
    main()
