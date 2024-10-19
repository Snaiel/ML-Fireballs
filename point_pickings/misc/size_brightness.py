import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from point_pickings.compare import (FireballPickingsComparison,
                                    retrieve_comparison, visual_comparison)


def main():
    comparison: FireballPickingsComparison = retrieve_comparison("071_CAO_RASC")
    visual_comparison(comparison, False)

    ## Visualise Brightnesses
    blobs_x_values = range(1, len(comparison.fireball.brightnesses) + 1)

    brightness_series = pd.Series(comparison.fireball.brightnesses)
    brightness_moving_avg = brightness_series.rolling(window=5, center=True).mean()
    brightness_percent_difference = ((brightness_series - brightness_moving_avg) / brightness_moving_avg) * 100

    # Create figure and axes
    ax2: Axes
    _, ax2 = plt.subplots()

    # Plot the data
    ax2.plot(blobs_x_values, comparison.fireball.brightnesses, marker='o', label="Blob Brightness")

    # Plot the moving average
    ax2.plot(blobs_x_values, brightness_moving_avg, color='red', linestyle='--', label="Moving Average")
    ax2.plot(blobs_x_values, brightness_percent_difference, color='orange', linestyle='--', label="% Difference from Moving Average")

    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys())

    # Set labels and title
    ax2.set_title('Blob Brightnesses')
    ax2.set_xlabel('Blob')
    ax2.set_ylabel('Brightness')

    # Add grid
    ax2.grid(True)


    ## Visualise Blob sizes
    size_series = pd.Series(comparison.fireball.fireball_blobs[:, 2])
    size_moving_avg = size_series.rolling(window=5, center=True).mean()
    size_percent_difference = ((size_series - size_moving_avg) / size_moving_avg) * 100

    # Create figure and axes
    ax3: Axes
    _, ax3 = plt.subplots()

    # Plot the data
    ax3.plot(blobs_x_values, comparison.fireball.fireball_blobs[:, 2], marker='o', label="Blob Size")

    # Plot the moving average
    ax3.plot(blobs_x_values, size_moving_avg, color='red', linestyle='--', label="Moving Average")
    ax3.plot(blobs_x_values, size_percent_difference, color='orange', linestyle='--', label="% Difference from Moving Average")

    handles, labels = ax3.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax3.legend(by_label.values(), by_label.keys())

    ax3.set_title('Blob Sizes')
    ax3.set_xlabel('Blob')
    ax3.set_ylabel('Radius')
    ax3.grid(True)



    ## Visualise mean blob size and brightness moving averages
    ax4: Axes
    _, ax4 = plt.subplots()

    ax4.plot(blobs_x_values, (brightness_percent_difference + size_percent_difference) / 2, color='lime', linestyle='--', label="Combined % Difference")
    ax4.axhline(y=-20, color='red', linestyle='-', label="Threshold")

    handles, labels = ax4.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax4.legend(by_label.values(), by_label.keys())

    ax4.set_title('Mean Percentage Difference of Blob Size and Brightness Moving Averages')
    ax4.set_xlabel('Blob')
    ax4.set_ylabel('Percentage Difference')
    ax4.grid(True)

    # Display the figures
    plt.show()


if __name__ == "__main__":
    main()