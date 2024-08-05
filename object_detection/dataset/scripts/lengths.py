import inspect
import os
import statistics
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dataset import GFO_PICKINGS
from object_detection.dataset.point_pickings import PointPickings
from view.points import show_points


def main():
    lengths = []
    long_fireballs = []

    condition = lambda l: l > 1000

    for pickings_csv in os.listdir(GFO_PICKINGS):
        pp = PointPickings(Path(GFO_PICKINGS, pickings_csv))
        length = sqrt((pp.pp_max_x - pp.pp_min_x)**2 + (pp.pp_max_y - pp.pp_min_y)**2)
        if condition(length):
            long_fireballs.append((pickings_csv, length))
        lengths.append(length)

    # Convert the list to a NumPy array
    lengths_array = np.array(lengths)

    # Calculate mean
    mean_length = statistics.mean(lengths)
    print(f"Mean length: {mean_length}")

    # Calculate median
    median_length = statistics.median(lengths)
    print(f"Median length: {median_length}")

    # Min
    print(f"Min: {min(lengths)}")

    # Max
    print(f"Max: {max(lengths)}")

    # Calculate the range
    data_range = max(lengths) - min(lengths)
    print(f"Range: {data_range}")

    # Calculate standard deviation
    std_dev = statistics.stdev(lengths)
    print(f"Standard deviation: {std_dev}")

    # Calculate interquartile range (IQR)
    lengths_sorted = sorted(lengths)
    q1 = np.percentile(lengths_sorted, 25)
    q3 = np.percentile(lengths_sorted, 75)
    iqr = q3 - q1
    print(f"Q1 (25th percentile): {q1}")
    print(f"Q3 (75th percentile): {q3}")
    print(f"Interquartile range (IQR): {iqr}")

    # Create a figure and axis for the first histogram
    fig1, ax1 = plt.subplots()
    ax1.hist(lengths_array, bins=100, range=(0, 7500), edgecolor='black')
    ax1.set_title('Histogram of Fireball Lengths')
    ax1.set_xlabel('Length')
    ax1.set_ylabel('Frequency')

    # Create a figure and axis for the second histogram
    fig2, ax2 = plt.subplots()
    ax2.hist(lengths_array, bins=100, range=(0, 1000), edgecolor='black')
    ax2.set_title('Histogram of Fireball Lengths')
    ax2.set_xlabel('Length')
    ax2.set_ylabel('Frequency')

    # Show the figures separately
    plt.show()


    print(f"Condition: {''.join(inspect.getsourcelines(condition)[0]).strip()}")
    print(f"Number passed condition: {len(long_fireballs)}")

    for long in long_fireballs:
        fireball_name = long[0].split(".")[0]
        print(fireball_name, long[1])
        show_points(fireball_name)



if __name__ == "__main__":
    main()