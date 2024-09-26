import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from object_detection.dataset import DATA_FOLDER
import os


def main() -> None:
    b_sizes_folder = Path(DATA_FOLDER, "bsizes")

    # Initialize lists to store cumulative sums
    sum_box_totals = []
    sum_recalls = []
    sum_precisions = []

    # Step 2: Read and accumulate the data
    for file in os.listdir(b_sizes_folder):
        df = pd.read_csv(Path(b_sizes_folder, file))
        sum_box_totals.append(df['box_totals'])
        sum_recalls.append(df['recall'])
        sum_precisions.append(df['precision'])

    # Normalize each list before converting to DataFrames
    normalized_lists = [lst / lst.iloc[0] for lst in sum_box_totals]

    df_box_totals = pd.concat(sum_box_totals, axis=1).mean(axis=1)
    df_normalised_boxes = pd.concat(normalized_lists, axis=1).mean(axis=1)
    df_recalls = pd.concat(sum_recalls, axis=1).mean(axis=1)
    df_precisions = pd.concat(sum_precisions, axis=1).mean(axis=1)

    # Print statements for debugging
    print("Average Box Totals Across the 5 Splits:")
    print(df_box_totals)
    print("\nAverage Normalized Box Totals Across the 5 Splits:")
    print(df_normalised_boxes)
    print("\nAverage Recalls Across the 5 Splits:")
    print(df_recalls)
    print("\nAverage Precisions Across the 5 Splits:")
    print(df_precisions)

    # Step 3: Plotting
    plt.figure(figsize=(10, 6), dpi=120)
    plt.plot(df_normalised_boxes.index, df_recalls, marker='o', label="Recall")
    plt.plot(df_normalised_boxes.index, df_precisions, marker='x', label='Precision')
    plt.plot(df_normalised_boxes.index, df_normalised_boxes, marker='s', label='Normalised Box Totals')

    plt.title("Average Individual Tile Detections with Different Border Sizes using iom=0.45")
    plt.xlabel("Border Sizes")
    plt.ylabel("Percentage")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()