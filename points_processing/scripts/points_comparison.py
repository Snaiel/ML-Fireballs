import os
import time
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from pipelines.full_auto import FullAutoFireball, retrieve_fireball
from skimage import io, transform
from dataclasses import dataclass
import numpy as np


@dataclass
class FireballPickingsComparison:
    fireball_name: str
    crop: tuple[tuple[int, int], tuple[int, int]]
    image_path: str
    csv_path: str
    image: np.ndarray
    fireball:  FullAutoFireball
    auto_df: pd.DataFrame
    manual_df: pd.DataFrame


SCALE_FACTOR = 4


## Fireball Cropping
# manual croppings for now
DFN_FIREBALL_CROPPINGS = {
    "025_Elginfield": ((4900, 1150), (5600, 2800)),
    "044_Vermilion": ((1050, 3040), (2250, 3690)),
    "051_Kanandah": ((2500, 370), (3200, 500)),
    "071_CAO_RASC": ((4950, 2220), (6300, 2430))
}


def retrieve_comparison(fireball_name: str = "071_CAO_RASC") -> FireballPickingsComparison:

    dfn_highlights_folder = Path(Path(__file__).parents[2], "data", "dfn_highlights")
    for folder in os.listdir(dfn_highlights_folder):
        if folder == fireball_name:
            for file in os.listdir(Path(dfn_highlights_folder, folder)):
                if file.endswith("-G.jpeg"):
                    image_path = Path(dfn_highlights_folder, folder, file)
                if file.endswith(".csv"):
                    csv_path = Path(dfn_highlights_folder, folder, file)

    current_fireball_croppings = DFN_FIREBALL_CROPPINGS[fireball_name]

    image = io.imread(image_path)
    cropped_image = image[
        current_fireball_croppings[0][1]:current_fireball_croppings[1][1],
        current_fireball_croppings[0][0]:current_fireball_croppings[1][0]
    ]
    print(cropped_image.shape)

    ## Upscaling image
    scaled_image = transform.resize(
        cropped_image,
        (cropped_image.shape[0] * SCALE_FACTOR, cropped_image.shape[1] * SCALE_FACTOR),
        preserve_range=True
    )

    ## Picking points
    start_time = time.time()
    fireball = retrieve_fireball(scaled_image, min_radius=12, threshold=4)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time)

    print("Result")
    points = fireball.fireball_points


    # Create dataframe from auto pickings
    auto_df = pd.DataFrame(columns=['auto_x', 'auto_y', 'de_bruijn_index', 'zero_or_one'])

    for point in points:
        x = (point.x / SCALE_FACTOR) + current_fireball_croppings[0][0]
        y = (point.y / SCALE_FACTOR) + current_fireball_croppings[0][1]
        auto_df = pd.concat(
            [auto_df, pd.DataFrame([[x, y, point.de_bruijn_pos, point.de_bruijn_val]],
            columns=['auto_x', 'auto_y', 'de_bruijn_index', 'zero_or_one'])],
            ignore_index=True
        )

    print(auto_df.to_string(index=False))


    ## Read manual pickings
    pd.set_option('display.max_columns', None)
    manual_df = pd.read_csv(csv_path)
    manual_df = manual_df.iloc[:, [0, 1, 4, 6]]

    manual_df = manual_df.rename(columns={
        "x_image": "manual_x",
        "y_image": "manual_y",
        "de_bruijn_sequence_element_index": "de_bruijn_index"
    })

    print(manual_df.to_string(index=False))

    return FireballPickingsComparison(
        fireball_name,
        current_fireball_croppings,
        image_path,
        csv_path,
        image,
        fireball,
        auto_df,
        manual_df
    )


def visual_comparison(comparison: FireballPickingsComparison, show_plot: bool = True) -> None:
    ## Visualise Fireball
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_title(comparison.fireball_name)


    # Plot the image
    ax.imshow(comparison.fireball.image, cmap='gray', aspect='equal')


    # Plot Blobs
    for node in comparison.fireball.fireball_blobs:
        x, y, r = node
        # blob radius
        outer_circle = patches.Circle((x, y), r, color='lime', linewidth=2, fill=False)
        # Add the patches to the axis
        ax.add_patch(outer_circle)

    # Plot removed blobs based on size and brightness
    for node in comparison.fireball.removed_blobs_size_brightness:
        x, y, r = node
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        ax.add_patch(c)

    ## Distance Groups
    cumulative_node_count = 0
    for i in range(len(comparison.fireball.distance_groups)):
        node = comparison.fireball.fireball_blobs[cumulative_node_count]
        x, y, r = node
        c = plt.Circle((x, y), r, color='pink', linewidth=2, fill=False)
        ax.add_patch(c)

        cumulative_node_count += len(comparison.fireball.distance_groups[i])

    # Plot Auto Pickings
    for index, row in comparison.auto_df.iterrows():
        x = row['auto_x']
        y = row['auto_y']

        x = (x - comparison.crop[0][0]) * SCALE_FACTOR
        y = (y - comparison.crop[0][1]) * SCALE_FACTOR

        ax.add_patch(
            patches.Circle((x, y), 0.5, color='lime', fill=True)
        )

        ax.text(x, y + 50, int(row['zero_or_one']), color="pink").set_clip_on(True)
        ax.text(x, y + 100, int(row['de_bruijn_index']), color="pink").set_clip_on(True)


    # Plot Manual Pickings
    for index, row in comparison.manual_df.iterrows():
        x = row['manual_x']
        y = row['manual_y']

        x = (x - comparison.crop[0][0]) * SCALE_FACTOR
        y = (y - comparison.crop[0][1]) * SCALE_FACTOR

        if comparison.fireball.rotated:
            x, y = y, x
            y = comparison.fireball.image.shape[0] - y

        ax.add_patch(
            patches.Circle((x, y), 0.5, color='red', fill=True)
        )

        ax.text(x, y - 30, int(row['zero_or_one']), color="white").set_clip_on(True)
        ax.text(x, y - 80, int(row['de_bruijn_index']), color="white").set_clip_on(True)

    if show_plot:
        plt.show()


def main():
    comparison = retrieve_comparison()
    visual_comparison(comparison, True)


if __name__ == "__main__":
    main()