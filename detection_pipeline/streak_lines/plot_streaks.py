import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import skimage as ski
from matplotlib.axes import Axes

from detection_pipeline.core import FilteredDetections


def main():
    @dataclass
    class Args:
        folder_path: str
    
    parser = argparse.ArgumentParser(
        description="Plot streaks from a camera folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("folder_path", type=str, help="Detection outputs folder containing subfolders for each image")

    args = Args(**vars(parser.parse_args()))
    print("\nargs:", json.dumps(vars(args), indent=4), "\n")


    folder_path = Path(args.folder_path)
    subfolders = sorted([d for d in folder_path.iterdir() if d.is_dir()])


    with open(Path(subfolders[0], subfolders[0].name + ".json")) as json_file:
        json_data = json.load(json_file)

    if "original_image" in json_data:
        original_image = Path(subfolders[0], json_data["original_image"])
    else:
        original_image = list(subfolders[0].glob("*.thumb.jpg"))[0]


    detections_images: list[Path] = []

    filtered_detections = FilteredDetections(folder_path)
    for i in filtered_detections.final_detections:
        detections_images.append(Path(folder_path, i + ".differenced.jpg"))

    original = ski.io.imread(original_image)


    ax1: Axes
    ax2: Axes

    # Create two side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))  # 1 row, 2 columns

    # --- Plot 1: All Detections ---
    ax1.imshow(original, cmap="gray", aspect="equal")
    ax1.set_title(f"All Detections on {folder_path.name}")

    colors = ["blue", "yellow", "orange", "green", "red", "purple", "cyan"]

    print("Invalid lines:")

    for idx, detection in enumerate(filtered_detections.all_detections):
        streak = filtered_detections.streak_lines[detection]

        if not streak.is_valid:
            print(detection)
            continue

        start_point = streak.startpoint
        end_point = streak.endpoint

        ax1.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            linestyle="-",
            color=colors[idx % len(colors)]
        )

        ax1.text(
            streak.midpoint[0],
            streak.midpoint[1],
            streak.number,
            color="white"
        )

    ax1.axis("off")

    # --- Plot 2: Final Detections ---
    ax2.imshow(original, cmap="gray", aspect="equal")
    ax2.set_title(f"Final Detections on {folder_path.name}")

    for idx, detection in enumerate(filtered_detections.final_detections):
        streak = filtered_detections.streak_lines[detection]

        start_point = streak.startpoint
        end_point = streak.endpoint

        ax2.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            linestyle="-",
            color=colors[idx % len(colors)]
        )

        ax2.text(
            streak.midpoint[0],
            streak.midpoint[1],
            streak.number,
            color="white"
        )

    ax2.axis("off")

    # Ensure layout is properly adjusted
    plt.tight_layout()
    plt.show()

    

if __name__ == "__main__":
    main()
