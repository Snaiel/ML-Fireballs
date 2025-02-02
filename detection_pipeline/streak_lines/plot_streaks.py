import argparse
import json
import os
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import skimage as ski

from detection_pipeline.streak_lines.utils import create_streak_line


def main():
    @dataclass
    class Args:
        folder_path: str
    
    parser = argparse.ArgumentParser(
        description="Plot a .differenced.jpg detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("folder_path", type=str, help="Detection outputs folder containing subfolders for each image")

    args = Args(**vars(parser.parse_args()))
    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    folder_path = Path(args.folder_path)

    subfolders = sorted([d for d in folder_path.iterdir() if d.is_dir()])

    with open(Path(subfolders[0], subfolders[0].name + ".json")) as json_file:
        json_data = json.load(json_file)

    original_image = Path(subfolders[0], json_data["original_image"])

    detections_images: list[Path] = []

    for subfolder in subfolders:
        with open(Path(subfolder, subfolder.name + ".json")) as json_file:
            json_data = json.load(json_file)
        for detection in map(lambda x: x["name"], json_data["detections"]):
            detections_images.append(Path(subfolder, detection + ".differenced.jpg"))

    original = ski.io.imread(original_image)

    _, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(original, cmap="gray", aspect="equal")
    ax.set_title(f"Detections on {folder_path.name}")

    colors = ["blue", "yellow", "orange", "green", "red", "purple", "cyan"]

    with Pool() as pool:
        streak_lines = pool.map(create_streak_line, detections_images)

    print("Invalid lines:")

    for idx, streak in enumerate(streak_lines):

        if not streak.is_valid:
            print(detections_images[idx].name)
            continue

        start_point = streak.startpoint
        end_point = streak.endpoint

        ax.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            linestyle="-",
            color=colors[idx % len(colors)]
        )

        ax.text(
            streak.midpoint[0],
            streak.midpoint[1],
            streak.number,
            color="white"
        )

    plt.axis("off")
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()
