import os
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import skimage as ski

from detection_pipeline.streak_lines.utils import create_streak_line


def main():
    folder_path = "data/detections_dfn-l0-20151101/dfn-l0-20151101/"

    camera_folders = [i for i in sorted(os.listdir(folder_path)) if Path(folder_path, i).is_dir()]

    for camera in camera_folders:

        camera_folder = Path(folder_path, camera)

        subfolders = [i for i in sorted(os.listdir(camera_folder)) if "log" not in i]

        if not subfolders:
            continue

        original_image = Path(camera_folder, subfolders[0], subfolders[0] + ".thumb.jpg")

        detections_images = []

        for subfolder in subfolders:
            detections = [i for i in sorted(os.listdir(Path(camera_folder, subfolder))) if "differenced" in i and "thumb" not in i]
            for detection in detections:
                detections_images.append(Path(camera_folder, subfolder, detection))

        original = ski.io.imread(original_image)

        _, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(original, cmap="gray", aspect="equal")
        ax.set_title(f"Detections on {Path(camera_folder).name}")

        colors = ["blue", "yellow", "orange", "green", "red", "purple", "cyan"]

        with Pool() as pool:
            streak_lines = pool.map(create_streak_line, detections_images)

        for idx, streak in enumerate(streak_lines):

            if not streak.is_valid:
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
