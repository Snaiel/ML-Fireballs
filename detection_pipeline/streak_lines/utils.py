import os
from multiprocessing import Pool
from pathlib import Path

from detection_pipeline.streak_lines.streak_line import StreakLine


def create_streak_line(detection: Path) -> StreakLine:
    return StreakLine(detection)


def get_streak_lines(camera_folder: Path) -> dict[str, StreakLine]:
    streak_lines_dict = {}

    subfolders = [i for i in sorted(os.listdir(camera_folder)) if Path(camera_folder, i).is_dir()]

    if not subfolders:
        return streak_lines_dict

    detection_images: list[Path] = []

    for subfolder in subfolders:

        detections = [
            i for i in
            sorted(os.listdir(Path(camera_folder, subfolder)))
            if "differenced" in i and "thumb" not in i
        ]

        for detection in detections:
            detection_images.append(Path(camera_folder, subfolder, detection))
    
    with Pool() as pool:
        streak_lines_list = pool.map(create_streak_line, detection_images)

    streak_lines_dict = {
        str(detection.relative_to(camera_folder)).removesuffix(".differenced.jpg"): streak_line
        for detection, streak_line in
        zip(detection_images, streak_lines_list)
    }

    return streak_lines_dict