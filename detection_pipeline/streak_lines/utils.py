import os
from multiprocessing import Pool
from pathlib import Path

from detection_pipeline.streak_lines.streak_line import StreakLine


def create_streak_line(detection: Path) -> StreakLine:
    """
    detection is the path to the differenced detection. e.g.

    ```txt
    41_2015-01-14_110659_DSC_0108_20_5148-2666-5216-2805.differenced.jpg 
    ```
    """
    return StreakLine(detection)


def get_streak_lines(camera_folder: Path) -> dict[str, StreakLine]:
    """
    camera_folder is the folder containing subfolders for each image

    ```txt
    DFNSMALL41 <- this one
        41_2015-01-14_110659_DSC_0108
            41_2015-01-14_110659_DSC_0108.json
            41_2015-01-14_110659_DSC_0108.thumb.differenced.jpg 
            41_2015-01-14_110659_DSC_0108.thumb.jpg 
            41_2015-01-14_110659_DSC_0108_20_5148-2666-5216-2805.differenced.jpg 
            41_2015-01-14_110659_DSC_0108_20_5148-2666-5216-2805.jpg
        41_2015-01-14_110729_DSC_0109
        41_2015-01-14_110959_DSC_0114
    ```

    returns dictionary of detection name with streaks

    e.g.
    ```txt
    {
        "41_2015-01-14_110659_DSC_0108/41_2015-01-14_110659_DSC_0108_20_5148-2666-5216-2805": StreakLine(...)
    }
    ```
    """

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
        detection.relative_to(camera_folder).as_posix().removesuffix(".differenced.jpg"): streak_line
        for detection, streak_line in
        zip(detection_images, streak_lines_list)
    }

    return streak_lines_dict