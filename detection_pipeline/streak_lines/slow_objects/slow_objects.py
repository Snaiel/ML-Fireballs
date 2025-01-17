import os
from pathlib import Path

from detection_pipeline.streak_lines import StreakLine


def find_slow_objects(camera_folder: Path, streak_lines: dict[str, StreakLine] = {}) -> list[list[str]]:
    """
        Goes through each detection from a camera's night of images and compares
        its streak lines with the lines from neighbouring detections.

        Neighbouring detections means detections from images taken recently in
        the future (e.g. checking 5 images ahead).

        Returns a list of lists of detections where each sublist contains detections
        whose lines follow a similar to each other.
    """

    subfolders = [
        i for i in
        sorted(os.listdir(camera_folder))
        if (Path(camera_folder, i)).is_dir()
    ]

    groups: list[list] = []

    for i, subfolder in enumerate(subfolders):
        
        number = int(subfolder.split("_")[-1])
        neighbours = [
            n for n in
            subfolders[i+1:max(len(subfolders),i+3+1)]
            if abs(int(n.split("_")[-1]) - number) <= 5
        ]

        if not neighbours:
            continue

        differenced_detections = [
            i for i in
            sorted(os.listdir(Path(camera_folder, subfolder)))
            if "differenced" in i and not "thumb" in i
        ]

        for current_detection_image in differenced_detections:
            same_trajectory = False

            current_detection = f"{subfolder}/{current_detection_image.removesuffix('.differenced.jpg')}"

            if current_detection not in streak_lines:
                streak_lines[current_detection] = StreakLine(Path(camera_folder, subfolder, current_detection_image))

            current_streak: StreakLine = streak_lines[current_detection]

            if not current_streak.is_valid:
                continue
        
            for neighbour in neighbours:

                if same_trajectory:
                    break
                
                neighbour_detection_images = [
                    i for i in
                    sorted(os.listdir(Path(camera_folder, neighbour)))
                    if "differenced" in i and not "thumb" in i
                ]

                for neighbour_detection_image in neighbour_detection_images:

                    neighbour_detection = f"{neighbour}/{neighbour_detection_image.removesuffix('.differenced.jpg')}"

                    if neighbour_detection not in streak_lines:
                        streak_lines[neighbour_detection] = StreakLine(Path(camera_folder, neighbour, neighbour_detection_image))
                    
                    current_neighbour_streak: StreakLine = streak_lines[neighbour_detection]

                    if not current_neighbour_streak.is_valid:
                        continue
                    
                    # print(image, detection)

                    if current_streak.same_trajectory(current_neighbour_streak):

                        same_trajectory = True

                        for group in groups:
                            if current_detection in group:
                                group.append(neighbour_detection)
                                break
                            elif neighbour_detection in group:
                                group.append(current_detection)
                                group.sort()
                                break
                        else:
                            groups.append([current_detection, neighbour_detection])

                        break

    return groups
