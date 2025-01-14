import os
import warnings
from pathlib import Path

from sklearn.exceptions import UndefinedMetricWarning

from detection_pipeline.streak_lines import StreakLine


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def main():
    folder_path = "data/detections_dfn-l0-20151101/dfn-l0-20151101/"
    camera_folders = [i for i in sorted(os.listdir(folder_path)) if (Path(folder_path, i)).is_dir()]

    total = 0

    for camera in camera_folders:

        camera_path = Path(folder_path, camera)

        subfolders = [i for i in sorted(os.listdir(camera_path)) if (Path(folder_path, camera, i)).is_dir()]

        streak_lines: dict = {}
        groups: list[list] = []

        for i, subfolder in enumerate(subfolders):
            
            number = int(subfolder.split("_")[-1])
            neighbours = [n for n in subfolders[i+1:max(len(subfolders),i+3+1)] if abs(int(n.split("_")[-1]) - number) <= 5]

            if not neighbours:
                continue

            differenced_detections = [
                i for i in sorted(os.listdir(Path(camera_path, subfolder))) if "differenced" in i and not "thumb" in i
            ]

            for image in differenced_detections:
                same_trajectory = False

                if image not in streak_lines:
                    streak_lines[image] = StreakLine(Path(camera_path, subfolder, image))

                current_streak: StreakLine = streak_lines[image]

                if not current_streak.is_valid:
                    continue
            
                for neighbour in neighbours:

                    if same_trajectory:
                        break

                    for detection in [i for i in sorted(os.listdir(Path(camera_path, neighbour))) if "differenced" in i and not "thumb" in i]:

                        if detection not in streak_lines:
                            streak_lines[detection] = StreakLine(Path(camera_path, neighbour, detection))
                        
                        current_neighbour_streak: StreakLine = streak_lines[detection]

                        if not current_neighbour_streak.is_valid:
                            continue
                        
                        # print(image, detection)

                        if current_streak.same_trajectory(current_neighbour_streak):

                            same_trajectory = True

                            for group in groups:
                                if image in group:
                                    group.append(detection)
                                    break
                                if detection in group:
                                    group.append(image)
                                    group.sort()
                                    break
                            else:
                                groups.append([image, detection])

                            break
        
        print()
        print(camera)
        for group in groups:
            print()
            for i in group:
                print(i)
            total += len(group)
        print()

    print(total)


if __name__ == "__main__":
    main()