import os
from pathlib import Path

from detection_pipeline.streak_lines import (find_similar_lines,
                                             get_streak_lines)


def main():
    folder_path = "data/detections_dfn-l0-20151101/dfn-l0-20151101/"
    camera_folders = [i for i in sorted(os.listdir(folder_path)) if (Path(folder_path, i)).is_dir()]

    total = 0

    for camera in camera_folders:
        
        print()
        print(camera)

        camera_folder = Path(folder_path, camera)
        streak_lines = get_streak_lines(camera_folder)
        groups = find_similar_lines(streak_lines)
        
        for group in groups:
            print()
            for i in group:
                print(i)
            total += len(group)
        print()

    print(total)


if __name__ == "__main__":
    main()