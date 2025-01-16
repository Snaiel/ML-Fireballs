import os
from pathlib import Path

from detection_pipeline.streak_lines import find_slow_objects


def main():

    folder_path = "data/detections_dfn-l0-20151101/dfn-l0-20151101/"
    camera_folders = [i for i in sorted(os.listdir(folder_path)) if (Path(folder_path, i)).is_dir()]

    total = 0

    for camera in camera_folders:

        camera_folder = Path(folder_path, camera)
        groups = find_slow_objects(camera_folder)
        
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