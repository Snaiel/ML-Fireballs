import os
from pathlib import Path

from detection_pipeline.streak_lines import StreakLine, create_streak_line

from multiprocessing import Pool




def main():
    folder_path = "data/detections_dfn-l0-20151101/dfn-l0-20151101/"
    camera_folders = [i for i in sorted(os.listdir(folder_path)) if (Path(folder_path, i)).is_dir()]

    total = 0

    for camera in camera_folders:
        
        print()
        print(camera)

        camera_folder = Path(folder_path, camera)

        subfolders = [i for i in sorted(os.listdir(camera_folder)) if "log" not in i]

        if not subfolders:
            continue

        detections_images: list[Path] = []

        for subfolder in subfolders:
            detections = [i for i in sorted(os.listdir(Path(camera_folder, subfolder))) if "differenced" in i and "thumb" not in i]
            for detection in detections:
                detections_images.append(Path(camera_folder, subfolder, detection))
        
        with Pool() as pool:
            streak_lines_list = pool.map(create_streak_line, detections_images)

        streak_lines_dict = {
            image: streak_line for image, streak_line in zip(detections_images, streak_lines_list)
        }

        groups: list[list] = []

        for i in range(0, len(streak_lines_list) - 1):
            
            current_image: Path = detections_images[i]
            current_streak = streak_lines_dict[current_image]

            if not current_streak.is_valid:
                continue
            
            for j in range(i + 1, len(streak_lines_list)):
                
                other_image = detections_images[j]
                other_streak = streak_lines_dict[other_image]

                if not other_streak.is_valid:
                    continue
                
                # print(current_image.name)
                # print(other_image.name)
                # print(
                #     current_streak.midpoint_to_midpoint(other_streak),
                #     current_streak.angle_between(other_streak),
                #     current_streak.length,
                #     other_streak.length
                # )
                # print()

                if current_streak.similar_line(other_streak):
                    for group in groups:
                        if current_image in group:
                            if other_image not in group:
                                group.append(other_image)
                            break
                        elif other_image in group:
                            if current_image not in group:
                                group.append(current_image)
                                group.sort()
                            break
                    else:
                        groups.append([current_image, other_image])
        
        for group in groups:
            print()
            for i in group:
                i: Path
                print(i.name)
            total += len(group)
        print()

    print(total)


if __name__ == "__main__":
    main()