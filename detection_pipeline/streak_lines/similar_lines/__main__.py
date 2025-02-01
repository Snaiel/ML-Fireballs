import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from detection_pipeline.streak_lines import (find_similar_lines,
                                             get_streak_lines)


def main():
    @dataclass
    class Args:
        folder_path: str
    
    parser = argparse.ArgumentParser(
        description="Plot a .differenced.jpg detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("folder_path", type=str, help="Detection outputs folder which contains folders for each camera")

    args = Args(**vars(parser.parse_args()))
    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    camera_folders = [i for i in sorted(os.listdir(args.folder_path)) if (Path(args.folder_path, i)).is_dir()]

    total = 0

    for camera in camera_folders:
        
        print()
        print(camera)

        camera_folder = Path(args.folder_path, camera)
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