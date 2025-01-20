import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from object_detection.dataset import DATA_FOLDER, DATA_YAML, GFO_THUMB_EXT
from object_detection.dataset.differenced.differenced_tiles import \
    DifferencedTiles


@dataclass
class Args:
    differenced_images_folder: str
    original_images_folder: str
    overwrite: bool


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Generate dataset for object detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--differenced_images_folder', type=str, required=True, help='Folder containing differenced images.')
    parser.add_argument('--original_images_folder', type=str, required=True, help='Folder containing original images.')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite the output directory if it exists.')

    args = Args(**vars(parser.parse_args()))
    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    object_detection_folder = Path(DATA_FOLDER, "object_detection", "2015_differenced")
    all_folder = Path(object_detection_folder, "all")
    all_images_folder = Path(all_folder, "images")
    all_labels_folder = Path(all_folder, "labels")

    if Path(object_detection_folder).exists():
        if args.overwrite:
            print("\nremoving existing folder...")
            shutil.rmtree(object_detection_folder)
        else:
            print(f"\"{object_detection_folder}\" already exists. include --overwrite option to overwrite folders.")
            return
    
    os.mkdir(object_detection_folder)
    os.mkdir(all_folder)
    os.mkdir(all_images_folder)
    os.mkdir(all_labels_folder)

    shutil.copy(DATA_YAML, all_folder)
    with open(Path(all_folder, "data.yaml"), "r+") as yaml_file:
        content = yaml_file.read()
        content = content.replace("object_detection", f"{str(object_detection_folder.relative_to(DATA_FOLDER))}")
        content = content.replace("images/train", "images")
        content = content.replace("images/val", "images")
        yaml_file.seek(0)
        yaml_file.write(content)
        yaml_file.truncate()

    fireballs = list(map(lambda x: x.replace(".thumb.jpg", ""), sorted(os.listdir(args.differenced_images_folder))))

    with open(Path(all_folder, "fireballs.txt"), "w") as fireballs_file:
        fireballs_file.write(
            "\n".join(fireballs)
        )

    def process_fireball(f):
        tiles = DifferencedTiles(
            Path(args.differenced_images_folder, f + GFO_THUMB_EXT),
            Path(args.original_images_folder, f + GFO_THUMB_EXT)
        )
        tiles.save_tiles(all_images_folder, all_labels_folder)
        return len(tiles.fireball_tiles), len(tiles.negative_tiles)

    print()

    with ThreadPoolExecutor() as executor:
        tile_counts = list(
            tqdm(
                executor.map(lambda f: process_fireball(f), fireballs),
                total=len(fireballs),
                desc="Training Set"
            )
        )
    
    fireball_tiles_total = sum(fireball_count for fireball_count, _ in tile_counts)
    negative_tiles_total = sum(negative_count for _, negative_count in tile_counts)

    print()

    print(f"Total fireball tiles: {fireball_tiles_total}")
    print(f"Total negative tiles: {negative_tiles_total}")


if __name__ == "__main__":
    main()