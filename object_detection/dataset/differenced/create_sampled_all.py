import argparse
import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from utils.constants import DATA_FOLDER, DATA_YAML, RANDOM_SEED


@dataclass
class Args:
    all_folder_path: str
    negative_ratio: int
    overwrite: bool


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Generate dataset for object detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--all_folder_path', type=str, required=True, 
                        help='Folder containing all tiles.')
    parser.add_argument('--negative_ratio', type=int, default=-1, 
                        help='Ratio of negative examples to positive examples.')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite the output directory if it exists.')

    args = Args(**vars(parser.parse_args()))
    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    all_folder = Path(args.all_folder_path)
    folder_name = f"all_1_to_{args.negative_ratio}"
    object_detection_folder = Path(DATA_FOLDER, "object_detection", "2015_differenced", folder_name)

    if Path(object_detection_folder).exists():
        if args.overwrite:
            print("removing existing folder...\n")
            shutil.rmtree(object_detection_folder)
        else:
            print(f"\"{object_detection_folder}\" already exists. include --overwrite option to overwrite folders.")
            return
    else:
        os.makedirs(object_detection_folder, exist_ok=True)
        
    for folder in ("images", "labels"):
        os.makedirs(Path(object_detection_folder, folder), exist_ok=True)

    shutil.copy(DATA_YAML, object_detection_folder)
    with open(Path(object_detection_folder, "data.yaml"), "r+") as yaml_file:
        content = yaml_file.read()
        content = content.replace("object_detection", f"{str(object_detection_folder.relative_to(DATA_FOLDER))}")
        content = content.replace("images/train", "images")
        content = content.replace("images/val", "images")
        yaml_file.seek(0)
        yaml_file.write(content)
        yaml_file.truncate()

    tiles_in_all_folder = list(
        map(
            lambda x: x.replace(".jpg", ""),
            sorted(os.listdir(Path(all_folder, "images")))
        )
    )

    print("Tiles in all folder:", len(tiles_in_all_folder), "\n")

    fireball_tiles = []
    negative_tiles = []

    for t in tiles_in_all_folder:
        if "negative" in t:
            negative_tiles.append(t)
        else:
            fireball_tiles.append(t)
    
    print("Total Fireball tiles:", len(fireball_tiles))
    print("Total Negative tiles:", len(negative_tiles), "\n")

    if len(fireball_tiles) * args.negative_ratio > len(negative_tiles):
        sample_size = len(negative_tiles)
    else:
        sample_size = len(fireball_tiles) * args.negative_ratio
    
    negative_tiles = random.Random(RANDOM_SEED).sample(negative_tiles, sample_size)
    
    print("Retrieved Fireball tiles:", len(fireball_tiles))
    print("Retrieved Negative tiles:", len(negative_tiles), "\n")

    def copy_file(tile_file):
        image_filename = tile_file + ".jpg"
        shutil.copy(
            Path(all_folder, "images", image_filename),
            Path(object_detection_folder, "images", image_filename)
        )
        label_filename = tile_file + ".txt"
        shutil.copy(
            Path(all_folder, "labels", label_filename),
            Path(object_detection_folder, "labels", label_filename)
        )

    tile_files = fireball_tiles + negative_tiles

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda tf: copy_file(tf), tile_files), total=len(tile_files)))


if __name__ == "__main__":
    main()