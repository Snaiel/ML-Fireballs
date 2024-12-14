import argparse
import json
import os
import random
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from object_detection.dataset import DATA_FOLDER, DATA_YAML


@dataclass
class Args:
    all_folder_path: str
    negative_ratio: int
    overwrite: bool


def get_2015_halved_dataset(all_folder_path: str) -> tuple[list]:
    fireballs = list(
        map(
            lambda x: x.replace(".jpg", ""),
            sorted(os.listdir(Path(all_folder_path, "images")))
        )
    )
    pattern_train = r"_2015-(07|08|09|10|11|12)-\d{2}_"
    
    train_fireballs = []
    val_fireballs = []

    for fireball in fireballs:
        if re.search(pattern_train, fireball):
            train_fireballs.append(fireball)
        else:
            val_fireballs.append(fireball)

    return fireballs, train_fireballs, val_fireballs


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
    folder_name = "halved" if args.negative_ratio == -1 else f"halved_1_to_{args.negative_ratio}"
    object_detection_folder = Path(DATA_FOLDER, "object_detection", "2015_differenced", folder_name)

    if Path(object_detection_folder).exists():
        if args.overwrite:
            print("removing existing folder...")
            shutil.rmtree(object_detection_folder)
        else:
            print(f"\"{object_detection_folder}\" already exists. include --overwrite option to overwrite folders.")
            return
    else:
        os.makedirs(object_detection_folder, exist_ok=True)
        
    for folder in ("images", "labels"):
        for sub_folder in ("train", "val"):
            os.makedirs(Path(object_detection_folder, folder, sub_folder), exist_ok=True)

    shutil.copy(DATA_YAML, object_detection_folder)
    with open(Path(object_detection_folder, "data.yaml"), "r+") as yaml_file:
        content = yaml_file.read()
        content = content.replace("object_detection", f"{str(object_detection_folder.relative_to(DATA_FOLDER))}")
        yaml_file.seek(0)
        yaml_file.write(content)
        yaml_file.truncate()


    fireballs, train_fireballs, val_fireballs = get_2015_halved_dataset(args.all_folder_path)

    print()
    print("{:<17} {:<25} {}".format("All Fireballs", "Jan-June (Validation)", "Jul-Dec (Training)"))
    print("{:<17} {:<25} {}".format(len(fireballs), len(val_fireballs), len(train_fireballs)))
    print()

    def filter_and_sample_fireballs(fireballs, negative_ratio):
        fireball_tiles = []
        negative_tiles = []

        for t in fireballs:
            if "negative" in t:
                negative_tiles.append(t)
            else:
                fireball_tiles.append(t)
        
        if negative_ratio == -1:
            return fireball_tiles, negative_tiles
        
        if len(fireball_tiles) * negative_ratio > len(negative_tiles):
            sample_size = len(negative_tiles)
        else:
            sample_size = len(fireball_tiles) * negative_ratio
        negative_tiles = random.sample(negative_tiles, sample_size)

        return fireball_tiles, negative_tiles

    def copy_file(tile_file, dest_folder):
        image_filename = tile_file + ".jpg"
        shutil.copy(
            Path(all_folder, "images", image_filename),
            Path(object_detection_folder, "images", dest_folder, image_filename)
        )
        label_filename = tile_file + ".txt"
        shutil.copy(
            Path(all_folder, "labels", label_filename),
            Path(object_detection_folder, "labels", dest_folder, label_filename)
        )

    def copy_files(tile_files, dest_folder):
        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(lambda tf: copy_file(tf, dest_folder), tile_files), total=len(tile_files)))


    train_fireball_tiles, train_negative_tiles = filter_and_sample_fireballs(train_fireballs, args.negative_ratio)
    val_fireball_tiles, val_negative_tiles = filter_and_sample_fireballs(val_fireballs, args.negative_ratio)
    copy_files(train_fireball_tiles + train_negative_tiles, "train")
    copy_files(val_fireball_tiles + val_negative_tiles, "val")

    print()

    train_fireball_total = len(train_fireball_tiles)
    train_negative_total = len(train_negative_tiles)
    val_fireball_total = len(val_fireball_tiles)
    val_negative_total = len(val_negative_tiles)
    
    print(f"Total fireball tiles in train: {train_fireball_total}")
    print(f"Total negative tiles in train: {train_negative_total}")
    print(f"Total fireball tiles in validation: {val_fireball_total}")
    print(f"Total negative tiles in validation: {val_negative_total}")
    print()

    train_ratio = train_negative_total / train_fireball_total
    print(f"Train tiles ratio: 1 to {train_ratio:.2f}")
    
    val_ratio = val_negative_total / val_fireball_total
    print(f"Validation tiles ratio: 1 to {val_ratio:.2f}")


if __name__ == "__main__":
    main()