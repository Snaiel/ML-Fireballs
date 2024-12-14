import argparse
import json
import os
import re
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
    folder_path: str
    negative_ratio: int
    overwrite: bool


def get_2015_differenced_dataset(path_2015_differenced_folder: str) -> tuple[list]:
    fireballs = list(map(lambda x: x.replace(".thumb.jpg", ""), sorted(os.listdir(path_2015_differenced_folder))))
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

    parser.add_argument('--folder_path', type=str, required=True, 
                        help='Folder containing source images.')
    parser.add_argument('--negative_ratio', type=int, default=-1, 
                        help='Ratio of negative examples to positive examples.')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite the output directory if it exists.')

    args = Args(**vars(parser.parse_args()))
    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    fireballs, train_fireballs, val_fireballs = get_2015_differenced_dataset(args.folder_path)

    print("{:<16} {:<25} {:<20}".format("All Fireballs", "Jan-June (Validation)", "Jul-Dec (Training)"))
    print("{:<16} {:<25} {:<20}".format(len(fireballs), len(train_fireballs), len(val_fireballs)))

    object_detection_folder = Path(DATA_FOLDER, "object_detection", "2015_differenced", "all")

    if Path(object_detection_folder).exists():
        if args.overwrite:
            print("\nremoving existing folder...")
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

    def process_fireball(i, dataset):
        tiles = DifferencedTiles(Path(args.folder_path, i + GFO_THUMB_EXT))
        tiles.save_tiles(Path(object_detection_folder, "images", dataset), Path(object_detection_folder, "labels", dataset))

    print()

    with ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(lambda i: process_fireball(i, "train"), train_fireballs),
                total=len(train_fireballs),
                desc="Training Set"
            )
        )
        list(
            tqdm(
                executor.map(lambda i: process_fireball(i, "val"), val_fireballs),
                total=len(val_fireballs),
                desc="Validation Set"
            )
        )


if __name__ == "__main__":
    main()