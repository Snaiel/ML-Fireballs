import os
import re
import shutil
from pathlib import Path

from object_detection.dataset import DATA_FOLDER, DATA_YAML, GFO_JPEGS
from object_detection.dataset.generate import GenerateDatasetArgs, get_args
from object_detection.dataset.generate.utils import create_tiles
import json


def get_2015_removed_dataset() -> tuple[list]:
    fireballs = list(map(lambda x: x.replace(".thumb.jpg", ""), sorted(os.listdir(GFO_JPEGS))))
    pattern = r"_2015-\d{2}-\d{2}_"
    
    train_fireballs = []
    val_fireballs = []

    for fireball in fireballs:
        if re.search(pattern, fireball):
            val_fireballs.append(fireball)
        else:
            train_fireballs.append(fireball)

    return fireballs, train_fireballs, val_fireballs


def generate_dataset_2015_removed(args: GenerateDatasetArgs) -> None:

    fireballs, train_fireballs, val_fireballs = get_2015_removed_dataset()

    print("{:<16} {:<30} {:<30}".format("All Fireballs", "2015 Removed (For Training)", "2015 Fireballs (For Validation)"))
    print("{:<16} {:<30} {:<30}".format(len(fireballs), len(train_fireballs), len(val_fireballs)))

    object_detection_folder_name = f"object_detection_1_to_{args.negative_ratio}_2015_removed"
    object_detection_folder = Path(DATA_FOLDER, object_detection_folder_name)

    if Path(object_detection_folder).exists():
        if args.overwrite:
            print("\nremoving existing folder...")
            shutil.rmtree(object_detection_folder)
        else:
            print(f"\"{object_detection_folder}\" already exists. include --overwrite option to overwrite folders.")
            return
    
    os.mkdir(object_detection_folder)
    for folder in ("images", "labels"):
        os.mkdir(Path(object_detection_folder, folder))
        for sub_folder in ("train", "val"):
            os.mkdir(Path(object_detection_folder, folder, sub_folder))

    shutil.copy(DATA_YAML, object_detection_folder)
    with open(Path(object_detection_folder, "data.yaml"), "r+") as yaml_file:
        content = yaml_file.read()
        content = content.replace("object_detection", f"{object_detection_folder_name}")
        yaml_file.seek(0)
        yaml_file.write(content)
        yaml_file.truncate()

    print("\nTrain Set (2015 Fireballs Removed)")
    create_tiles(
        args.num_processes,
        args.negative_ratio,
        train_fireballs,
        Path(object_detection_folder, "images", "train"),
        Path(object_detection_folder, "labels", "train")
    )

    print("\n\nValidation Set (2015 Fireballs)")
    create_tiles(
        args.num_processes,
        args.negative_ratio,
        val_fireballs,
        Path(object_detection_folder, "images", "val"),
        Path(object_detection_folder, "labels", "val")
    )


def main() -> None:
    args = get_args()
    print("\nargs:", json.dumps(vars(args), indent=4), "\n")
    generate_dataset_2015_removed(args)


if __name__ == "__main__":
    main()