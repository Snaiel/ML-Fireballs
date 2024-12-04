import os
import shutil
from pathlib import Path

from object_detection.dataset import DATA_FOLDER, DATA_YAML, GFO_JPEGS
from object_detection.dataset.generate import GenerateDatasetArgs, get_args
from object_detection.dataset.generate.utils import create_tiles
import json


def generate_dataset_all(args: GenerateDatasetArgs) -> None:
    object_detection_folder_name = f"object_detection_1_to_{args.negative_ratio}"
    object_detection_folder = Path(DATA_FOLDER, object_detection_folder_name)
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
        content = content.replace("object_detection", f"{object_detection_folder_name}/all")
        content = content.replace("images/train", "images")
        content = content.replace("images/val", "images")
        yaml_file.seek(0)
        yaml_file.write(content)
        yaml_file.truncate()

    fireballs = list(map(lambda x: x.replace(".thumb.jpg", ""), sorted(os.listdir(GFO_JPEGS))))

    with open(Path(all_folder, "fireballs.txt"), "w") as fireballs_file:
        fireballs_file.write(
            "\n".join(fireballs)
        )

    create_tiles(args.num_processes, args.negative_ratio, fireballs, all_images_folder, all_labels_folder)


def main() -> None:
    args = get_args()
    print("\nargs:", json.dumps(vars(args), indent=4), "\n")
    generate_dataset_all(args)


if __name__ == "__main__":
    main()