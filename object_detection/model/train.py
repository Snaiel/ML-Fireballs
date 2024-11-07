import os
from pathlib import Path

import yaml
from ultralytics import YOLO, settings

from object_detection.dataset import DATA_FOLDER

settings.update({"wandb": False})


def main():
    data_files: list[Path] = []
    for od_folder in [i for i in os.listdir(DATA_FOLDER) if i.startswith("object_detection")]:
        for folder in os.listdir(Path(DATA_FOLDER, od_folder)):
            data_yaml_file = Path(DATA_FOLDER, od_folder, folder, "data.yaml")
            if data_yaml_file.exists():
                data_files.append(data_yaml_file)

    if len(data_files) == 0:
        print("No datasets found. Consider running")
        print("  python3 -m object_detection.dataset.generate_dataset")
        print("  python3 -m object_detection.dataset.create_kfold_dataset")
        return

    data_files = sorted(data_files)

    print("Datasets found:")

    for i in range(len(data_files)):
        print(f"  [{i+1}] {data_files[i].relative_to(DATA_FOLDER)}")

    while True:
        try:
            user_input = int(input("Enter which dataset to use: "))
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    data = data_files[user_input-1]

    model = YOLO("yolov8n.pt")

    kwargs = {}
    with open(Path(Path(__file__).parents[1], "cfg", "split_tiles.yaml"), 'r') as file:
        kwargs = yaml.safe_load(file)

    model.train(
        data=data,
        epochs=100,
        imgsz=416,
        pretrained=True,
        val=False if "/all/" in str(data) else True,
        **kwargs
    )


if __name__ == "__main__":
    main()