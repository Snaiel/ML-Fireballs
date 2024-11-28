import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from ultralytics import YOLO, settings

from object_detection.dataset import DATA_FOLDER


settings.update({"wandb": False})


def main():
    @dataclass
    class Args:
        data_yaml_path: str = ""
        yolo_model: str = ""

    parser = argparse.ArgumentParser(description='Train a YOLO model on a specified dataset.')
    parser.add_argument('--data_yaml_path', type=str, help='Path to the data.yaml file')
    parser.add_argument('--yolo_model', type=str, help='YOLO model to use e.g. yolov8n.pt')

    args = Args(**vars(parser.parse_args()))

    print("\nargs:", json.dumps(vars(args), indent=4), "\n")


    data_files: list[Path] = []

    od_folders = [
        i for i in os.listdir(DATA_FOLDER)
        if i.startswith("object_detection") and Path(DATA_FOLDER, i).is_dir()
    ]

    for od_folder in od_folders:
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

    if args.data_yaml_path:
        data = Path(args.data_yaml_path).resolve()
        if data not in data_files:
            print(f"Specified yaml path {args.data_yaml_path} not found in available datasets.")
            return
    else:
        print("Datasets found:")
        for i in range(len(data_files)):
            print(f"  [{i+1}] {data_files[i].relative_to(DATA_FOLDER)}")
        
        while True:
            try:
                user_input = int(input("Enter which dataset to use: "))
                data = data_files[user_input-1]
                break
            except (ValueError, IndexError):
                print("Invalid input. Please enter a valid integer.")

    if args.yolo_model:
        model_user_input = args.yolo_model
        try:
            model = YOLO(model_user_input)
        except FileNotFoundError:
            print("Invalid input. Enter a YOLO model e.g. yolov8n.pt yolo11x.pt")
            return
    else:
        while True:
            try:
                model_user_input = input("Enter which model to train: ")
                model = YOLO(model_user_input)
            except FileNotFoundError:
                print("Invalid input. Enter a YOLO model e.g. yolov8n.pt yolo11x.pt")

    kwargs = {}
    with open(Path(Path(__file__).parents[1], "cfg", "split_tiles.yaml"), 'r') as file:
        kwargs = yaml.safe_load(file)

    run_name = str(data.parent.relative_to(DATA_FOLDER)).replace("object_detection_", "").replace("/", "-") + "-" + model_user_input

    model.train(
        data=data,
        epochs=100,
        imgsz=416,
        pretrained=True,
        batch=0.8,
        cache=False,
        name=run_name,
        val=False if "/all/" in str(data) else True,
        **kwargs
    )


if __name__ == "__main__":
    main()