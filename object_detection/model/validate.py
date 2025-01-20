import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO


def main():
    @dataclass
    class Args:
        data_yaml_path: str
        yolo_model: str

    parser = argparse.ArgumentParser(description='Train a YOLO model on a specified dataset.')
    parser.add_argument('--data_yaml_path', type=str, required=True, help='Path to the data.yaml file')
    parser.add_argument('--yolo_model', type=str, required=True, help='YOLO model to use e.g. yolov8n.pt')

    args = Args(**vars(parser.parse_args()))

    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    data = Path(args.data_yaml_path).resolve()

    try:
        model = YOLO(args.yolo_model)
    except FileNotFoundError:
        print("Invalid input. Enter a YOLO model e.g. yolov8n.pt yolo11x.pt")
        return

    model.val(
        imgsz=416,
        data=data
    )


if __name__ == "__main__":
    main()