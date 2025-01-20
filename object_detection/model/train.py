import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from ultralytics import YOLO, settings

from utils.constants import DATA_FOLDER


settings.update({"wandb": False})


def main():
    @dataclass
    class Args:
        data_yaml_path: str
        yolo_model: str
        batch_size: float

    parser = argparse.ArgumentParser(description='Train a YOLO model on a specified dataset.')
    parser.add_argument('--data_yaml_path', type=str, required=True, help='Path to the data.yaml file')
    parser.add_argument('--yolo_model', type=str, required=True, help='YOLO model to use e.g. yolov8n.pt')
    parser.add_argument('--batch_size', type=float, default=0.8, help='How many samples to consider during a pass')

    args = Args(**vars(parser.parse_args()))

    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    data = Path(args.data_yaml_path).resolve()

    try:
        model = YOLO(args.yolo_model)
    except FileNotFoundError:
        print("Invalid input. Enter a YOLO model e.g. yolov8n.pt yolo11x.pt")
        return

    kwargs = {}
    with open(Path(Path(__file__).parents[1], "cfg", "split_tiles.yaml"), 'r') as file:
        kwargs = yaml.safe_load(file)

    run_name = str(data.parent.relative_to(DATA_FOLDER)).replace("object_detection_", "").replace("/", "-") + "-" + args.yolo_model

    model.train(
        data=data,
        epochs=100,
        imgsz=416,
        pretrained=True,
        batch=int(args.batch_size) if args.batch_size > 1 else args.batch_size,
        cache=False,
        name=run_name,
        val=False if "/all/" in str(data) else True,
        **kwargs
    )


if __name__ == "__main__":
    main()