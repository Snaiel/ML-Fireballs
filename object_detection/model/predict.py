import argparse
import json
from dataclasses import dataclass

from ultralytics import YOLO


def main():
    @dataclass
    class Args:
        yolo_model: str
        image_path: str

    parser = argparse.ArgumentParser(description='Train a YOLO model on a specified dataset.')
    parser.add_argument('--yolo_model', type=str, required=True, help='YOLO model to use e.g. yolov8n.pt')
    parser.add_argument('--image_path', type=str, required=True, help='path to image tile')

    args = Args(**vars(parser.parse_args()))

    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    try:
        model = YOLO(args.yolo_model)
    except FileNotFoundError:
        print("Invalid input. Enter a YOLO model e.g. yolov8n.pt yolo11x.pt")
        return

    results = model.predict(args.image_path)

    for result in results:
        boxes = result.boxes
        print(boxes)


if __name__ == "__main__":
    main()