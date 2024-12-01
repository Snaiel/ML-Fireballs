import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import skimage.io as io
from ultralytics import YOLO

from fireball_detection.detect import detect_fireballs


def load_model(yolo_pt_path: str) -> YOLO:
    try:
        return YOLO(yolo_pt_path)
    except FileNotFoundError as e:
        print(e)
        return None


def process_image(image_path: Path, model: YOLO) -> float:
    image = io.imread(image_path)
    t0 = time.time()
    detect_fireballs(image, model, 5)
    t1 = time.time()
    return t1 - t0


def main():
    parser = argparse.ArgumentParser(
        description="Detect fireballs in images and calculate average inference time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--images_folder', type=str, required=True, help='Path to the folder containing images.')
    parser.add_argument('--yolo_pt_path', type=str, required=True, help='Path to the YOLO model file (YOLO .pt file).')
    parser.add_argument('--number', type=int, required=True, help='The number of detections to average.')

    @dataclass
    class Args:
        images_folder: str
        yolo_pt_path: str
        number: int

    args = Args(**vars(parser.parse_args()))
    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    try:
        model = YOLO(args.yolo_pt_path)
    except FileNotFoundError as e:
        print(e)
        return None

    images = sorted(list(Path(args.images_folder).glob('*.jpg')))
    if len(images) < args.number:
        print(f"Not enough images in the folder. ({len(images)} < {args.number})")

    total_time = 0
    for image_path in images[:args.number]:
        inference_time = process_image(image_path, model)
        total_time += inference_time

        print(f"Inference time for {image_path}: {inference_time:.5f} seconds")

    average_time = total_time / args.number
    print(f"\nAverage inference time for {args.number} images: {average_time:.5f} seconds")


if __name__ == "__main__":
    main()