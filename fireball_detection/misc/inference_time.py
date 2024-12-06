import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import skimage.io as io
from ultralytics import YOLO

from fireball_detection.detect import detect_fireballs


def process_image(image_path: Path, model: YOLO) -> float:
    image = io.imread(image_path)
    t0 = time.time()
    detect_fireballs(image, model, 5)
    t1 = time.time()
    return t1 - t0


def main():
    @dataclass
    class Args:
        images_folder: str
        model_path: str
        number: int
    
    parser = argparse.ArgumentParser(
        description="Detect fireballs in images and calculate average inference time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--images_folder', type=str, required=True, help='Path to the folder containing images.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file (.pt or .onnx or .engine).')
    parser.add_argument('--number', type=int, required=True, help='The number of detections to average.')

    args = Args(**vars(parser.parse_args()))
    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    try:
        model = YOLO(args.model_path, task="detect")
    except FileNotFoundError as e:
        print(e)
        return None

    images = sorted(list(Path(args.images_folder).glob('*.jpg')))
    if len(images) < args.number:
        print(f"Not enough images in the folder. ({len(images)} < {args.number})")

    inference_times = []
    for image_path in images[:args.number]:
        inference_time = process_image(image_path, model)
        inference_times.append(inference_time)

        print(f"Inference time for {image_path}: {inference_time:.5f} seconds")

    if args.number == 1:
        average_time = inference_times[0]
        average_time_excluding_first = "N/A"
    else:
        average_time = sum(inference_times) / args.number
        average_time_excluding_first = sum(inference_times[1:]) / (args.number - 1)

    print(f"\n{'Time for first':<20}{'Average time all':<20}{'Average time excluding first':<30}")
    print(f"{inference_times[0]:<20}{average_time:<20}{average_time_excluding_first:<30}")


if __name__ == "__main__":
    main()