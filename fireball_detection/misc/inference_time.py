import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import skimage.io as io

from fireball_detection.detect import detect_fireballs
from object_detection.detectors import Detector, get_detector


def process_image(image_path: Path, detector: Detector) -> float:
    image = io.imread(image_path)
    t0 = time.time()
    detect_fireballs(image, detector, 5)
    t1 = time.time()
    return t1 - t0


def main():
    @dataclass
    class Args:
        images_folder: str
        model_path: str
        number: int
        detector: str
    
    parser = argparse.ArgumentParser(
        description="Detect fireballs in images and calculate average inference time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--images_folder', type=str, required=True, help='Path to the folder containing images.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file (.pt or .onnx or .engine).')
    parser.add_argument('--number', type=int, required=True, help='The number of detections to average.')
    parser.add_argument('--detector', type=str, choices=['Ultralytics', 'ONNX'], default='Ultralytics', help='The type of detector to use.')

    args = Args(**vars(parser.parse_args()))
    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    detector = get_detector(args.detector, args.model_path)
    if not detector: return

    images = sorted(list(Path(args.images_folder).glob('*.jpg')))
    if len(images) < args.number:
        print(f"Not enough images in the folder. ({len(images)} < {args.number})")

    # Perform a warmup inference - this time will be reported separately
    warmup_time = process_image(images[0], detector)
    print(f"Warmup inference time for {images[0]}: {warmup_time:.5f} seconds")

    inference_times = []
    # Perform actual inferences and store their times
    for image_path in images[:args.number]:
        inference_time = process_image(image_path, detector)
        inference_times.append(inference_time)
        print(f"Inference time for {image_path.name}: {inference_time:.5f} seconds")

    average_time_excluding_warmup = sum(inference_times) / args.number

    print(f"\n{'Time for warmup':<20}{'Average time excluding warmup':<30}")
    print(f"{warmup_time:<20.5f}{average_time_excluding_warmup:<30.5f}")


if __name__ == "__main__":
    main()