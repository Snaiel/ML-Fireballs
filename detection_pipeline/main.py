import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import structlog
from skimage import io
from tqdm import tqdm

from detection_pipeline import MAX_TIME_DIFFERENCE
from detection_pipeline.image_differencing import difference_images
from fireball_detection.detect import detect_fireballs
from object_detection.detectors import DetectorSingleton

from utils.logging import get_logger


logger = get_logger()


@dataclass
class Args:
    folder_path: str
    model_path: str
    processes: int
    detector: str


@dataclass
class ProcessTripleArgs:
    folder_path: Path
    output_folder: Path
    model_path: Path
    detector: str
    before: str | None
    current: str
    after: str | None


def get_time_seconds(fireball_label: str) -> int:
    pattern = r"\d{4}-\d{2}-\d{2}_\d{6}"
    datetime_str = re.search(pattern, fireball_label).group(0)
    dt = datetime.strptime(datetime_str, "%Y-%m-%d_%H%M%S")
    time_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
    return time_seconds


def process_triple(args: ProcessTripleArgs) -> None:

    detector = DetectorSingleton.get_detector(args.detector, args.model_path)

    structlog.contextvars.bind_contextvars(image=args.current)
    
    image_current = io.imread(Path(args.folder_path, args.current))

    if args.before and args.after:
        image_before = io.imread(Path(args.folder_path, args.before))
        image_after = io.imread(Path(args.folder_path, args.after))

        differenced_image_pair1 = difference_images(image_current, image_before)
        differenced_image_pair2 = difference_images(image_current, image_after)

        brightness_image_pair1 = np.mean(differenced_image_pair1)
        brightness_image_pair2 = np.mean(differenced_image_pair2)

        differenced_image = differenced_image_pair1 if brightness_image_pair1 < brightness_image_pair2 else differenced_image_pair2

    else:
        differenced_image = difference_images(
            image_current,
            io.imread(Path(args.folder_path, args.before)) if args.before else io.imread(Path(args.folder_path, args.after))
        )
    
    fireball_name = args.current.split(".")[0]
    fireballs = detect_fireballs(differenced_image, detector)

    detections = [vars(fireball) for fireball in fireballs]
    logger.info("detected_fireballs", detected_fireballs=detections)

    if fireballs:
        fireball_folder = Path(args.output_folder, fireball_name)
        os.mkdir(fireball_folder)

        shutil.copy(Path(args.folder_path, args.current), fireball_folder)

        for f in fireballs:
            coords = list(map(int, f.box))
            x1, y1, x2, y2 = coords
            tile_name = f"{fireball_name}_{int(f.conf * 100)}_{'-'.join(map(str, coords))}"
            io.imsave(
                Path(fireball_folder, tile_name + ".jpg"), image_current[y1:y2, x1:x2],
                check_contrast=False,
                quality=100                                                 
            )
            io.imsave(
                Path(fireball_folder, tile_name + "_differenced.jpg"), differenced_image[y1:y2, x1:x2],
                check_contrast=False,
                quality=100
            )

        output_data = {"detections": detections}

        output_json = json.dumps(output_data, indent=4)

        with open(Path(fireball_folder, fireball_name + ".json"), 'w') as json_file:
            json_file.write(output_json)
        
    structlog.contextvars.unbind_contextvars("image")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect fireballs from images in a folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the YOLO model file")
    parser.add_argument('--processes', type=int, default=8, help="Number of processes to use as workers")
    parser.add_argument('--detector', type=str, choices=['Ultralytics', 'ONNX'], default='Ultralytics', help='The type of detector to use.')
    
    args = Args(**vars(parser.parse_args()))
    logger.info("args", args=vars(args))

    folder_path = Path(args.folder_path)
    if not folder_path.exists() or folder_path.is_file():
        print(f"{folder_path} is not a valid folder.")
        return

    output_folder = Path(folder_path, folder_path.name)
    if not output_folder.exists():
        os.mkdir(output_folder)

    model_path = Path(args.model_path)

    images = [i for i in sorted(os.listdir(folder_path)) if i.endswith(".jpg")]

    args_list = []

    for i in range(len(images)):
        
        current: str = images[i]
        time_current: int = get_time_seconds(current)

        before: str | None = None
        after: str | None = None

        if i > 0 and time_current - get_time_seconds(images[i-1]) <= MAX_TIME_DIFFERENCE:
            before = images[i-1]

        if i < len(images) - 1 and get_time_seconds(images[i+1]) - time_current <= MAX_TIME_DIFFERENCE:
            after = images[i+1]

        if not (before or after):
            continue
        
        args_list.append(
            ProcessTripleArgs(
                folder_path,
                output_folder,
                model_path,
                args.detector,
                before,
                current,
                after
            )
        )
    
    logger.info(
        "starting detections using multiprocessing, the following logs will be ordered by image sequence, timestamps likely to be in different order"
    )

    with Pool(args.processes) as pool:
        # list(tqdm(pool.imap(process_triple, args_list), total=len(args_list)))
        list(pool.imap(process_triple, args_list))
        

if __name__ == "__main__":
    main()