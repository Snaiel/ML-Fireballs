import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import structlog
from skimage import io
from structlog.typing import WrappedLogger, EventDict
from tqdm import tqdm

from detection_pipeline import MAX_TIME_DIFFERENCE, MIN_DIAGONAL_LENGTH
from detection_pipeline.image_differencing import difference_images
from fireball_detection.detect import (detect_differenced_tiles,
                                       get_absolute_fireball_boxes,
                                       merge_bboxes)
from object_detection.detectors import Detector, DetectorSingleton
from object_detection.utils import diagonal_length


VERSION = "1.0.0"


def module_processor(logger: WrappedLogger, log_method: str, event_dict: EventDict) -> EventDict:
    path = event_dict.pop("pathname")
    module_path = str(Path(path).relative_to(Path(__file__).parents[1]))
    module = module_path.replace("/", ".").replace(".py", "")
    event_dict["module"] = module
    return event_dict


def reorder_event_dict(logger: WrappedLogger, log_method: str, event_dict: EventDict) -> EventDict:
    reordered_dict = {}
    reordered_dict["timestamp"] = event_dict.pop("timestamp")
    reordered_dict["process_name"] = event_dict.pop("process_name")
    reordered_dict["module"] = event_dict.pop("module")
    reordered_dict["function"] = event_dict.pop("func_name")
    reordered_dict["level"] = event_dict.pop("level")
    reordered_dict["event"] = event_dict.pop("event")

    if "image" in event_dict:
        reordered_dict["image"] = event_dict.pop("image")
    
    for key in sorted(event_dict.keys()):
        reordered_dict[key] = event_dict[key]
    
    return reordered_dict


structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.CallsiteParameterAdder(parameters=[
            structlog.processors.CallsiteParameter.FUNC_NAME,
            structlog.processors.CallsiteParameter.PATHNAME,
            structlog.processors.CallsiteParameter.PROCESS_NAME
        ]),
        module_processor,
        reorder_event_dict,
        structlog.processors.JSONRenderer(indent=4),
    ]
)

logger: structlog.stdlib.BoundLogger = structlog.get_logger()


@dataclass
class Args:
    folder_path: str
    output_path: str | None
    model_path: str
    processes: int
    detector: str


@dataclass
class WorkerProcessArgs:
    triples_queue: mp.Queue
    bar_queue: mp.Queue
    detection_log_queue: mp.Queue
    folder_path: Path
    output_folder: Path
    model_path: Path
    detector: str


@dataclass
class ImagesTriple:
    index: int
    before: str | None
    current: str
    after: str | None


class DetectionWorkerProcess(mp.Process):
    
    args: WorkerProcessArgs
    detector: Detector
    log_messages: list[str]


    def __init__(self, args: WorkerProcessArgs):
        super().__init__()
        self.args = args
        self.log_messages = []


    def run(self):
        self.detector = DetectorSingleton.get_detector(self.args.detector, self.args.model_path)

        processors: list = structlog.get_config()["processors"]
        processors.insert(len(processors) - 1, DetectionLoggingProcessor(self))

        while True:
            triple: ImagesTriple = self.args.triples_queue.get()
            if triple is None:
                break
            self.process_triple(triple)
            self.args.detection_log_queue.put((triple.index, "\n".join(self.log_messages)))
            self.log_messages.clear()
            self.args.bar_queue.put(1)
        
        processors.pop(len(processors) - 1)


    def process_triple(self, images: ImagesTriple) -> None:
        structlog.contextvars.bind_contextvars(image=images.current)
        
        image_current = io.imread(Path(self.args.folder_path, images.current))

        if images.before and images.after:
            image_before = io.imread(Path(self.args.folder_path, images.before))
            image_after = io.imread(Path(self.args.folder_path, images.after))

            differenced_image_pair1 = difference_images(image_current, image_before)
            differenced_image_pair2 = difference_images(image_current, image_after)

            brightness_image_pair1 = np.mean(differenced_image_pair1)
            brightness_image_pair2 = np.mean(differenced_image_pair2)

            differenced_image = differenced_image_pair1 if brightness_image_pair1 < brightness_image_pair2 else differenced_image_pair2

        else:
            differenced_image = difference_images(
                image_current,
                io.imread(Path(self.args.folder_path, images.before)) if images.before else io.imread(Path(self.args.folder_path, images.after))
            )
        
        fireball_name = images.current.split(".")[0]

        detected_tiles = detect_differenced_tiles(differenced_image, self.detector, 5)
        detections = []

        if detected_tiles:
            detected_tiles.sort(key=lambda x: x.position)
            logger.info(
                "tile_detections",
                tile_detections=[i.to_dict() for i in detected_tiles]
            )
    
            fireball_boxes = get_absolute_fireball_boxes(detected_tiles)
            
            detected_fireballs = merge_bboxes(fireball_boxes)
            detected_fireballs = [f for f in detected_fireballs if diagonal_length(f.box) > MIN_DIAGONAL_LENGTH]

            if detected_fireballs:
                fireball_folder = Path(self.args.output_folder, fireball_name)
                os.mkdir(fireball_folder)

                shutil.copy(Path(self.args.folder_path, images.current), fireball_folder)

                io.imsave(
                    Path(fireball_folder, images.current.removesuffix("jpg") + "differenced.jpg"),
                    differenced_image,
                    check_contrast=False,
                    quality=100
                )

                for f in detected_fireballs:
                    coords = list(map(int, f.box))
                    x1, y1, x2, y2 = coords

                    detection_name = f"{fireball_name}_{int(f.conf * 100)}_{'-'.join(map(str, coords))}"

                    detection = {"name": detection_name, **vars(f)}
                    detections.append(detection)

                    io.imsave(
                        Path(fireball_folder, detection_name + ".jpg"),
                        image_current[y1:y2, x1:x2],
                        check_contrast=False,
                        quality=100                                                 
                    )
                    io.imsave(
                        Path(fireball_folder, detection_name + ".differenced.jpg"),
                        differenced_image[y1:y2, x1:x2],
                        check_contrast=False,
                        quality=100
                    )

                output_data = {"detections": detections}
                output_json = json.dumps(output_data, indent=4)

                with open(Path(fireball_folder, fireball_name + ".json"), 'w') as json_file:
                    json_file.write(output_json)
        
        logger.info("detected_fireballs", detected_fireballs=detections)
        structlog.contextvars.unbind_contextvars("image")


class DetectionLoggingProcessor:
    
    def __init__(self, worker_process: DetectionWorkerProcess):
        self.worker_process = worker_process

    
    def __call__(self, logger: WrappedLogger, name: str, event_dict: EventDict):
        self.worker_process.log_messages.append(json.dumps(event_dict, indent=4))
        raise structlog.DropEvent


class DetectionLoggerProcess(mp.Process):
    
    detection_messages: dict
    current_index: int
    max_index: int
    logger: structlog.WriteLogger


    def __init__(self, detection_log_queue: mp.Queue, max_index: int):
        super().__init__()
        self.detection_log_queue = detection_log_queue
        self.detection_messages = dict()
        self.current_index = 0
        self.max_index = max_index
        self.logger = structlog.get_config()["logger_factory"]()
    

    def run(self):
        while self.current_index < self.max_index:
            index, msg = self.detection_log_queue.get(True)

            if index == self.current_index:
                self.logger.msg(msg)
                self.current_index += 1
                while self.current_index in self.detection_messages:
                    self.logger.msg(self.detection_messages.pop(self.current_index))
                    self.current_index += 1
            else:
                self.detection_messages[index] = msg


def update_bar(bar_queue: mp.Queue, total: int) -> None:
    pbar = tqdm(total=total, desc="detecting fireballs")
    while True:
        bar_queue.get(True)
        pbar.update(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect fireballs from images in a folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing images")
    parser.add_argument("--output_path", type=str, required=False, default=None, help="Path to the folder to contain the output folder")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the YOLO model file")
    parser.add_argument('--processes', type=int, default=8, help="Number of processes to use as workers")
    parser.add_argument('--detector', type=str, choices=['Ultralytics', 'ONNX'], default='Ultralytics', help='The type of detector to use.')
    
    args = Args(**vars(parser.parse_args()))
    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    folder_path = Path(args.folder_path)
    if not folder_path.exists() or folder_path.is_file():
        print(f"{folder_path} is not a valid folder.")
        return

    output_folder = Path(
        folder_path if args.output_path is None else args.output_path,
        folder_path.name
    )
    if output_folder.exists():
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    log_file = Path(output_folder, f"{'_'.join(folder_path.parts[-2:])}.log")

    structlog.configure(
        logger_factory=structlog.WriteLoggerFactory(
            file=log_file.open("wt")
        )
    )

    logger.info("version", version=VERSION)
    logger.info("args", args=vars(args))

    model_path = Path(args.model_path)

    images = [i for i in sorted(os.listdir(folder_path)) if i.endswith(".jpg")]

    def get_time_seconds(fireball_label: str) -> int:
        pattern = r"\d{4}-\d{2}-\d{2}_\d{6}"
        datetime_str = re.search(pattern, fireball_label).group(0)
        dt = datetime.strptime(datetime_str, "%Y-%m-%d_%H%M%S")
        time_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        return time_seconds


    triples_queue = mp.Queue()
    index = 0

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
            logger.info("image skipped due to lack of recent before or after images", image=current)
            continue
        
        triples_queue.put(
            ImagesTriple(
                index,
                before,
                current,
                after
            )
        )

        index += 1

    bar_queue = mp.Queue()
    bar_process = mp.Process(target=update_bar, args=(bar_queue, triples_queue.qsize()), daemon=True)
    bar_process.start()

    detection_log_queue = mp.Queue()
    detection_logger = DetectionLoggerProcess(detection_log_queue, triples_queue.qsize())
    detection_logger.start()

    process_args = WorkerProcessArgs(
        triples_queue,
        bar_queue,
        detection_log_queue,
        folder_path,
        output_folder,
        model_path,
        args.detector
    )

    processes = [DetectionWorkerProcess(process_args) for _ in range(args.processes)]

    for _ in processes:
        triples_queue.put(None)

    logger.info(
        "starting detections using multiprocessing, the following logs will be ordered by image sequence, timestamps likely to be in different order"
    )

    for p in processes:
        p.start()
    
    for p in processes:
        p.join()
        

if __name__ == "__main__":
    main()