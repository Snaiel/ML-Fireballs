import argparse
import gc
import json
import os
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import io
from tqdm import tqdm

from fireball_detection.detect import detect_fireballs, plot_boxes
from fireball_detection.val import VAL_FIREBALL_DETECTION_FOLDER
from object_detection.detectors import DetectorSingleton
from utils.paths import DATA_FOLDER


@dataclass
class Args:
    val_folder_name: str
    model_path: str
    detector: str
    processes: int
    border_size: int
    num_samples: int | float


@dataclass
class TestFireballArgs:
    val_folder: Path
    model_path: Path
    detector: str
    fireball_file: str
    detected_boxes: list[str]
    preds: list[str]
    border_size: int


def test_fireball(args: TestFireballArgs) -> None:
    detector = DetectorSingleton.get_detector(args.detector, args.model_path)

    fireball_name = args.fireball_file.split(".")[0]
        
    if fireball_name + ".txt" in args.detected_boxes and fireball_name + ".jpg" in args.preds:
        return

    image = io.imread(Path(args.val_folder, "images", args.fireball_file))
    
    fireballs = detect_fireballs(image, detector, args.border_size)
    
    with open(Path(args.val_folder, "boxes", fireball_name + ".txt"), "w") as boxes_file:
        lines = []
        for fireball in fireballs:
            lines.append(repr(fireball))
        boxes_file.write("\n".join(lines))

    fig, ax = plot_boxes(image, fireballs)

    pp_bb_path = Path(args.val_folder, "pp_bb", fireball_name + ".txt")
    
    if pp_bb_path.exists():
        with open(pp_bb_path) as file:
            pp_bb = [float(x) for x in file.readline().split(" ")]

        ax.add_patch(
            Rectangle(
                (pp_bb[0], pp_bb[1]),
                pp_bb[2] - pp_bb[0],
                pp_bb[3] - pp_bb[1],
                linewidth=1,
                edgecolor='lime',
                facecolor='none'
            )
        )

    fig.savefig(Path(args.val_folder, "preds", fireball_name + ".jpg"), bbox_inches='tight', pad_inches=0, dpi=600)
    
    plt.cla()
    plt.clf()
    plt.close('all')
    del fig, ax, image, fireballs
    gc.collect()


def test(args: Args) -> None:
    matplotlib.use("agg")

    if not Path(VAL_FIREBALL_DETECTION_FOLDER).exists():
        print(f"{VAL_FIREBALL_DETECTION_FOLDER.relative_to(DATA_FOLDER.parent)} does not exist. Run\n")
        print(f"python3 -m fireball_detection.val.val_full_images create")
        return

    print(f"testing {args.val_folder_name}...\n")

    val_folder = Path(VAL_FIREBALL_DETECTION_FOLDER, args.val_folder_name)

    full_fireball_files = os.listdir(Path(val_folder, "images"))

    num_samples = args.num_samples
    if isinstance(num_samples, float):
        num_samples = max(1, int(len(full_fireball_files) * num_samples))

    num_samples = min(num_samples, len(full_fireball_files))
    fireball_files = full_fireball_files[:num_samples]

    detected_boxes = os.listdir(Path(val_folder, "boxes"))
    preds = os.listdir(Path(val_folder, "preds"))

    args_list = [TestFireballArgs(
            val_folder,
            args.model_path,
            args.detector,
            fireball_file,
            detected_boxes,
            preds,
            args.border_size
        ) for fireball_file in fireball_files]
    
    with Pool(args.processes) as pool:
        list(tqdm(pool.imap(test_fireball, args_list), total=len(args_list)))


def _parse_num_samples(value):
    try:
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError(f"'{value}' must be non-negative.")
        return ivalue
    except ValueError:
        try:
            fvalue = float(value)
            if not (0.0 < fvalue <= 1.0):
                raise argparse.ArgumentTypeError(f"'{value}' must be a positive float between 0 and 1 if it is a proportion.")
            return fvalue
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer or float value.")


def main() -> None:

    parser = argparse.ArgumentParser(
        description="A script to test detections on full images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--val_folder_name', type=str, choices=os.listdir(VAL_FIREBALL_DETECTION_FOLDER), required=True, help="Specify folder to run on.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the YOLO model (.pt .onnx .engine) file')
    parser.add_argument('--detector', type=str, choices=['Ultralytics', 'ONNX'], default='Ultralytics', help='The type of detector to use.')
    parser.add_argument('--processes', type=int, default=8, help="Number of processes to use as workers")
    parser.add_argument('--border_size', type=int, default=5, help="Specify the border size")
    parser.add_argument('--num_samples', type=_parse_num_samples, default=1.0, help="Specify the number of samples to verify. Integer for number, float (0-1] for proportion.")

    args = Args(**vars(parser.parse_args()))

    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    test(args)


if __name__ == "__main__":
    main()