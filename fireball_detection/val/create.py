import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from fireball_detection.val import (VAL_FIREBALL_DETECTION_FOLDER,
                                    discard_fireballs)
from object_detection.dataset import GFO_JPEGS, GFO_PICKINGS
from object_detection.dataset.create_kfold_dataset import \
    retrieve_fireball_splits
from object_detection.dataset.generate.exclude_2015 import \
    get_2015_removed_dataset
from object_detection.dataset.point_pickings import PointPickings


def create_splits() -> None:
    fireballs, splits = retrieve_fireball_splits()

    for split, (_, test_indexes) in splits:
        split_folder = Path(VAL_FIREBALL_DETECTION_FOLDER, f"split{split}")
        split_fireballs = [fireballs[i] for i in test_indexes]
        create_val(split_folder, split_fireballs, f"split{split} samples")


def create_2015_removed() -> None:
    _, _, fireballs = get_2015_removed_dataset()
    folder_path = Path(VAL_FIREBALL_DETECTION_FOLDER, "2015_removed")
    create_val(folder_path, fireballs, f"2015 removed")


def create_val(source_folder: Path, fireballs: list[str], desc: str) -> None:
    if Path(source_folder).exists():
        shutil.rmtree(source_folder)
    os.mkdir(source_folder)

    for sub_folder in ("images", "pp_bb", "boxes", "preds"):
        os.mkdir(Path(source_folder, sub_folder))

    for fireball_name in tqdm(fireballs, desc=desc):
        if fireball_name in discard_fireballs:
            continue
        
        fireball_image_filename = fireball_name + ".thumb.jpg"
        shutil.copyfile(Path(GFO_JPEGS, fireball_image_filename), Path(source_folder, "images", fireball_image_filename))
        
        pp = PointPickings(Path(GFO_PICKINGS, fireball_name + ".csv"))
        with open(Path(source_folder, "pp_bb", fireball_name + ".txt"), "w") as pp_bb_file:
            pp_bb_file.write(f"{pp.bb_min_x} {pp.bb_min_y} {pp.bb_max_x} {pp.bb_max_y}")


def main() -> None:
    @dataclass
    class Args:
        option: str = None

    parser = argparse.ArgumentParser(
        description="A script to test detections on full images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--option',
        choices=['splits', '2015_removed'],
        default='splits',
        help="Choose between 'splits' or '2015_removed'. Default is 'splits'."
    )

    args = Args(**vars(parser.parse_args()))

    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    if not VAL_FIREBALL_DETECTION_FOLDER.exists():
        os.mkdir(VAL_FIREBALL_DETECTION_FOLDER)

    if args.option == "splits":
        create_splits()
    elif args.option == "2015_removed":
        create_2015_removed()


if __name__ == "__main__":
    main()