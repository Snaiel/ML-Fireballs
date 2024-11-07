import argparse
import multiprocessing as mp
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from object_detection.dataset import DATA_FOLDER, DATA_YAML, GFO_JPEGS


def retrieve_fireball_splits() -> tuple[list[str], tuple[int, tuple[np.ndarray, np.ndarray]]]:
    """
    Retrieves image filenames and their corresponding K-Fold splits.

    Returns:
        a tuple containing a list of fireball image files from GFO_JPEGS and
        a list of each split (i, (train indexes, test indexes))
    """
    fireball_images = sorted(os.listdir(GFO_JPEGS))
    kf = KFold(n_splits=5)
    splits = list(enumerate(kf.split(fireball_images)))
    return fireball_images, splits


def _update_bar(bar_queue: mp.Queue, total: int) -> None:
    """
    Update a tqdm progress bar based on signals from the queue.

    Parameters:
    - bar_queue (mp.Queue): A queue to receive progress signals.
    - total (int): Total number of tasks/items to process.
    """
    pbar = tqdm(total=total, desc="creating kfold dataset")
    while True:
        bar_queue.get(True)
        pbar.update(1)


def _create_split_dataset(
        bar_queue: mp.Queue,
        fireball_images: list[str], 
        split: int,
        name: str,
        indexes,
        object_detection_folder: Path,
        all_images_folder: Path,
        all_labels_folder: Path
    ) -> None:
    """
    Copy images and corresponding label files into structured directories for a 
    specific split of a K-Fold cross-validation.

    Args:
        fireball_images (list[str]): List of fireball image filenames.
        split (int): Index of the current split.
        name (str): The type of dataset split (either 'train' or 'val').
        indexes: List of indices indicating which images to include in this split.
        object_detection_folder (Path): Path to the root directory of the k-fold dataset.
        all_images_folder (Path): Path to the directory containing all images.
        all_labels_folder (Path): Path to the directory containing all labels.
    """
    
    for i in indexes:
        # Get base fireball name without extension.
        fireball_name = fireball_images[i].split(".")[0]
        
        # Get all image files corresponding to this fireball.
        fireball_tile_images = [file for file in os.listdir(all_images_folder) if fireball_name in file]
        
        # Copy the images and their corresponding label files to the split folders.
        for tile_file in fireball_tile_images:
            shutil.copy(
                Path(all_images_folder, tile_file),
                Path(object_detection_folder, f"split{split}", "images", name)
            )
            shutil.copy(
                Path(all_labels_folder, tile_file.replace("jpg", "txt")),
                Path(object_detection_folder, f"split{split}", "labels", name)
            )
        
        bar_queue.put_nowait(1)


def main() -> None:
    """
    Main function to orchestrate the creation of K-Fold cross-validation datasets
    for object detection.
    
    Parses command-line arguments, manages directory setup, and initiates the 
    dataset creation by splitting the data into 'train' and 'val' sets and 
    distributing them across multiple processes.
    """
    @dataclass
    class Args:
        """
        Data class to store command-line arguments.
        """
        negative_ratio: int = 1
        overwrite: bool = False

    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description="Generate dataset for object detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--negative_ratio', type=int, default=1, required=True, 
                        help='Ratio of negative examples to positive examples.')
    parser.add_argument('--overwrite', action='store_true', 
                        help='Overwrite the output directory if it exists.')

    args = Args(**vars(parser.parse_args()))
    print(f"\nArgs: {vars(args)}")

    # Define paths for the root and specific folders for images and labels.
    object_detection_folder_name = f"object_detection_1_to_{args.negative_ratio}"
    object_detection_folder = Path(DATA_FOLDER, object_detection_folder_name)
    all_folder = Path(object_detection_folder, "all")
    all_images_folder = Path(all_folder, "images")
    all_labels_folder = Path(all_folder, "labels")

    # Check if k-fold dataset directory exists; else inform to generate the dataset first.
    if not Path.exists(object_detection_folder):
        print(f"\"{object_detection_folder}\" does not exist. generate dataset first with\n")
        print(f"python3 -m object_detection.dataset.generate_dataset --negative_ratio {args.negative_ratio}")
        return

    # Check for existing split directories and handle them based on 'overwrite' argument.
    existing_split_folders = []
    for i in os.listdir(object_detection_folder):
        if "split" in i:
            existing_split_folders.append(i)

    if len(existing_split_folders) > 0:
        if args.overwrite:
            print("\nremoving existing folders...")
            for i in existing_split_folders:
                shutil.rmtree(Path(object_detection_folder, i))
        else:
            print(f"the following folders already exist:")
            for i in existing_split_folders:
                print(i)
            print("\ninclude --overwrite option to overwrite folders.")
            return

    # Retrieve the images and their respective splits.
    fireball_images, splits = retrieve_fireball_splits()

    # Create directories for each split and copy base YAML config files.
    print("\nsetting up folders...")
    for i in range(len(splits)):
        split_folder = Path(object_detection_folder, f"split{i}")
        os.mkdir(split_folder)
        shutil.copy(DATA_YAML, split_folder)
        
        # Update the YAML content for each split.
        with open(Path(split_folder, "data.yaml"), "r+") as yaml_file:
            content = yaml_file.read()
            content = content.replace("object_detection", f"{object_detection_folder_name}/split{i}")
            yaml_file.seek(0)
            yaml_file.write(content)
            yaml_file.truncate()
        
        # Create subdirectories for images and labels, for 'train' and 'val'.
        for folder in ("images", "labels"):
            os.mkdir(Path(split_folder, folder))
            for sub_folder in ("train", "val"):
                os.mkdir(Path(split_folder, folder, sub_folder))

    # print split details.
    print("\nkfold split details:")
    for i, (train_indexes, val_indexes) in splits:
        print(f"Split {i}:")
        print(f"  Train: index={train_indexes} length={len(train_indexes)}")
        print(f"  Test:  index={val_indexes} length={len(val_indexes)}")

    bar_queue = mp.Queue()
    # Process for updating the progress bar
    bar_process = mp.Process(target=_update_bar, args=(bar_queue, len(fireball_images) * len(splits)), daemon=True)
    bar_process.start()

    print()
    # Begin creation of split datasets using multiprocessing for efficiency.
    procs: list[mp.Process] = []
    for i, (train_indexes, val_indexes) in splits:
        # Create and start a process for training dataset creation for each split.
        train_proc = mp.Process(
            target=_create_split_dataset,
            name=f"proc-split{i}-train",
            args=(
                bar_queue,
                fireball_images,
                i,
                "train",
                train_indexes,
                object_detection_folder,
                all_images_folder,
                all_labels_folder
            )
        )

        procs.append(train_proc)
        train_proc.start()
        
        # Create and start a process for validation dataset creation for each split.
        val_proc = mp.Process(
            target=_create_split_dataset,
            name=f"proc-split{i}-val",
            args=(
                bar_queue,
                fireball_images,
                i,
                "val",
                val_indexes,
                object_detection_folder,
                all_images_folder,
                all_labels_folder
            )
        )

        procs.append(val_proc)
        val_proc.start()

    # Wait for all processes to complete.
    for proc in procs:
        proc.join()
    
    print("kfold dataset created!")


if __name__ == "__main__":
    main()
