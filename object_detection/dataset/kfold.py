import multiprocessing as mp
import os
import shutil
import signal
from pathlib import Path
from queue import Empty, Full

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from object_detection.dataset import DATA_FOLDER, DATA_YAML, GFO_JPEGS
from object_detection.dataset.split_tiles import SplitTilesFireball
from object_detection.utils import print_tree

KFOLD_OBJECT_DETECTION_FOLDER = Path(DATA_FOLDER, "kfold_object_detection")
ALL_FOLDER = Path(KFOLD_OBJECT_DETECTION_FOLDER, "all")
ALL_IMAGES_FOLDER = Path(ALL_FOLDER, "images")
ALL_LABELS_FOLDER = Path(ALL_FOLDER, "labels")


SENTINEL = None


def retrieve_fireball_splits() -> tuple[list[str], tuple[int, tuple[np.ndarray, np.ndarray]]]:
    """
    Retrieves image filenames and their corresponding K-Fold splits.

    Returns:
        a tuple containing a list of fireball image files from GFO_JPEGS and
        a enumeration of each split (i, (train indexes, test indexes))
    """
    fireball_images = sorted(os.listdir(GFO_JPEGS))
    kf = KFold(n_splits=5)
    splits = enumerate(kf.split(fireball_images))
    return fireball_images, splits


def generate_tiles(fireball_name: str) -> None:
    fireball = SplitTilesFireball(fireball_name)
    fireball.save_images(ALL_IMAGES_FOLDER)
    fireball.save_labels(ALL_LABELS_FOLDER)


def run_generate_tiles(names_queue: mp.Queue, bar_queue: mp.Queue) -> None:
    try:
        while True:
            fireball_name = names_queue.get()
            if fireball_name is SENTINEL:
                break
            generate_tiles(fireball_name)
            bar_queue.put_nowait(1)
    except (Full, Empty) as e:
        print(type(e))
        return


def update_bar(bar_queue: mp.Queue, total: int) -> None:
    pbar = tqdm(total=total, desc="generating tiles")
    while True:
        bar_queue.get(True)
        pbar.update(1)


def create_folders() -> None:
    """
    The directory structure created will look like this:
    ```
    kfold_object_detection
    ├── fold0
    │   ├── data.yaml
    │   ├── images
    │   │   ├── train
    │   │   └── val
    │   └── labels
    │       ├── train
    │       └── val
    ├── fold1
    │   ├── data.yaml
    ... ...
    ```
    """
    if Path(KFOLD_OBJECT_DETECTION_FOLDER).exists():
        shutil.rmtree(KFOLD_OBJECT_DETECTION_FOLDER)
    os.mkdir(KFOLD_OBJECT_DETECTION_FOLDER)

    os.mkdir(ALL_FOLDER)
    os.mkdir(ALL_IMAGES_FOLDER)
    os.mkdir(ALL_LABELS_FOLDER)

    for i in range(5):
        split_folder = Path(KFOLD_OBJECT_DETECTION_FOLDER, f"split{i}")
        os.mkdir(split_folder)
        shutil.copy(DATA_YAML, split_folder)
        with open(Path(split_folder, "data.yaml"), "r+") as yaml_file:
            content = yaml_file.read()
            content = content.replace("object_detection", f"kfold_object_detection/split{i}")
            yaml_file.seek(0)
            yaml_file.write(content)
            yaml_file.truncate()
        for folder in ("images", "labels"):
            os.mkdir(Path(split_folder, folder))
            for sub_folder in ("train", "val"):
                os.mkdir(Path(split_folder, folder, sub_folder))
    
    print("\n\nCreated folder structure in data folder:\n")
    print_tree(KFOLD_OBJECT_DETECTION_FOLDER)


def create_tiles() -> None:
    fireball_images, _ = retrieve_fireball_splits()

    num_processes = 8

    names_queue = mp.Queue()
    for fireball_image in fireball_images:
        names_queue.put_nowait(fireball_image.split(".")[0])
    
    for _ in range(num_processes):
        names_queue.put(SENTINEL)
    
    print("setting up processes...")

    bar_queue = mp.Queue()
    bar_process = mp.Process(target=update_bar, args=(bar_queue, len(fireball_images)), daemon=True)
    bar_process.start()

    processes: list[mp.Process] = []

    for _ in range(num_processes):
        process = mp.Process(target=run_generate_tiles, args=(names_queue, bar_queue))
        processes.append(process)
        process.start()

    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        names_queue.close()
        bar_queue.close()
        for process in processes:
            process.terminate()
            process.join()
        os.kill(os.getpid(), signal.SIGTERM)


def create_split_dataset(fireball_images: list[str], split: int, name: str, indexes) -> None:
    position = split * 2 + (1 if name == "val" else 0)
    for i in tqdm(indexes, position=position, desc=f"split {split} {name}"):
        fireball_name = fireball_images[i].split(".")[0]
        fireball_tile_images = [file for file in os.listdir(ALL_IMAGES_FOLDER) if fireball_name in file]
        for tile_file in fireball_tile_images:
            shutil.copy(
                Path(ALL_IMAGES_FOLDER, tile_file),
                Path(KFOLD_OBJECT_DETECTION_FOLDER, f"split{split}", "images", name)
            )
            shutil.copy(
                Path(ALL_LABELS_FOLDER, tile_file.replace("jpg", "txt")),
                Path(KFOLD_OBJECT_DETECTION_FOLDER, f"split{split}", "labels", name)
            )


def create_splits() -> None:
    fireball_images = sorted(os.listdir(GFO_JPEGS))
    kf = KFold(n_splits=5)

    for i, (train_indexes, val_indexes) in enumerate(kf.split(fireball_images)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_indexes} length={len(train_indexes)}")
        print(f"  Test:  index={val_indexes} length={len(val_indexes)}")
    
    print("\n\nCreating split datasets...\n")
    procs: list[mp.Process] = []
    for i, (train_indexes, val_indexes) in enumerate(kf.split(fireball_images)):
        train_proc = mp.Process(target=create_split_dataset, args=(fireball_images, i, "train", train_indexes))
        procs.append(train_proc)
        train_proc.start()
        val_proc = mp.Process(target=create_split_dataset, args=(fireball_images, i, "val", val_indexes))
        procs.append(val_proc)
        val_proc.start()

    for proc in procs:
        proc.join()


if __name__ == "__main__":
    create_splits()