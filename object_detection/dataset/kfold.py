import multiprocessing as mp
import os
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from object_detection.dataset import (DATA_FOLDER, DATA_YAML, GFO_JPEGS)
from object_detection.dataset.split_tiles import SplitTilesFireball
from object_detection.utils import print_tree


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


def create_kfolds() -> None:
    """
    Create k-fold splits for object detection datasets and prepare the associated directory structure.

    This function performs the following:
    1. Retrieves fireball images and splits.
    2. Creates k-fold splits using KFold from scikit-learn.
    3. Creates a directory structure for the k-folds.
    4. Generates and saves images and labels for each fold in designated train and validation folders.

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
    fireball_images, splits = retrieve_fireball_splits()

    kf = KFold(n_splits=5)
    
    for i, (train_indexes, val_indexes) in splits:
        print(f"Fold {i}:")
        print(f"  Train: index={train_indexes} length={len(train_indexes)}")
        print(f"  Test:  index={val_indexes} length={len(val_indexes)}")
    
    KFOLD_OBJECT_DETECTION_FOLDER = Path(DATA_FOLDER, "kfold_object_detection")
    if Path(KFOLD_OBJECT_DETECTION_FOLDER).exists():
        shutil.rmtree(KFOLD_OBJECT_DETECTION_FOLDER)
    os.mkdir(KFOLD_OBJECT_DETECTION_FOLDER)

    for i in range(5):
        fold_folder = Path(KFOLD_OBJECT_DETECTION_FOLDER, f"fold{i}")
        os.mkdir(fold_folder)
        shutil.copy(DATA_YAML, fold_folder)
        with open(Path(fold_folder, "data.yaml"), "r+") as yaml_file:
            content = yaml_file.read()
            content = content.replace("object_detection", f"kfold_object_detection/fold{i}")
            yaml_file.seek(0)
            yaml_file.write(content)
            yaml_file.truncate()
        for folder in ("images", "labels"):
            os.mkdir(Path(fold_folder, folder))
            for sub_folder in ("train", "val"):
                os.mkdir(Path(fold_folder, folder, sub_folder))
    
    print("\n\nCreated folder structure in data folder:\n")
    print_tree(KFOLD_OBJECT_DETECTION_FOLDER)

    def create_fold_dataset(fold: int, name: str, indexes) -> None:
        position = fold * 2 + (1 if name == "val" else 0)
        for i in tqdm(indexes, position=position, desc=f"fold {fold} {name}"):
            fireball_name = fireball_images[i].split(".")[0]
            fireball = SplitTilesFireball(fireball_name)
            fireball.save_images(Path(KFOLD_OBJECT_DETECTION_FOLDER, f"fold{fold}", "images", name))
            fireball.save_labels(Path(KFOLD_OBJECT_DETECTION_FOLDER, f"fold{fold}", "labels", name))

    print("\n\nCreating fold datasets...\n")

    procs: list[mp.Process] = []
    for i, (train_indexes, val_indexes) in splits:
        train_proc = mp.Process(target=create_fold_dataset, args=(i, "train", train_indexes))
        procs.append(train_proc)
        train_proc.start()
        val_proc = mp.Process(target=create_fold_dataset, args=(i, "val", val_indexes))
        procs.append(val_proc)
        val_proc.start()
    
    for proc in procs:
        proc.join()


if __name__ == "__main__":
    create_kfolds()