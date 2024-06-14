import os
import shutil
from pathlib import Path

from dataset import DATA_YAML, DATASET_FOLDER
from dataset.fireball import Fireball
from dataset.utils import get_train_val_test_split


MIN_BB_DIM_SIZE = 20


def create_dataset(fireball_type: Fireball, dataset_size: int = 6555):
    # delete output, create new empty output folder
    if Path(DATASET_FOLDER).exists():
        shutil.rmtree(DATASET_FOLDER)
    os.mkdir(DATASET_FOLDER)

    ## Create folder structure
    # dataset_folder
    #     images
    #         train
    #         val
    #         test
    #     labels
    #         train
    #         val
    #         test
    #     data.yaml

    # Copy the data.yaml file to the dataset folder
    shutil.copy(DATA_YAML, DATASET_FOLDER)

    # Create folders
    folders = ("images", "labels")
    sub_folders = ("train", "val", "test")

    for folder in folders:
        os.mkdir(Path(DATASET_FOLDER, folder))
        for sub_folder in sub_folders:
            os.mkdir(Path(DATASET_FOLDER, folder, sub_folder))
    
    fireball_dataset = get_train_val_test_split(dataset_size)
    for dataset, fireballs in fireball_dataset.items():
        print(f"Creating {dataset} dataset...")
        for fireball_filename in fireballs:
            fireball_name = fireball_filename.split(".")[0]
            fireball: Fireball = fireball_type(fireball_name)
            fireball.save_image(Path(DATASET_FOLDER, "images", dataset))
            fireball.save_label(Path(DATASET_FOLDER, "labels", dataset))