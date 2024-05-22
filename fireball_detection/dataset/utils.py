import os
import shutil
from pathlib import Path

from dataset import RANDOM_SEED
from dataset.fireball import Fireball
from sklearn.model_selection import train_test_split
from dataset import DATASET_FOLDER, DATA_YAML, GFO_JPEGS


def prepare_dataset(dataset_size: int = 6555) -> dict:
    """
    returns a dictionary containing "train", "val", "test"
    lists of fireballs jpg filenames from the gfo folder.
    """
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
    
    fireball_images = os.listdir(GFO_JPEGS)

    if dataset_size != len(fireball_images):
        fireball_images, _ = train_test_split(fireball_images, train_size=dataset_size, random_state=RANDOM_SEED)

    temp_fireballs, test_fireballs = train_test_split(fireball_images, train_size=0.8, random_state=RANDOM_SEED)
    train_fireballs, val_fireballs = train_test_split(temp_fireballs, train_size=0.8, random_state=RANDOM_SEED)
    # 64% train, 16% val, 20% test

    fireball_dataset = {
        "train": train_fireballs,
        "val": val_fireballs,
        "test": test_fireballs
    }

    return fireball_dataset


def create_dataset(fireball_type: Fireball, dataset_size: int = None):
    fireball_dataset = prepare_dataset(dataset_size)
    for dataset, fireballs in fireball_dataset.items():
        print(f"Creating {dataset} dataset...")
        for fireball_filename in fireballs:
            fireball: Fireball = fireball_type(fireball_filename)
            fireball.save_image(dataset)
            fireball.save_label(dataset)