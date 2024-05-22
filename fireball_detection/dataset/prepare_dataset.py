from pathlib import Path
import os, shutil
from sklearn.model_selection import train_test_split
from dataset import RANDOM_SEED

file_path = Path(__file__)
root_folder = file_path.parents[2]
# gfo dataset folder containing jpegs and point picking csvs
gfo_dataset_folder = Path(root_folder, "data", "GFO_fireball_object_detection_training_set")

gfo_jpegs = Path(gfo_dataset_folder, "jpegs")
gfo_pickings = Path(gfo_dataset_folder, "point_pickings_csvs")

# output folder
dataset_folder = "yolov8_fireball_dataset/"
# YOLOv8 data config file
data_yaml = "data.yaml"

def prepare_dataset(dataset_size: int = 6555) -> dict:
    """
    returns a dictionary containing "train", "val", "test"
    lists of fireballs jpg filenames from the gfo folder.
    """
    # delete output, create new empty output folder
    if Path(dataset_folder).exists():
        shutil.rmtree(dataset_folder)
    os.mkdir(dataset_folder)

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
    shutil.copy(data_yaml, dataset_folder)

    # Create folders
    folders = ("images", "labels")
    sub_folders = ("train", "val", "test")

    for folder in folders:
        os.mkdir(Path(dataset_folder, folder))
        for sub_folder in sub_folders:
            os.mkdir(Path(dataset_folder, folder, sub_folder))
    
    fireball_images = os.listdir(gfo_jpegs)

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