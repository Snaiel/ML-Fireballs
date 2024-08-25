import os
import shutil

from sklearn.model_selection import train_test_split

from object_detection.dataset import RANDOM_SEED, DATA_FOLDER
from pathlib import Path

ORIGINAL_VAL_IMAGES = Path(DATA_FOLDER, "object_detection/images/val")
ORIGINAL_VAL_LABELS = Path(DATA_FOLDER, "object_detection/labels/val")

fireball_images = [i[:-4] for i in os.listdir(ORIGINAL_VAL_IMAGES)]
train_fireballs, val_fireballs = train_test_split(fireball_images, train_size=0.8, random_state=RANDOM_SEED)

TUNE_DATA_FOLDER = Path(DATA_FOLDER, "object_detection_tune")

if TUNE_DATA_FOLDER.exists():
    shutil.rmtree(TUNE_DATA_FOLDER)

os.mkdir(TUNE_DATA_FOLDER)
folders = ("images", "labels")
sub_folders = ("train", "val")

for folder in folders:
    os.mkdir(Path(TUNE_DATA_FOLDER, folder))
    for sub_folder in sub_folders:
        os.mkdir(Path(TUNE_DATA_FOLDER, folder, sub_folder))

for train_fireball in train_fireballs:
    jpg = f"{train_fireball}.jpg"
    shutil.copyfile(Path(ORIGINAL_VAL_IMAGES, jpg), Path(TUNE_DATA_FOLDER, "images/train/", jpg))
    txt = f"{train_fireball}.txt"
    shutil.copyfile(Path(ORIGINAL_VAL_LABELS, txt), Path(TUNE_DATA_FOLDER, "labels/train/", txt))

for val_fireball in val_fireballs:
    jpg = f"{val_fireball}.jpg"
    shutil.copyfile(Path(ORIGINAL_VAL_IMAGES, jpg), Path(TUNE_DATA_FOLDER, "images/val/", jpg))
    txt = f"{val_fireball}.txt"
    shutil.copyfile(Path(ORIGINAL_VAL_LABELS, txt), Path(TUNE_DATA_FOLDER, "labels/val/", txt))


DATA_TUNE_YAML = Path(Path(__file__).parents[2], "cfg", "data_tune.yaml")
shutil.copyfile(DATA_TUNE_YAML, Path(TUNE_DATA_FOLDER, "data.yaml"))