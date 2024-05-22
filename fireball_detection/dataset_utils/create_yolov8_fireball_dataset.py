from pathlib import Path
import os, shutil
from sklearn.model_selection import train_test_split
from point_pickings_to_bounding_boxes import get_yolov8_label_from_point_pickings_csv

# gfo dataset folder containing jpegs and point picking csvs
gfo_dataset_folder = "GFO_fireball_object_detection_training_set/"
# output folder
dataset_folder = "yolov8_fireball_dataset/"
# YOLOv8 data config file
data_yaml = "data.yaml"

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

RANDOM_SEED = 2024
DATASET_SIZE = 6555

gfo_jpegs = Path(gfo_dataset_folder, "jpegs")
gfo_pickings = Path(gfo_dataset_folder, "point_pickings_csvs")

fireball_images = os.listdir(gfo_jpegs)

if DATASET_SIZE != len(fireball_images):
    fireball_images, _ = train_test_split(fireball_images, train_size=DATASET_SIZE, random_state=RANDOM_SEED)

temp_fireballs, test_fireballs = train_test_split(fireball_images, train_size=0.8, random_state=RANDOM_SEED)
train_fireballs, val_fireballs = train_test_split(temp_fireballs, train_size=0.8, random_state=RANDOM_SEED)
# 64% train, 16% val, 20% test

fireball_datasets = {
    "train": train_fireballs,
    "val": val_fireballs,
    "test": test_fireballs
}

print(len(train_fireballs), len(val_fireballs), len(test_fireballs))

for dataset, fireballs in fireball_datasets.items():
    for fireball in fireballs:
        fireball_name = fireball.split(".")[0]
        
        shutil.copy(
            Path(gfo_jpegs, fireball),
            Path(dataset_folder, "images", dataset, fireball_name + ".jpg")
        )
        
        label = get_yolov8_label_from_point_pickings_csv(Path(gfo_pickings, fireball_name + ".csv"))
        label.insert(0, 0)

        with open(Path(dataset_folder, "labels", dataset, fireball_name + ".txt"), 'x') as label_file:
            label_file.write(" ".join(str(item) for item in label))