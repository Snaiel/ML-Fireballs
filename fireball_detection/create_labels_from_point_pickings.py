from pathlib import Path
from point_pickings_to_bounding_boxes import get_yolov8_label_from_point_pickings_csv
import os, shutil

file_path = Path(__file__)
parent_folder = file_path.parents[0]

GFO_DATASET_FOLDER = "GFO_fireball_object_detection_training_set"

point_pickings_path = Path(parent_folder, GFO_DATASET_FOLDER, "point_pickings_csvs")

output_folder = Path(parent_folder, "yolov8_labels")

if output_folder.exists():
    shutil.rmtree(output_folder)
os.mkdir(output_folder)

for csv_file in os.listdir(point_pickings_path):
    label = get_yolov8_label_from_point_pickings_csv(Path(point_pickings_path, csv_file))
    label.insert(0, 0)
    with open(Path(output_folder, csv_file.split(".")[0] + ".txt"), 'x') as label_file:
        label_file.write(" ".join(str(item) for item in label))