from pathlib import Path

RANDOM_SEED = 2024
IMAGE_DIM = (7360, 4912)
BB_PADDING = 0.05

# gfo dataset folder containing jpegs and point picking csvs
GFO_DATASET_FOLDER = Path(Path(__file__).parents[2], "data", "GFO_fireball_object_detection_training_set")

GFO_JPEGS = Path(GFO_DATASET_FOLDER, "jpegs")
GFO_PICKINGS = Path(GFO_DATASET_FOLDER, "point_pickings_csvs")

GFO_THUMB_EXT = ".thumb.jpg"

# output folder
DATASET_FOLDER = Path("yolov8_fireball_dataset")
# YOLOv8 data config file
DATA_YAML = Path("data.yaml")