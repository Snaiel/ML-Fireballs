from pathlib import Path

RANDOM_SEED = 2024
IMAGE_DIM = (7360, 4912)
BB_PADDING = 0.05


DATA_FOLDER = Path(Path(__file__).parents[2], "data")

# gfo dataset folder containing jpegs and point picking csvs
GFO_DATASET_FOLDER = Path(DATA_FOLDER, "GFO_fireball_object_detection_training_set")
GFO_JPEGS = Path(GFO_DATASET_FOLDER, "jpegs")
GFO_PICKINGS = Path(GFO_DATASET_FOLDER, "point_pickings_csvs")

GFO_FIXES_FOLDER = Path(DATA_FOLDER, "gfo_fixes")

GFO_THUMB_EXT = ".thumb.jpg"


DATASET_FOLDER = Path(DATA_FOLDER, "fireball_object_detection")
DATA_YAML = Path(Path(__file__).parents[1], "cfg", "data.yaml")