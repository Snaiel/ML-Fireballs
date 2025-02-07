from pathlib import Path

DATA_FOLDER = Path(Path(__file__).parents[1], "data")

GFO_DATASET_FOLDER = Path(DATA_FOLDER, "GFO_fireball_object_detection_training_set")
GFO_JPEGS = Path(GFO_DATASET_FOLDER, "jpegs")
GFO_PICKINGS = Path(GFO_DATASET_FOLDER, "point_pickings_csvs")

GFO_FIXES_FOLDER = Path(DATA_FOLDER, "gfo_fixes")

GFO_THUMB_EXT = ".thumb.jpg"

DATASET_FOLDER = Path(DATA_FOLDER, "object_detection")
DATA_YAML = Path("object_detection", "cfg", "data.yaml")