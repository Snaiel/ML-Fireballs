from pathlib import Path
from object_detection.dataset import DATA_FOLDER


VAL_FIREBALL_DETECTION_FOLDER = Path(DATA_FOLDER, "val_fireball_detection")

discard_fireballs = {
    "24_2015-03-18_140528_DSC_0352" # image requires two bounding boxes
}