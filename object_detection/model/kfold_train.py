import argparse
from ultralytics import YOLO
import yaml
from pathlib import Path

# Set up argument parser
parser = argparse.ArgumentParser(description='Train YOLOv8 on a specific fold of k-fold data.')
parser.add_argument(
    '--fold', 
    type=int, 
    required=True, 
    help='Specify the fold number to train (e.g., 0, 1, 2, 3, 4).'
)
args = parser.parse_args()

# Load the model
model = YOLO("yolov8n.pt")

# Construct the path using the specified fold number
fold_number = args.fold
data = Path(Path(__file__).parents[2], f"data/kfold_object_detection/fold{fold_number}/data.yaml")

# Load additional training configurations
kwargs = {}
with open(Path(Path(__file__).parents[1], "cfg", "split_tiles.yaml"), 'r') as file:
    kwargs = yaml.safe_load(file)

# Train the model
model.train(
    data=data,
    epochs=100,
    imgsz=400,
    pretrained=True,
    **kwargs
)

# Validate the model
model.val(
    data=data,
    imgsz=400,
    plots=True
)
