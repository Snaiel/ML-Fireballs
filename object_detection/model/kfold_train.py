from ultralytics import YOLO
import yaml
from pathlib import Path


model = YOLO("yolov8n.pt")

data = Path(Path(__file__).parents[2], "data/kfold_object_detection/fold3/data.yaml")


kwargs = {}

with open(Path(Path(__file__).parents[1], "cfg", "split_tiles.yaml"), 'r') as file:
    kwargs = yaml.safe_load(file)


model.train(
    data=data,
    epochs=100,
    imgsz=400,
    pretrained=True,
    **kwargs
)

model.val(
    data=data,
    imgsz=400,
    plots=True
)