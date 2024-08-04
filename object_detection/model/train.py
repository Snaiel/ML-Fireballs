from ultralytics import YOLO
import yaml
from pathlib import Path


model = YOLO("yolov8n.pt")
# model = YOLO("runs/detect/train16/weights/last.pt")

data = "../data/object_detection/data.yaml"


kwargs = {}

USE_TUNED_ARGS = False
tuned_args_yaml = Path(Path(__file__).parents[1], "cfg", "tuned.yaml")
if USE_TUNED_ARGS:
    with open(tuned_args_yaml, 'r') as file:
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
    plots=True,
    split="test"
)