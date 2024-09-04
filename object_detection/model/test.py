from ultralytics import YOLO
from pathlib import Path


model = YOLO(Path(Path(__file__).parents[2], "data", "e15.pt"))
data = Path(Path(__file__).parents[2], "data", "object_detection", "data.yaml")

model.val(
    data=data,
    imgsz=400,
    plots=True,
    split="test",
    conf=0.25
)