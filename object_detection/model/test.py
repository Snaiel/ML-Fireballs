from ultralytics import YOLO
from pathlib import Path


model = YOLO(Path(Path(__file__).parents[2], "data", "kfold_runs", "split0", "weights", "last.pt"))
data = Path(Path(__file__).parents[2], "data", "1_to_1_kfold_object_detection", "split0", "data.yaml")

model.val(
    data=data,
    imgsz=400,
    plots=True,
    split="val",
    conf=0.25,
    augment=False,
)