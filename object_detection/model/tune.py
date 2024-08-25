from ultralytics import YOLO
import yaml
from pathlib import Path

model = YOLO("yolov8n.pt")

data = Path(Path(__file__).parents[2], "data/object_detection_tune/data.yaml")

kwargs = {}
with open(Path(Path(__file__).parents[1], "cfg", "split_tiles.yaml"), 'r') as file:
    kwargs = yaml.safe_load(file)

search_space = {  # key: (min, max, gain(optional))
    # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
    "lr0": (1e-5, 1e-1),  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    "lrf": (0.0001, 0.1),  # final OneCycleLR learning rate (lr0 * lrf)
    "momentum": (0.7, 0.98, 0.3),  # SGD momentum/Adam beta1
    "weight_decay": (0.0, 0.001),  # optimizer weight decay 5e-4
    "warmup_epochs": (0.0, 5.0),  # warmup epochs (fractions ok)
    "warmup_momentum": (0.0, 0.95),  # warmup initial momentum
    "box": (1.0, 20.0),  # box loss gain
    "cls": (0.2, 4.0),  # cls loss gain (scale with pixels)
    "dfl": (0.4, 6.0),  # dfl loss gain
    "hsv_h": (0.0, 0.1),  # image HSV-Hue augmentation (fraction)
    "hsv_s": (0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
    "hsv_v": (0.0, 0.9),  # image HSV-Value augmentation (fraction)
    "degrees": (0.0, 180),  # image rotation (+/- deg)
    "translate": (0.0, 0.9),  # image translation (+/- fraction)
    "scale": (0.0, 0.95),  # image scale (+/- gain)
    "flipud": (0.0, 1.0),  # image flip up-down (probability)
    "fliplr": (0.0, 1.0),  # image flip left-right (probability)
    "mosaic": (0.0, 1.0),  # image mixup (probability)
}

model.tune(
    kwargs=kwargs,
    data=data,
    space=search_space,
    imgsz=400,
    pretrained=True,
    epochs=50,
    iterations=100,
    optimizer="Adam",
    val=False,
    save=False,
    plots=False
)