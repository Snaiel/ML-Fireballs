from ultralytics import YOLO
from train_greyscale_model import GreyScaleValidator

model = YOLO("runs/detect/train17/weights/best.pt")
validation_results = model.val(
    GreyScaleValidator,
    data="yolov8_fireball_dataset/data.yaml",
    imgsz=640,
    plots=True,
    split="test",
    ch=1
)