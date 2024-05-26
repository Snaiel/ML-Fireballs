from ultralytics import YOLO

model = YOLO("runs/detect/train13/weights/best.pt")
test_results = model.val(
    data="yolov8_fireball_dataset/data.yaml",
    imgsz=640,
    plots=True,
    split="test"
)