from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.tune(
    data="../data/object_detection/data.yaml",
    imgsz=640,
    pretrained=True,
    epochs=75,
    iterations=50,
    optimizer="AdamW"
)