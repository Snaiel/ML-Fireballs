from ultralytics import YOLO

model = YOLO("yolov8n.pt")

data = "../data/fireball_object_detection/data.yaml"

model.train(
    data=data,
    epochs=100,
    imgsz=640,
    pretrained=True
)

model.val(
    data=data,
    imgsz=640,
    plots=True,
    split="test"
)