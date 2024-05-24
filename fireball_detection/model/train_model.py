from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(
    data="yolov8_fireball_dataset/data.yaml",
    epochs=100,
    imgsz=640,
    pretrained=False,
    save_period=20
)