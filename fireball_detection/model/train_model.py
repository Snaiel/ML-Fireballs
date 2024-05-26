from ultralytics import YOLO

model = YOLO("yolov8m.pt")
results = model.train(
    data="yolov8_fireball_dataset/data.yaml",
    epochs=100,
    imgsz=1280,
    pretrained=True,
    save_period=20,
    batch=-1
)