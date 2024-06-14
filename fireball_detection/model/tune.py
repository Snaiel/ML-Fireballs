from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.tune(
    data="../data/fireball_object_detection/data.yaml",
    imgsz=640,
    pretrained=True,
    epochs=75,
    iterations=50,
    optimizer="AdamW"
)

# model.train(
#     project="fireballs",
#     data="../data/fireball_object_detection/data.yaml",
#     epochs=100,
#     imgsz=640,
#     pretrained=True,
#     save_period=20
# )

# model.val(
#     data="../data/fireball_object_detection/data.yaml",
#     imgsz=640,
#     plots=True,
#     split="test"
# )