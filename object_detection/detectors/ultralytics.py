from object_detection.detectors import Detector
from ultralytics import YOLO
import numpy as np


class UltralyticsDetector(Detector):

    def __init__(self, path) -> None:
        self.model = YOLO(path, task="detect")
    

    def detect(self, image: np.ndarray):
        results = self.model.predict(
            image,
            verbose=False,
            imgsz=416
        )

        boxes = results[0].boxes
        
        return boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()