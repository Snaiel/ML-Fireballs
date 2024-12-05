from object_detection.detectors import Detector
from ultralytics import YOLO
import numpy as np


class UltralyticsDetector(Detector):

    def __init__(self, path: str) -> None:
        self.model = YOLO(path)
    

    def detect(self, image: np.ndarray):
        results = self.model.predict(
            image,
            verbose=False,
            imgsz=416
        )

        boxes = results[0].boxes
        
        return boxes.xyxy.cpu(), boxes.conf.cpu(), boxes.cls.cpu()