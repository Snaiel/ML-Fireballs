import numpy as np
from ultralytics import YOLO

from object_detection.detectors.detector import Detector


class UltralyticsDetector(Detector):

    def __init__(self, path) -> None:
        self.model = YOLO(path, task="detect")
    

    def detect(self, image: np.ndarray):
        results = self.model.predict(
            image,
            verbose=False,
            imgsz=416,
            conf=0.25,
            iou=0.5
        )

        boxes = results[0].boxes
        
        return boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()