import numpy as np
from ultralytics import YOLO

from object_detection.detectors.detector import Detector
from utils.constants import DETECTOR_IOU


class UltralyticsDetector(Detector):

    def __init__(self, path: str, conf: float) -> None:
        super().__init__(path, conf)
        self.model = YOLO(self.path, task="detect")
    

    def detect(self, image: np.ndarray) -> tuple:
        results = self.model.predict(
            image,
            verbose=False,
            imgsz=416,
            conf=self.conf,
            iou=DETECTOR_IOU
        )

        boxes = results[0].boxes
        
        return boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()