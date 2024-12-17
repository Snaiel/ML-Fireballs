from object_detection.detectors.detector import Detector
from object_detection.detectors.onnx import ONNXDetector
from object_detection.detectors.ultralytics import UltralyticsDetector


def get_detector(name: str, path: str) -> Detector:
    detector = None
    if name == 'Ultralytics':
        detector = UltralyticsDetector(path)
    elif name == 'ONNX':
        detector = ONNXDetector(path)
    else:
        print(f"Unsupported detector type: {name}")
    return detector