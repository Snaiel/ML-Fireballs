from object_detection.detectors.detector import Detector
from object_detection.detectors.onnx import ONNXDetector
from object_detection.detectors.ultralytics import UltralyticsDetector


class DetectorSingleton:
    
    _detector: Detector = None

    @staticmethod
    def get_detector(detector: str, model_path: str, conf: float) -> Detector:
        if DetectorSingleton._detector is None:
            DetectorSingleton._detector = get_detector(detector, model_path, conf)
        return DetectorSingleton._detector


def get_detector(name: str, path: str, conf: float) -> Detector:
    detector = None
    if name == 'Ultralytics':
        detector = UltralyticsDetector(path, conf)
    elif name == 'ONNX':
        detector = ONNXDetector(path, conf)
    else:
        print(f"Unsupported detector type: {name}")
    return detector