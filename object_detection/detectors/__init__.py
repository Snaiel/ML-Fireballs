from object_detection.detectors.detector import Detector
from object_detection.detectors.onnx import ONNXDetector
from object_detection.detectors.ultralytics import UltralyticsDetector
from utils.logging import get_logger


logger = get_logger()


class DetectorSingleton:
    
    _detector: Detector = None

    @staticmethod
    def get_detector(detector: str, model_path: str) -> Detector:
        if DetectorSingleton._detector is None:
            detector_instance = get_detector(detector, model_path)
            logger.info(
                "new_detector_instance",
                new_detector_instance={
                    "detector": detector,
                    "model_path": model_path,
                    "detector_instance": detector_instance
                }
            )
            DetectorSingleton._detector = detector_instance
        return DetectorSingleton._detector


def get_detector(name: str, path: str) -> Detector:
    detector = None
    if name == 'Ultralytics':
        detector = UltralyticsDetector(path)
    elif name == 'ONNX':
        detector = ONNXDetector(path)
    else:
        print(f"Unsupported detector type: {name}")
    return detector