"""
    above2300atfull
    above1000at2560
    500to1000at1280
    below500at640
"""

from ultralytics.models.yolo.detect import DetectionValidator

args = dict(
    model="runs/detect/train15/weights/best.pt",
    data="../data/multi_tiered_test_data/below500at640/multi_tiered.yaml",
    imgsz=640,
    plots=True,
    split="test",
    mode="multi"
)
validator = DetectionValidator(args=args)
validator()