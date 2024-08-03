"""
    above2300atfull
    1000to2300at2560
    500to1000at1280
    below500at640

    ../data/multi_tiered/lengths_test_set/above2300atfull/multi_tiered.yaml
    ../data/multi_tiered/lengths_all_data/above2300atfull/multi_tiered.yaml


    atfull
    at2560
    at1280
    at640

    ../data/multi_tiered/tiered_test_set/atfull/multi_tiered.yaml
"""

from ultralytics.models.yolo.detect import DetectionValidator

args = dict(
    model="runs/detect/train15/weights/best.pt",
    data="../data/multi_tiered/lengths_test_set/below500at640/multi_tiered.yaml",
    imgsz=640,
    plots=True,
    split="test",
    mode="multi"
)
validator = DetectionValidator(args=args)
validator()