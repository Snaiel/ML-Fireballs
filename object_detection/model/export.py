import argparse
import json
from dataclasses import dataclass

from ultralytics import YOLO
from onnx.checker import check_model

def main() -> None:
    @dataclass
    class Args:
        yolo_pt_path: str
    
    parser = argparse.ArgumentParser(description="Export a YOLO model to ONNX format.")
    parser.add_argument(
        "--yolo_pt_path",
        type=str,
        required=True,
        help="Path to the YOLO model weights in .pt format."
    )
    
    args = Args(**vars(parser.parse_args()))
    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    model = YOLO(args.yolo_pt_path)
    onnx_model_path = model.export(format="onnx")

    check_model(onnx_model_path)
    

if __name__ == "__main__":
    main()