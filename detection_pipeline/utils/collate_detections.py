import os
from pathlib import Path
from dataclasses import dataclass
import argparse
import json

from detection_pipeline.utils.parse_logs import parse_logs


@dataclass
class Args:
    processed_folder: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collate detections from cameras into one file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--processed_folder",
        type=str,
        required=True,
        help="Path to the folder of a given night containing the detections for each camera."
    )
    
    args = Args(**vars(parser.parse_args()))
    print("\nargs:", json.dumps(vars(args), indent=4), "\n")

    processed_folder_path = Path(args.processed_folder)

    camera_folders = [
        i for i in
        sorted(os.listdir(processed_folder_path))
        if Path(processed_folder_path, i).is_dir()
    ]
    
    detections = []
    erroneous_detections = []
    final_detections = []

    for camera in camera_folders:
        log_file = next(
            (
                file for file
                in os.listdir(Path(processed_folder_path, camera))
                if ".log" in file
            ),
            None
        )

        log_file_path = Path(processed_folder_path, camera, log_file)

        print(log_file_path)

        parsed_logs = parse_logs(log_file_path)

        added_detections = False
        added_erroneous_detections = False
        added_final_detections = False

        new_label = lambda x: f"{camera}/{x}"

        for log in reversed(parsed_logs):

            if "detections" in log and not added_detections:
                detections.extend(map(new_label, log["detections"]))
                added_detections = True

            if "erroneous_detections" in log and not added_erroneous_detections:
                erroneous_detections.extend(map(new_label, log["erroneous_detections"]))
                added_erroneous_detections = True

            if "final_detections" in log and not added_final_detections:
                final_detections.extend(map(new_label, log["final_detections"]))
                added_final_detections = True
            
            if added_detections and added_erroneous_detections and added_final_detections:
                break
    
    detections.sort()
    erroneous_detections.sort()
    final_detections.sort()

    print()
    for i in detections:
        print(i)
    
    print()
    for i in erroneous_detections:
        print(i)
    
    print()
    for i in final_detections:
        print(i)
    
    json_data = {
        "total_detections": len(detections),
        "detections": detections,
        "total_erroneous_detections": len(erroneous_detections),
        "erroneous_detections": erroneous_detections,
        "total_final_detections": len(final_detections),
        "final_detections": final_detections
    }

    with open(Path(processed_folder_path, f"{processed_folder_path.name}.json"), "w") as json_file:
        json.dump(json_data, json_file, indent=4)


if __name__ == "__main__":
    main()