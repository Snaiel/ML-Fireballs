import json
import os
import shutil
from pathlib import Path


def remove_saved_detection(output_folder: Path, detection: str):
    (image_folder, detection_name) = detection.split("/")

    image_folder_path = Path(output_folder, image_folder)
    json_file_path = Path(image_folder_path, image_folder + ".json")

    with open(json_file_path) as json_file:
        json_data: dict = json.load(json_file)
    
    json_data["detections"] = [d for d in json_data["detections"] if d["name"] != detection_name]
    
    if len(json_data["detections"]) == 0:
        shutil.rmtree(image_folder_path)
    else:
        with open(json_file_path, "w") as json_file:
            json.dump(json_data, json_file, indent=4)
        
        for ext in [".jpg", ".differenced.jpg"]:
            file_path = image_folder_path / f"{detection_name}{ext}"
            if file_path.exists():
                os.remove(file_path)