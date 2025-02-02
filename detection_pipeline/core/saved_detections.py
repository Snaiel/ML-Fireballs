import json
import os
import shutil
from pathlib import Path


def get_detections(output_folder: str) -> list[str]:
    """
    output_folder is the folder containing subfolders for each image

    ```txt
    DFNSMALL41 <- this one
        41_2015-01-14_110659_DSC_0108
            41_2015-01-14_110659_DSC_0108.json
            41_2015-01-14_110659_DSC_0108.thumb.differenced.jpg 
            41_2015-01-14_110659_DSC_0108.thumb.jpg 
            41_2015-01-14_110659_DSC_0108_20_5148-2666-5216-2805.differenced.jpg 
            41_2015-01-14_110659_DSC_0108_20_5148-2666-5216-2805.jpg
        41_2015-01-14_110729_DSC_0109
        41_2015-01-14_110959_DSC_0114
    ```
    """

    detections = []

    image_folders = [i for i in sorted(os.listdir(output_folder)) if Path(output_folder, i).is_dir()]

    for image_folder in image_folders:
        with open(Path(output_folder, image_folder, f"{image_folder}.json")) as json_file:
            json_data = json.load(json_file)
            for detection_name in map(lambda x: x["name"], json_data["detections"]):
                detections.append(f"{image_folder}/{detection_name}")

    return detections


def remove_saved_detection(output_folder: Path, detection: str):
    """
    output_folder is the folder containing subfolders for each image

    ```txt
    DFNSMALL41 <- this one
        41_2015-01-14_110659_DSC_0108
            41_2015-01-14_110659_DSC_0108.json
            41_2015-01-14_110659_DSC_0108.thumb.differenced.jpg 
            41_2015-01-14_110659_DSC_0108.thumb.jpg 
            41_2015-01-14_110659_DSC_0108_20_5148-2666-5216-2805.differenced.jpg 
            41_2015-01-14_110659_DSC_0108_20_5148-2666-5216-2805.jpg
        41_2015-01-14_110729_DSC_0109
        41_2015-01-14_110959_DSC_0114
    ```
    
    detection is in the format <image>/<detection_name> e.g.

    ```txt
    41_2015-01-14_110659_DSC_0108/41_2015-01-14_110659_DSC_0108_20_5148-2666-5216-2805
    ```
    """

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