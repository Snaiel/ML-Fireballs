from pathlib import Path
import os, shutil


MULTI_TIERED_FOLDER = Path(Path(__file__).parents[3], "data", "multi_tiered")
MULTI_YAML = Path(Path(__file__).parents[2], "cfg", "multi_tiered.yaml")


def get_folder_path(folder_name: str) -> Path:
    return Path(MULTI_TIERED_FOLDER, folder_name)


def prepare_folders(folder_name: str, sub_folders: tuple[str]) -> None:
    if not MULTI_TIERED_FOLDER.exists():
        os.mkdir(MULTI_TIERED_FOLDER)

    folder_path = get_folder_path(folder_name)
    if Path(folder_path).exists():
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)
    
    data_components = ("images", "labels")
    sub_sets = ("train", "val", "test") # train and val aren't used. just required format by yolov8.

    for sub_folder in sub_folders:
        os.mkdir(Path(folder_path, sub_folder))

        with open(MULTI_YAML, 'r') as yaml_file:
            yaml_content = yaml_file.read()

        yaml_content = yaml_content.replace(
            "multi_tiered/",
            f"multi_tiered/{folder_name}/{sub_folder}"
        )

        with open(Path(folder_path, sub_folder, "multi_tiered.yaml"), 'w') as file:
            file.write(yaml_content)

        for component in data_components:
            os.mkdir(Path(folder_path, sub_folder, component))
            for sub_set in sub_sets:
                os.mkdir(Path(folder_path, sub_folder, component, sub_set))