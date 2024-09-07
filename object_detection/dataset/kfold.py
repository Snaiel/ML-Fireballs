import os
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold

from object_detection.dataset import (DATA_YAML, DATASET_FOLDER, GFO_JPEGS,
                                      RANDOM_SEED, DATA_FOLDER)
from object_detection.dataset.split_tiles import SplitTilesFireball
from tqdm import tqdm
import multiprocessing as mp


def get_train_val_test_split(dataset_size: int = None) -> dict:
    """
    returns a dictionary containing "train", "val", "test"
    lists of fireballs jpg filenames from the gfo folder.
    """
    fireball_images = os.listdir(GFO_JPEGS)

    if dataset_size is not None and dataset_size < len(fireball_images):
        fireball_images, _ = train_test_split(fireball_images, train_size=dataset_size, random_state=RANDOM_SEED)

    temp_fireballs, test_fireballs = train_test_split(fireball_images, train_size=0.8, random_state=RANDOM_SEED)
    train_fireballs, val_fireballs = train_test_split(temp_fireballs, train_size=0.8, random_state=RANDOM_SEED)
    # 64% train, 16% val, 20% test

    print("Train Val Test")
    print(len(train_fireballs), len(val_fireballs), len(test_fireballs))

    fireball_dataset = {
        "train": train_fireballs,
        "val": val_fireballs,
        "test": test_fireballs
    }

    return fireball_dataset


def create_dataset():
    # delete output, create new empty output folder
    if Path(DATASET_FOLDER).exists():
        shutil.rmtree(DATASET_FOLDER)
    os.mkdir(DATASET_FOLDER)

    ## Create folder structure
    # dataset_folder
    #     images
    #         train
    #         val
    #         test
    #     labels
    #         train
    #         val
    #         test
    #     data.yaml

    # Copy the data.yaml file to the dataset folder
    shutil.copy(DATA_YAML, DATASET_FOLDER)

    # Create folders
    folders = ("images", "labels")
    sub_folders = ("train", "val")

    for folder in folders:
        os.mkdir(Path(DATASET_FOLDER, folder))
        for sub_folder in sub_folders:
            os.mkdir(Path(DATASET_FOLDER, folder, sub_folder))
    
    fireball_dataset = get_train_val_test_split()
    for dataset, fireballs in fireball_dataset.items():
        print(f"Creating {dataset} dataset...")
        for fireball_filename in fireballs:
            fireball_name = fireball_filename.split(".")[0]
            fireball = SplitTilesFireball(fireball_name)
            fireball.save_images(Path(DATASET_FOLDER, "images", dataset))
            fireball.save_labels(Path(DATASET_FOLDER, "labels", dataset))


# prefix components:
space =  '    '
branch = '│   '
# pointers:
tee =    '├── '
last =   '└── '


def print_tree(dir_path: Path) -> None:
    for line in tree(dir_path):
        print(line)


def tree(dir_path: Path, prefix: str=''):
    """    
    A recursive generator, given a directory Path object
    will yield a visual tree structure line by line
    with each line prefixed by the same characters
    
    https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
    """    
    
    contents = sorted(list(dir_path.iterdir()))
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents):
        yield prefix + pointer + path.name
        if path.is_dir(): # extend the prefix and recurse:
            extension = branch if pointer == tee else space 
            # i.e. space because last, └── , above so no more |
            yield from tree(path, prefix=prefix+extension)


def create_kfolds() -> None:
    fireball_images = sorted(os.listdir(GFO_JPEGS))

    kf = KFold(n_splits=5)
    
    for i, (train_indexes, val_indexes) in enumerate(kf.split(fireball_images)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_indexes} length={len(train_indexes)}")
        print(f"  Test:  index={val_indexes} length={len(val_indexes)}")
    
    KFOLD_OBJECT_DETECTION_FOLDER = Path(DATA_FOLDER, "kfold_object_detection")
    if Path(KFOLD_OBJECT_DETECTION_FOLDER).exists():
        shutil.rmtree(KFOLD_OBJECT_DETECTION_FOLDER)
    os.mkdir(KFOLD_OBJECT_DETECTION_FOLDER)

    for i in range(5):
        fold_folder = Path(KFOLD_OBJECT_DETECTION_FOLDER, f"fold{i}")
        os.mkdir(fold_folder)
        shutil.copy(DATA_YAML, fold_folder)
        with open(Path(fold_folder, "data.yaml"), "r+") as yaml_file:
            content = yaml_file.read()
            content = content.replace("object_detection", f"kfold_object_detection/fold{i}")
            yaml_file.seek(0)
            yaml_file.write(content)
            yaml_file.truncate()
        for folder in ("images", "labels"):
            os.mkdir(Path(fold_folder, folder))
            for sub_folder in ("train", "val"):
                os.mkdir(Path(fold_folder, folder, sub_folder))
    
    print("\n\nCreated folder structure in data folder:\n")
    print_tree(KFOLD_OBJECT_DETECTION_FOLDER)

    def create_fold_dataset(fold: int, name: str, indexes) -> None:
        position = fold * 2 + (1 if name == "val" else 0)
        for i in tqdm(indexes, position=position, desc=f"fold {fold} {name}"):
            fireball_name = fireball_images[i].split(".")[0]
            fireball = SplitTilesFireball(fireball_name)
            fireball.save_images(Path(KFOLD_OBJECT_DETECTION_FOLDER, f"fold{fold}", "images", name))
            fireball.save_labels(Path(KFOLD_OBJECT_DETECTION_FOLDER, f"fold{fold}", "labels", name))

    print("\n\nCreating fold datasets...\n")

    procs = []
    for i, (train_indexes, val_indexes) in enumerate(kf.split(fireball_images)):
        train_proc = mp.Process(target=create_fold_dataset, args=(i, "train", train_indexes))
        procs.append(train_proc)
        train_proc.start()
        val_proc = mp.Process(target=create_fold_dataset, args=(i, "val", val_indexes))
        procs.append(val_proc)
        val_proc.start()
    
    for proc in procs:
        proc.join()



if __name__ == "__main__":
    create_kfolds()