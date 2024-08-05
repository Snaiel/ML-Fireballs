import os

from sklearn.model_selection import train_test_split

from object_detection.dataset import GFO_JPEGS, RANDOM_SEED


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