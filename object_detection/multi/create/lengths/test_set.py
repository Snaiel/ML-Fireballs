from math import sqrt
from pathlib import Path

from dataset import GFO_PICKINGS
from object_detection.dataset.point_pickings import PointPickings
from object_detection.dataset.utils import get_train_val_test_split
from multi.create import prepare_folders
from multi.create.lengths import SUB_FOLDERS, create_folder


def main():
    folder_name = "lengths_test_set"
    prepare_folders(folder_name, SUB_FOLDERS)

    pp_lengths = []
    fireball_dataset = get_train_val_test_split()
    for fireball_filename in fireball_dataset["test"]:
        name = fireball_filename.split(".")[0]
        pp = PointPickings(Path(GFO_PICKINGS, name + ".csv"))
        length = sqrt((pp.pp_max_x - pp.pp_min_x)**2 + (pp.pp_max_y - pp.pp_min_y)**2)
        pp_lengths.append((name, pp, length))
    
    create_folder(folder_name, pp_lengths)


if __name__ == "__main__":
    main()