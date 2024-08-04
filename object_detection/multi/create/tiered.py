from pathlib import Path
from dataset import GFO_PICKINGS
from dataset.point_pickings import PointPickings
from dataset.create.raw import RawFireball
from dataset.tile_centred import TileCentredFireball
from dataset.utils import get_train_val_test_split
from multi.create import prepare_folders, get_folder_path


def main():
    folder_name = "tiered_test_set"
    folder_path = get_folder_path(folder_name)

    sub_folders = ("atfull", "at2560", "at1280", "at640")
    
    prepare_folders(folder_name, sub_folders)

    fireballs = []

    fireball_dataset = get_train_val_test_split()
    for fireball_filename in fireball_dataset["test"]:
        name = fireball_filename.split(".")[0]
        pp = PointPickings(Path(GFO_PICKINGS, name + ".csv"))
        fireballs.append((name, pp))

    print("atfull")
    for name, pp in fireballs:
        fireball = RawFireball(name, pp)
        fireball.save_image(Path(folder_path, "atfull", "images", "test"))
        fireball.save_label(Path(folder_path, "atfull", "labels", "test"))
    

    print("at2560")
    for name, pp in fireballs:
        fireball = TileCentredFireball(name, pp, (2560, 2560))
        fireball.save_image(Path(folder_path, "at2560", "images", "test"))
        fireball.save_label(Path(folder_path, "at2560", "labels", "test"))
    

    print("at1280")
    for name, pp in fireballs:
        fireball = TileCentredFireball(name, pp, (1280, 1280))
        fireball.save_image(Path(folder_path, "at1280", "images", "test"))
        fireball.save_label(Path(folder_path, "at1280", "labels", "test"))
    

    print("at640")
    for name, pp in fireballs:
        fireball = TileCentredFireball(name, pp, (640, 640))
        fireball.save_image(Path(folder_path, "at640", "images", "test"))
        fireball.save_label(Path(folder_path, "at640", "labels", "test"))


if __name__ == "__main__":
    main()