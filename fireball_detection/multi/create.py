from pathlib import Path
import os, shutil
import pandas as pd
from dataset import GFO_PICKINGS
from dataset.point_pickings import PointPickings
from dataset.create.raw import RawFireball
from dataset.create.tile_centred import TileCentredFireball
from math import sqrt


MULTI_TIERED_TEST_FOLDER = Path(Path(__file__).parents[2], "data", "multi_layered_test_data")
MULTI_YAML = Path(Path(__file__).parents[1], "cfg", "multi_layered.yaml")


def main():
    if Path(MULTI_TIERED_TEST_FOLDER).exists():
        shutil.rmtree(MULTI_TIERED_TEST_FOLDER)
    os.mkdir(MULTI_TIERED_TEST_FOLDER)
    
    folders = ("above2300atfull", "above1000at2560", "500to1000at1280", "below500at640")
    sub_folders = ("images", "labels")
    sub_sets = ("train", "val", "test") # train and val aren't used. just required format by yolov8.

    for folder in folders:
        os.mkdir(Path(MULTI_TIERED_TEST_FOLDER, folder))


        with open(MULTI_YAML, 'r') as yaml_file:
            yaml_content = yaml_file.read()

        yaml_content = yaml_content.replace(
            "multi_layered_test_data/",
            f"multi_layered_test_data/{folder}"
        )

        with open(Path(MULTI_TIERED_TEST_FOLDER, folder, "multi_layered.yaml"), 'w') as file:
            file.write(yaml_content)


        for sub_folder in sub_folders:
            os.mkdir(Path(MULTI_TIERED_TEST_FOLDER, folder, sub_folder))
            for sub_set in sub_sets:
                os.mkdir(Path(MULTI_TIERED_TEST_FOLDER, folder, sub_folder, sub_set))

    pp_lengths = []

    for pickings_csv in os.listdir(GFO_PICKINGS):
        name = pickings_csv.split(".")[0]
        pp = PointPickings(Path(GFO_PICKINGS, pickings_csv))
        length = sqrt((pp.pp_max_x - pp.pp_min_x)**2 + (pp.pp_max_y - pp.pp_min_y)**2)
        pp_lengths.append((name, pp, length))

    pp_df = pd.DataFrame(pp_lengths, columns=["name", "point_pickings", "length"])

    print(pp_df)

    pp_l_above_2300 = pp_df[pp_df["length"] > 2300]
    print(len(pp_l_above_2300))
    print(pp_l_above_2300)
    for index, row in pp_l_above_2300.iterrows():
        fireball = RawFireball(row[0], row[1])
        fireball.save_image(Path(MULTI_TIERED_TEST_FOLDER, "above2300atfull", "images", "test"))
        fireball.save_label(Path(MULTI_TIERED_TEST_FOLDER, "above2300atfull", "labels", "test"))
    

    pp_l_above_1000 = pp_df[pp_df["length"] > 1000]
    print(len(pp_l_above_1000))
    print(pp_l_above_1000)
    for index, row in pp_l_above_1000.iterrows():
        fireball = TileCentredFireball(row[0], row[1], (2560, 2560))
        fireball.save_image(Path(MULTI_TIERED_TEST_FOLDER, "above1000at2560", "images", "test"))
        fireball.save_label(Path(MULTI_TIERED_TEST_FOLDER, "above1000at2560", "labels", "test"))
    

    pp_l_500_to_1000 = pp_df[(500 <= pp_df["length"]) & (pp_df["length"] <= 1000)]
    print(len(pp_l_500_to_1000))
    print(pp_l_500_to_1000)
    for index, row in pp_l_500_to_1000.iterrows():
        fireball = TileCentredFireball(row[0], row[1], (1280, 1280))
        fireball.save_image(Path(MULTI_TIERED_TEST_FOLDER, "500to1000at1280", "images", "test"))
        fireball.save_label(Path(MULTI_TIERED_TEST_FOLDER, "500to1000at1280", "labels", "test"))
    

    pp_l_below_500 = pp_df[pp_df["length"] < 500]
    print(len(pp_l_below_500))
    print(pp_l_below_500)
    for index, row in pp_l_below_500.iterrows():
        fireball = TileCentredFireball(row[0], row[1], (640, 640))
        fireball.save_image(Path(MULTI_TIERED_TEST_FOLDER, "below500at640", "images", "test"))
        fireball.save_label(Path(MULTI_TIERED_TEST_FOLDER, "below500at640", "labels", "test"))


if __name__ == "__main__":
    main()