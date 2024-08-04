from pathlib import Path, PosixPath

import pandas as pd
from dataset.create.raw import RawFireball
from dataset.tile_centred import TileCentredFireball
from dataset.point_pickings import PointPickings
from multi.create import get_folder_path


SUB_FOLDERS = (
    "above2300atfull", 
    "1000to2300at2560", 
    "500to1000at1280", 
    "below500at640"
)


def create_folder(folder_name: PosixPath, pp_lengths: list[tuple[str, PointPickings, float]]) -> None:
    folder_path = get_folder_path(folder_name)
    pp_df = pd.DataFrame(pp_lengths, columns=["name", "point_pickings", "length"])

    print(pp_df)


    pp_l_above_2300 = pp_df[pp_df["length"] > 2300]
    print(f"\n\nabove2300atfull: {len(pp_l_above_2300)}")
    print(pp_l_above_2300)
    for index, row in pp_l_above_2300.iterrows():
        fireball = RawFireball(row[0], row[1])
        fireball.save_image(Path(folder_path, "above2300atfull", "images", "test"))
        fireball.save_label(Path(folder_path, "above2300atfull", "labels", "test"))


    # pp_l_above_1000 = pp_df[pp_df["length"] > 1000]
    # print(f"\n\nabove1000at2560: {len(pp_l_above_1000)}")
    # print(pp_l_above_1000)
    # for index, row in pp_l_above_1000.iterrows():
    #     fireball = TileCentredFireball(row[0], row[1], (2560, 2560))
    #     fireball.save_image(Path(folder_path, "above1000at2560", "images", "test"))
    #     fireball.save_label(Path(folder_path, "above1000at2560", "labels", "test"))


    pp_l_1000_to_2300 = pp_df[(1000 < pp_df["length"]) & (pp_df["length"] <= 2300)]
    print(f"\n\n1000to2300at2560: {len(pp_l_1000_to_2300)}")
    print(pp_l_1000_to_2300)
    for index, row in pp_l_1000_to_2300.iterrows():
        fireball = TileCentredFireball(row[0], row[1], (2560, 2560))
        fireball.save_image(Path(folder_path, "1000to2300at2560", "images", "test"))
        fireball.save_label(Path(folder_path, "1000to2300at2560", "labels", "test"))
    

    pp_l_500_to_1000 = pp_df[(500 <= pp_df["length"]) & (pp_df["length"] <= 1000)]
    print(f"\n\n500to1000at1280: {len(pp_l_500_to_1000)}")
    print(pp_l_500_to_1000)
    for index, row in pp_l_500_to_1000.iterrows():
        fireball = TileCentredFireball(row[0], row[1], (1280, 1280))
        fireball.save_image(Path(folder_path, "500to1000at1280", "images", "test"))
        fireball.save_label(Path(folder_path, "500to1000at1280", "labels", "test"))
    

    pp_l_below_500 = pp_df[pp_df["length"] < 500]
    print(f"\n\nbelow500at640: {len(pp_l_below_500)}")
    print(pp_l_below_500)
    for index, row in pp_l_below_500.iterrows():
        fireball = TileCentredFireball(row[0], row[1], (640, 640))
        fireball.save_image(Path(folder_path, "below500at640", "images", "test"))
        fireball.save_label(Path(folder_path, "below500at640", "labels", "test"))