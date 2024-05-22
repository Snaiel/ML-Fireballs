import pandas as pd
import numpy as np
from pathlib import Path

IMAGE_DIM = (7360, 4912)
PADDING = 0.05

def get_yolov8_label_from_point_pickings_csv(csv_file: str) -> list[float, float, float, float]:
    point_pickings_df = pd.read_csv(csv_file)

    pp_min_x = point_pickings_df['x_image_thumb'].min()
    pp_max_x = point_pickings_df['x_image_thumb'].max()

    pp_min_y = point_pickings_df['y_image_thumb'].min()
    pp_max_y = point_pickings_df['y_image_thumb'].max()

    point_pickings_dim = (pp_max_x - pp_min_x, pp_max_y - pp_min_y)


    bb_min_x = pp_min_x - (point_pickings_dim[0] * PADDING)
    bb_max_x = pp_max_x + (point_pickings_dim[0] * PADDING)

    bb_min_y = pp_min_y - (point_pickings_dim[1] * PADDING)
    bb_max_y = pp_max_y + (point_pickings_dim[1] * PADDING)

    bb_min_x = np.clip(bb_min_x, 0, IMAGE_DIM[0])
    bb_max_x = np.clip(bb_max_x, 0, IMAGE_DIM[0])

    bb_min_y = np.clip(bb_min_y, 0, IMAGE_DIM[1])
    bb_max_y = np.clip(bb_max_y, 0, IMAGE_DIM[1])


    bounding_box_dim = (bb_max_x - bb_min_x, bb_max_y - bb_min_y)


    bb_centre_x = (bb_min_x + bb_max_x) / 2
    bb_centre_y = (bb_min_y + bb_max_y) / 2

    norm_bb_centre_x = bb_centre_x / IMAGE_DIM[0]
    norm_bb_centre_y = bb_centre_y / IMAGE_DIM[1]

    norm_bb_width = bounding_box_dim[0] / IMAGE_DIM[0]
    norm_bb_height = bounding_box_dim[1] / IMAGE_DIM[1]

    label = [norm_bb_centre_x, norm_bb_centre_y, norm_bb_width, norm_bb_height]

    return label

if __name__ == "__main__":
    file_path = Path(__file__)
    csv_path = Path(file_path.parents[1], "fireball_detection/GFO_fireball_object_detection_training_set/point_pickings_csvs/03_2016-07-28_043558_K_DSC_8287.csv")
    print(get_yolov8_label_from_point_pickings_csv(csv_path))