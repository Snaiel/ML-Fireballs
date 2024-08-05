import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io

from fireball_detection.included import (SQUARE_SIZE,
                                         retrieve_included_coordinates)
from object_detection.dataset import GFO_JPEGS, GFO_THUMB_EXT, IMAGE_DIM
from object_detection.dataset.create import MIN_BB_DIM_SIZE, create_dataset
from object_detection.dataset.fireball import Fireball
from object_detection.dataset.point_pickings import PointPickings

included_coordinates = retrieve_included_coordinates()

MIN_POINTS_IN_TILE = 3


@dataclass
class FireballTile:
    position: pd.DataFrame
    points: list[float] = None
    bb_centre: tuple[float] = tuple()
    bb_dim: tuple[int] = tuple()
    image: np.ndarray = None


class SplitTilesFireball(Fireball):

    fireball_tiles: list[FireballTile]
    negative_tiles: list[FireballTile]

    def __init__(self, fireball_name: str, point_pickings: PointPickings = None, ) -> None:
        super().__init__(fireball_name, point_pickings)

        fireball_image = io.imread(Path(GFO_JPEGS, fireball_name + GFO_THUMB_EXT))

        fireball_tiles: list[FireballTile] = []
        negative_tiles: list[FireballTile] = []

        for tile_pos in included_coordinates:
            points_in_tile = []

            for point in self.pp.pp.itertuples(False, None):
                if (
                    tile_pos[0] <= point[0] and point[0] < tile_pos[0] + SQUARE_SIZE and \
                    tile_pos[1] <= point[1] and point[1] < tile_pos[1] + SQUARE_SIZE
                ):
                    points_in_tile.append(point)
            
            if len(points_in_tile) == 0:
                negative_tiles.append(FireballTile(tile_pos))
            elif len(points_in_tile) >= MIN_POINTS_IN_TILE:
                fireball_tiles.append(
                    FireballTile(
                        tile_pos,
                        pd.DataFrame(points_in_tile, columns=["x", "y"])
                    )
                )
        
        for tile in fireball_tiles:
            pp_min_x = tile.points['x'].min()
            pp_max_x = tile.points['x'].max()
            pp_min_y = tile.points['y'].min()
            pp_max_y = tile.points['y'].max()

            points_dim = (pp_max_x - pp_min_x, pp_max_y - pp_min_y)
            padding = 0.05

            bb_min_x = pp_min_x - (points_dim[0] * padding)
            bb_max_x = pp_max_x + (points_dim[0] * padding)

            bb_min_y = pp_min_y - (points_dim[1] * padding)
            bb_max_y = pp_max_y + (points_dim[1] * padding)

            bb_centre_x = (bb_min_x + bb_max_x) / 2
            bb_centre_y = (bb_min_y + bb_max_y) / 2

            bb_width = max(bb_max_x - bb_min_x, MIN_BB_DIM_SIZE)
            bb_height = max(bb_max_y - bb_min_y, MIN_BB_DIM_SIZE)

            norm_bb_centre_x = min((bb_centre_x - tile.position[0]) / SQUARE_SIZE, 1.0)
            norm_bb_centre_y = min((bb_centre_y - tile.position[1]) / SQUARE_SIZE, 1.0)

            norm_bb_width = min(bb_width / SQUARE_SIZE, 1.0)
            norm_bb_height = min(bb_height / SQUARE_SIZE, 1.0)

            tile.bb_centre = (norm_bb_centre_x, norm_bb_centre_y)
            tile.bb_dim = (norm_bb_width, norm_bb_height)

            tile.image = fireball_image[tile.position[1] : tile.position[1] + SQUARE_SIZE, tile.position[0] : tile.position[0] + SQUARE_SIZE]
        
        self.fireball_tiles = fireball_tiles

        negative_tiles = random.sample(negative_tiles, len(fireball_tiles))
        for tile in negative_tiles:
            tile.image = fireball_image[tile.position[1] : tile.position[1] + SQUARE_SIZE, tile.position[0] : tile.position[0] + SQUARE_SIZE]
        self.negative_tiles = negative_tiles


    def save_image(self, folder: str) -> None:
        for i, tile in enumerate(self.fireball_tiles):
            io.imsave(
                Path(folder, f"{self.fireball_name}_{i}.jpg"),
                tile.image,
                check_contrast=False
            )
        for i, tile in enumerate(self.negative_tiles):
            io.imsave(
                Path(folder, f"{self.fireball_name}_negative_{i}.jpg"),
                tile.image,
                check_contrast=False
            )


    def save_label(self, folder: str) -> None:
        for i, tile in enumerate(self.fireball_tiles):
            label = (0, tile.bb_centre[0], tile.bb_centre[1], tile.bb_dim[0], tile.bb_dim[1])
            with open(Path(folder, f"{self.fireball_name}_{i}.txt"), 'x') as label_file:
                label_file.write(" ".join(str(item) for item in label))
        for i in range(len(self.negative_tiles)):
            open(Path(folder, f"{self.fireball_name}_negative_{i}.txt"), 'x').close()


if __name__ == "__main__":
    create_dataset(SplitTilesFireball)