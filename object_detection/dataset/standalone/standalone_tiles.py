import random
from pathlib import Path

import pandas as pd
from skimage import io

from fireball_detection.tiling.included import (SQUARE_SIZE,
                                                retrieve_included_coordinates)
from object_detection.dataset import (GFO_JPEGS, GFO_PICKINGS, GFO_THUMB_EXT,
                                      MIN_POINTS_IN_TILE, RANDOM_SEED)
from object_detection.dataset.dataset_tiles import (DatasetTiles, FireballTile,
                                                    plot_fireball_tile)


included_coordinates = retrieve_included_coordinates()


class StandaloneTiles(DatasetTiles):

    def __init__(self, fireball_name: str, negative_ratio: int) -> None:
        super().__init__(fireball_name)

        fireball_image = io.imread(Path(GFO_JPEGS, self.fireball_name + GFO_THUMB_EXT))
        points = pd.read_csv(Path(GFO_PICKINGS, self.fireball_name + ".csv"))
        
        for tile_pos in included_coordinates:

            tile_image = fireball_image[tile_pos[1] : tile_pos[1] + SQUARE_SIZE, tile_pos[0] : tile_pos[0] + SQUARE_SIZE]

            points_in_tile = []

            for point in points.itertuples(False, None):
                if (
                    tile_pos[0] <= point[0] and point[0] < tile_pos[0] + SQUARE_SIZE and \
                    tile_pos[1] <= point[1] and point[1] < tile_pos[1] + SQUARE_SIZE
                ):
                    points_in_tile.append(point)
            
            if len(points_in_tile) == 0:
                self.negative_tiles.append(
                    FireballTile(
                        tile_pos,
                        tile_image
                    )
                )
            elif len(points_in_tile) >= MIN_POINTS_IN_TILE:
                self.fireball_tiles.append(
                    FireballTile(
                        tile_pos,
                        tile_image,
                        pd.DataFrame(points_in_tile, columns=["x", "y"])
                    )
                )
        
        self.assign_tile_bounding_boxes()

        # Assign images to negative tiles
        if len(self.fireball_tiles) * negative_ratio > len(self.negative_tiles):
            sample_size = len(self.negative_tiles)
        else:
            sample_size = len(self.fireball_tiles) * negative_ratio
        self.negative_tiles = random.Random(RANDOM_SEED).sample(self.negative_tiles, sample_size)


def main():
    fireball = StandaloneTiles("03_2020-07-20_041559_K_DSC_9695")
    for i, tile in enumerate(fireball.fireball_tiles):
        plot_fireball_tile(fireball.fireball_name, i, tile)


if __name__ == "__main__":
    main()