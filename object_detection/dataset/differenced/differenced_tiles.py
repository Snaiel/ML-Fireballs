from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io

from fireball_detection.tiling.included import (SQUARE_SIZE,
                                                retrieve_included_coordinates)
from object_detection.dataset import GFO_PICKINGS, MIN_POINTS_IN_TILE
from object_detection.dataset.fireball_tile import (DatasetTiles, FireballTile,
                                                    plot_fireball_tile)


PIXEL_BRIGHTNESS_THRESHOLD = 10
PIXEL_TOTAL_THRESHOLD = 200


included_coordinates = retrieve_included_coordinates()


class DifferencedTiles(DatasetTiles):

    def __init__(self, differenced_image_path: str) -> None:
        image_path = Path(differenced_image_path)
        super(image_path.name.split(".")[0])

        fireball_image = io.imread(image_path)
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
            
            if len(points_in_tile) >= MIN_POINTS_IN_TILE:
                self.fireball_tiles.append(
                    FireballTile(
                        tile_pos,
                        tile_image,
                        pd.DataFrame(points_in_tile, columns=["x", "y"])
                    )
                )

            if len(points_in_tile) != 0:
                continue
            
            pixels_over_threshold = np.sum(tile_image > PIXEL_BRIGHTNESS_THRESHOLD)
            if pixels_over_threshold < PIXEL_TOTAL_THRESHOLD: continue

            self.negative_tiles.append(
                FireballTile(
                    tile_pos,
                    tile_image
                )
            )
        
        self.assign_tile_bounding_boxes()        


def main():
    fireball = DifferencedTiles("data/2015_before_after/differenced_images/15_2015-12-19_122158_S_DSC_0829.thumb.jpg")
    print(len(fireball.fireball_tiles), len(fireball.negative_tiles))
    for i, tile in enumerate(fireball.fireball_tiles):
        print(np.sum(tile.image > PIXEL_BRIGHTNESS_THRESHOLD))
        plot_fireball_tile(fireball.fireball_name, i, tile)
    for i, tile in enumerate(fireball.negative_tiles):
        plot_fireball_tile(fireball.fireball_name + "_negative", i, tile)


if __name__ == "__main__":
    main()