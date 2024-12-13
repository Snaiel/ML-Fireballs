import random
from pathlib import Path

import pandas as pd
from skimage import io

from fireball_detection.tiling.included import (SQUARE_SIZE,
                                                retrieve_included_coordinates)
from object_detection.dataset import (GFO_JPEGS, GFO_PICKINGS, GFO_THUMB_EXT,
                                      MIN_BB_DIM_SIZE, MIN_POINTS_IN_TILE)
from object_detection.dataset.fireball_tile import (FireballTile,
                                                    assign_tile_bounding_box,
                                                    plot_fireball_tile)


included_coordinates = retrieve_included_coordinates()


class StandaloneTiles:

    fireball_name: str
    fireball_tiles: list[FireballTile]
    negative_tiles: list[FireballTile]

    def __init__(self, fireball_name: str, negative_ratio: int) -> None:
        self.fireball_name = fireball_name

        self.fireball_tiles = []
        self.negative_tiles = []

        fireball_image = io.imread(Path(GFO_JPEGS, fireball_name + GFO_THUMB_EXT))
        points = pd.read_csv(Path(GFO_PICKINGS, self.fireball_name + ".csv"))
        
        # Check if tile contains points
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
        
        # Create bounding boxes for each tile
        for tile in self.fireball_tiles:
            assign_tile_bounding_box(tile)

        
        # Assign images to negative tiles
        if len(self.fireball_tiles) * negative_ratio > len(self.negative_tiles):
            print(f"fireball_tiles * {negative_ratio} > negative_tiles", fireball_name)
            sample_size = len(self.negative_tiles)
        else:
            sample_size = len(self.fireball_tiles) * negative_ratio
        self.negative_tiles = random.sample(self.negative_tiles, sample_size)


    def save_images(self, folder: str) -> None:
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


    def save_labels(self, folder: str) -> None:
        for i, tile in enumerate(self.fireball_tiles):
            label = (0, tile.bb_centre[0], tile.bb_centre[1], tile.bb_dim[0], tile.bb_dim[1])
            with open(Path(folder, f"{self.fireball_name}_{i}.txt"), 'x') as label_file:
                label_file.write(" ".join(str(item) for item in label))
        for i in range(len(self.negative_tiles)):
            open(Path(folder, f"{self.fireball_name}_negative_{i}.txt"), 'x').close()


def main():
    fireball = StandaloneTiles("03_2020-07-20_041559_K_DSC_9695")
    for i, tile in enumerate(fireball.fireball_tiles):
        plot_fireball_tile(fireball.fireball_name, i, tile)


if __name__ == "__main__":
    main()