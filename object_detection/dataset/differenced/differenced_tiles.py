from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io

from fireball_detection.tiling.included import (SQUARE_SIZE,
                                                retrieve_included_coordinates)
from object_detection.dataset import GFO_PICKINGS, MIN_POINTS_IN_TILE
from object_detection.dataset.fireball_tile import (FireballTile,
                                                    assign_tile_bounding_box,
                                                    plot_fireball_tile)


PIXEL_BRIGHTNESS_THRESHOLD = 10
PIXEL_TOTAL_THRESHOLD = 200


included_coordinates = retrieve_included_coordinates()


class DifferencedTiles:

    fireball_name: str
    fireball_tiles: list[FireballTile]
    negative_tiles: list[FireballTile]

    def __init__(self, differenced_image_path: str) -> None:
        image_path = Path(differenced_image_path)
        self.fireball_name = image_path.name.split(".")[0]

        self.fireball_tiles = []
        self.negative_tiles = []

        fireball_image = io.imread(image_path)

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

        
        # Create bounding boxes for each tile
        for tile in self.fireball_tiles:
            assign_tile_bounding_box(tile)
            

    def save_images(self, folder: str) -> None:
        for i, tile in enumerate(self.fireball_tiles):
            if len(tile.image.shape) == 2:
                tile.image = np.stack([tile.image] * 3, axis=-1)
            io.imsave(
                Path(folder, f"{self.fireball_name}_{i}.jpg"),
                tile.image,
                check_contrast=False
            )
        for i, tile in enumerate(self.negative_tiles):
            if len(tile.image.shape) == 2:
                tile.image = np.stack([tile.image] * 3, axis=-1)
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
    fireball = DifferencedTiles("data/2015_before_after/differenced_images/15_2015-12-19_122158_S_DSC_0829.thumb.jpg")
    print(len(fireball.fireball_tiles), len(fireball.negative_tiles))
    for i, tile in enumerate(fireball.fireball_tiles):
        print(np.sum(tile.image > PIXEL_BRIGHTNESS_THRESHOLD))
        plot_fireball_tile(fireball.fireball_name, i, tile)
    for i, tile in enumerate(fireball.negative_tiles):
        plot_fireball_tile(fireball.fireball_name + "_negative", i, tile)


if __name__ == "__main__":
    main()