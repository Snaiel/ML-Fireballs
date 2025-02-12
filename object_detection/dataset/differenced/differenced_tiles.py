import random
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io
from sklearn.decomposition import PCA

from detection_pipeline.tile_preprocessing import satisfies_thresholds
from fireball_detection.tiling import (get_image_tile,
                                       retrieve_included_coordinates)
from object_detection.dataset.dataset_tiles import (DatasetTiles, FireballTile,
                                                    plot_fireball_tile)
from utils.constants import MIN_POINTS_IN_TILE, SQUARE_SIZE
from utils.paths import GFO_PICKINGS


included_coordinates = retrieve_included_coordinates()


class DifferencedTiles(DatasetTiles):

    def __init__(self, differenced_image_path: str | Path, original_image_path: str | Path) -> None:
        differenced_image_path = Path(differenced_image_path)
        original_image_path = Path(original_image_path)
        
        super().__init__(differenced_image_path.name.split(".")[0])

        differenced_image = io.imread(differenced_image_path)

        points = pd.read_csv(Path(GFO_PICKINGS, self.fireball_name + ".csv"))
        
        for tile_pos in included_coordinates:

            tile_image = get_image_tile(differenced_image, tile_pos)

            max_value = np.max(tile_image)
            norm_tile = tile_image.copy()
            if max_value > 0:
                norm_tile = (tile_image / max_value) * 255
            norm_tile = norm_tile.astype(np.uint8)

            differenced_tile = get_image_tile(differenced_image, tile_pos)

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
                        norm_tile,
                        pd.DataFrame(points_in_tile, columns=["x", "y"])
                    )
                )

            if len(points_in_tile) != 0: continue

            if not satisfies_thresholds(differenced_tile): continue

            self.negative_tiles.append(
                FireballTile(
                    tile_pos,
                    norm_tile
                )
            )
        
        self.assign_tile_bounding_boxes()


def main():
    fireball = DifferencedTiles(
        "data/2015_before_after/differenced_images/07_2015-03-18_140529_DSC_0351.thumb.jpg",
        "data/2015_before_after/07_2015-03-18_140529_DSC_0351.thumb.jpg"
    )
    print(len(fireball.fireball_tiles), len(fireball.negative_tiles))
    for i, tile in enumerate(fireball.fireball_tiles):
        plot_fireball_tile(fireball.fireball_name, i, tile)
    for i, tile in enumerate(fireball.negative_tiles):
        plot_fireball_tile(fireball.fireball_name + "_negative", i, tile)


if __name__ == "__main__":
    main()