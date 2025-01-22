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
from utils.constants import GFO_PICKINGS, MIN_POINTS_IN_TILE, SQUARE_SIZE

included_coordinates = retrieve_included_coordinates()


class DifferencedTiles(DatasetTiles):

    def __init__(self, differenced_image_path: str | Path, original_image_path: str | Path) -> None:
        differenced_image_path = Path(differenced_image_path)
        original_image_path = Path(original_image_path)
        
        super().__init__(differenced_image_path.name.split(".")[0])

        differenced_image = io.imread(differenced_image_path)
        original_image = io.imread(original_image_path)
        
        max_value = np.max(differenced_image)
        norm_differenced_image = differenced_image
        if max_value > 0:
            norm_differenced_image = (norm_differenced_image / max_value) * 255

        norm_differenced_image = norm_differenced_image.astype(np.uint8)
        norm_differenced_image = np.expand_dims(norm_differenced_image, axis=-1)

        img_4ch = np.concatenate((original_image, norm_differenced_image), axis=-1)

        weights = np.array([1, 1, 1, 10])
        weighted_image = img_4ch * weights

        img_flat = weighted_image.reshape(-1, 4)
        pca = PCA(n_components=3)
        img_pca = pca.fit_transform(img_flat)
        
        # print(differenced_image_path)
        # print(pca.explained_variance_ratio_)
        # print(pca.components_)
        
        img_3ch = img_pca.reshape(img_4ch.shape[0], img_4ch.shape[1], 3)

        image_result: np.ndarray = (img_3ch - img_3ch.min()) / (img_3ch.max() - img_3ch.min()) * 255
        image_result = image_result.astype(np.uint8)

        points = pd.read_csv(Path(GFO_PICKINGS, self.fireball_name + ".csv"))
        
        for tile_pos in included_coordinates:

            tile_image = get_image_tile(original_image, tile_pos)
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
                        differenced_tile,
                        pd.DataFrame(points_in_tile, columns=["x", "y"])
                    )
                )

            if len(points_in_tile) != 0: continue

            if not satisfies_thresholds(differenced_tile): continue

            self.negative_tiles.append(
                FireballTile(
                    tile_pos,
                    differenced_tile
                )
            )
        
        self.assign_tile_bounding_boxes()


def main():
    fireball = DifferencedTiles(
        "data/2015_before_after/differenced_images/39_2015-12-12_113628_DSC_0128.thumb.jpg",
        "data/2015_before_after/39_2015-12-12_113628_DSC_0128.thumb.jpg"
    )
    print(len(fireball.fireball_tiles), len(fireball.negative_tiles))
    for i, tile in enumerate(fireball.fireball_tiles):
        plot_fireball_tile(fireball.fireball_name, i, tile)
    for i, tile in enumerate(fireball.negative_tiles):
        plot_fireball_tile(fireball.fireball_name + "_negative", i, tile)


if __name__ == "__main__":
    main()