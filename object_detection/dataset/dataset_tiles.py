from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io
from matplotlib.axes import Axes

from fireball_detection.tiling.included import SQUARE_SIZE
from object_detection.dataset import MIN_BB_DIM_SIZE


@dataclass
class FireballTile:
    position: tuple[int]
    image: np.ndarray
    points: pd.DataFrame = None
    bb_centre: tuple[float] = tuple()
    bb_dim: tuple[int] = tuple()


class DatasetTiles:

    fireball_name: str
    fireball_tiles: list[FireballTile]
    negative_tiles: list[FireballTile]


    def __init__(self, fireball_name: str):
        self.fireball_name = fireball_name
        self.fireball_tiles = []
        self.negative_tiles = []


    def assign_tile_bounding_boxes(self) -> None:
        for tile in self.fireball_tiles:
            self._assign_tile_bounding_box(tile)


    def _assign_tile_bounding_box(self, tile: FireballTile) -> None:
        pp_min_x = tile.points['x'].min()
        pp_max_x = tile.points['x'].max()
        pp_min_y = tile.points['y'].min()
        pp_max_y = tile.points['y'].max()

        points_dim = (pp_max_x - pp_min_x, pp_max_y - pp_min_y)
        padding = 0.05

        bb_min_x = max(pp_min_x - (points_dim[0] * padding), tile.position[0])
        bb_max_x = min(pp_max_x + (points_dim[0] * padding), tile.position[0] + SQUARE_SIZE)

        bb_min_y = max(pp_min_y - (points_dim[1] * padding), tile.position[1])
        bb_max_y = min(pp_max_y + (points_dim[1] * padding), tile.position[1] + SQUARE_SIZE)

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


    def save_tiles(self, images_folder: str, labels_folder: str) -> None:

        for i, tile in enumerate(self.fireball_tiles):
            io.imsave(
                Path(images_folder, f"{self.fireball_name}_{i}.jpg"),
                tile.image,
                check_contrast=False
            )

            label = (0, tile.bb_centre[0], tile.bb_centre[1], tile.bb_dim[0], tile.bb_dim[1])
            with open(Path(labels_folder, f"{self.fireball_name}_{i}.txt"), 'x') as label_file:
                label_file.write(" ".join(str(item) for item in label))
        
        for i, tile in enumerate(self.negative_tiles):
            io.imsave(
                Path(images_folder, f"{self.fireball_name}_negative_{i}.jpg"),
                tile.image,
                check_contrast=False
            )

            open(Path(labels_folder, f"{self.fireball_name}_negative_{i}.txt"), 'x').close()
    

def plot_fireball_tile(fireball_name: str, i: int, tile: FireballTile) -> None:
    image = tile.image

    if image.ndim == 3 and image.shape[2] == 4:
        rgb_image = image[:, :, :3]
        alpha_channel = image[:, :, 3]
        _, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].set_title(f"{fireball_name}_{i} - RGB")
        axs[1].set_title(f"{fireball_name}_{i} - 4th Channel")
        plot_on_axes(axs[0], rgb_image, tile)
        plot_on_axes(axs[1], alpha_channel, tile)
    else:
        _, ax = plt.subplots(1)
        ax.set_title(f"{fireball_name}_{i}.jpg")
        plot_on_axes(ax, image, tile)
    
    plt.tight_layout()
    plt.show()


def plot_on_axes(ax: Axes, image: np.ndarray, tile: FireballTile) -> None:
    if image.ndim == 2:
        ax.imshow(image, cmap='gray')
    elif image.ndim == 3:
        ax.imshow(image)

    if tile.points is not None:
        bb_centre = tile.bb_centre
        bb_dim = tile.bb_dim

        # Adjust points to be relative to the tile's position
        relative_points = np.array(tile.points) - tile.position

        # Calculate bounding box corners (xyxy format)
        bb_min_x = (bb_centre[0] - (bb_dim[0] / 2)) * SQUARE_SIZE
        bb_min_y = (bb_centre[1] - (bb_dim[1] / 2)) * SQUARE_SIZE
        bb_width = bb_dim[0] * SQUARE_SIZE
        bb_height = bb_dim[1] * SQUARE_SIZE

        # Plot the points on the image
        if relative_points.any():
            ax.scatter(relative_points[:, 0], relative_points[:, 1], c='red', label='Points', s=12)

        # Add a rectangle for the bounding box
        rect = patches.Rectangle(
            (bb_min_x, bb_min_y),
            bb_width,
            bb_height,
            linewidth=5,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
    
    ax.axis('off')

    return ax