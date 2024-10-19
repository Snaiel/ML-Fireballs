import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from fireball_detection.tiling.included import (SQUARE_SIZE,
                                         retrieve_included_coordinates)
from object_detection.dataset import GFO_PICKINGS
from object_detection.dataset.split_tiles import MIN_POINTS_IN_TILE


included_coordinates = retrieve_included_coordinates()

fireball_tiles_list = []
for fireball_file in tqdm(os.listdir(GFO_PICKINGS), desc="processing point pickings"):
    fireball_name = fireball_file.split(".")[0]

    pp = pd.read_csv(Path(GFO_PICKINGS, fireball_name + ".csv"))

    fireball_tiles = 0
    
    for tile_pos in included_coordinates:
        points_in_tile = 0

        for point in pp.itertuples(False, None):
            if (
                tile_pos[0] <= point[0] and point[0] < tile_pos[0] + SQUARE_SIZE and \
                tile_pos[1] <= point[1] and point[1] < tile_pos[1] + SQUARE_SIZE
            ):
                points_in_tile += 1
        
        if points_in_tile >= MIN_POINTS_IN_TILE:
            fireball_tiles += 1
    
    fireball_tiles_list.append(fireball_tiles)

median_tiles = np.median(fireball_tiles_list)
average_tiles = np.mean(fireball_tiles_list)

print(f"{'total tiles:':<30} {len(included_coordinates)}")
print(f"{'median fireball tiles:':<30} {median_tiles:.2f}")
print(f"{'average fireball tiles:':<30} {average_tiles:.2f}")
print(f"{'median ratio:':<30} {int(median_tiles)}:{len(included_coordinates) - int(median_tiles)}")
print(f"{'average ratio:':<30} {int(average_tiles)}:{len(included_coordinates) - int(average_tiles)}")


plt.hist(fireball_tiles_list, bins=range(61), color="blue", alpha=0.7)
plt.title('Histogram of Fireball Tiles Count')
plt.xlabel('Number of Tiles')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
