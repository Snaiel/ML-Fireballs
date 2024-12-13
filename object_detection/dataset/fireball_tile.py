from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FireballTile:
    position: pd.DataFrame
    points: list[float] = None
    bb_centre: tuple[float] = tuple()
    bb_dim: tuple[int] = tuple()
    image: np.ndarray = None