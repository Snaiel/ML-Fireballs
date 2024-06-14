from dataset.create.tile_centred import TileCentredFireball
from dataset.utils import create_dataset

if __name__ == "__main__":
    create_dataset(TileCentredFireball, 1000)