from dataset.create import create_dataset
from dataset.create.tile_centred import TileCentredFireball

if __name__ == "__main__":
    create_dataset(TileCentredFireball, 1000)