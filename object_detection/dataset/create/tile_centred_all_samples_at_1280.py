from dataset.create import create_dataset
from dataset.tile_centred import TileCentredFireball


class TileCentredFireballAt1280(TileCentredFireball):
    window_dim = (1280, 1280)

if __name__ == "__main__":
    create_dataset(TileCentredFireballAt1280)