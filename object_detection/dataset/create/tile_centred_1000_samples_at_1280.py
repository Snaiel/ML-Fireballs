from dataset.create import create_dataset
from dataset.create.tile_centred_all_samples_at_1280 import TileCentredFireballAt1280

if __name__ == "__main__":
    create_dataset(TileCentredFireballAt1280, 1000)