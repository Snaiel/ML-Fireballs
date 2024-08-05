from object_detection.dataset.create import create_dataset
from object_detection.dataset.tile_centred import TileCentredFireball


class TileCentredFireballAt400(TileCentredFireball):
    window_dim = (400, 400)

if __name__ == "__main__":
    create_dataset(TileCentredFireballAt400)