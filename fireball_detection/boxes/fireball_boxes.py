from fireball_detection import Tile, FireballBox, IMAGE_DIMENSIONS


def compute_absolute_coordinates(tile: Tile, box: tuple):
    x0 = float(tile.position[0] + box[0])
    y0 = float(tile.position[1] + box[1])
    x1 = float(tile.position[0] + box[2])
    y1 = float(tile.position[1] + box[3])
    return x0, y0, x1, y1


def compute_normalised_coordinates(tile: Tile, box: tuple):
    x0_offset = (box[2] - box[0]) * 0.1
    y0_offset = (box[3] - box[1]) * 0.1
    x0 = float((tile.position[0] + box[0] - x0_offset) / IMAGE_DIMENSIONS[0])
    y0 = float((tile.position[1] + box[1] - y0_offset) / IMAGE_DIMENSIONS[1])
    x1 = float((tile.position[0] + box[2] + x0_offset) / IMAGE_DIMENSIONS[0])
    y1 = float((tile.position[1] + box[3] + y0_offset) / IMAGE_DIMENSIONS[1])
    return x0, y0, x1, y1


def create_fireball_box(tile: Tile, box: list, conf: float, compute_coordinates):
    x0, y0, x1, y1 = compute_coordinates(tile, box)
    return FireballBox((x0, y0, x1, y1), conf)


def get_absolute_fireball_boxes(tiles: list[Tile]) -> list[FireballBox]:
    return [
        create_fireball_box(tile, box, conf, compute_absolute_coordinates)
        for tile in tiles
        for box, conf in zip(tile.boxes, tile.confidences)
    ]


def get_normalised_fireball_boxes(tiles: list[Tile]) -> list[FireballBox]:
    return [
        create_fireball_box(tile, box, conf.cpu(), compute_normalised_coordinates)
        for tile in tiles
        for box, conf in zip(tile.boxes, tile.confidences)
    ]
