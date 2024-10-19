from dataclasses import dataclass, field
from numpy import ndarray


SQUARE_SIZE = 400
IMAGE_DIMENSIONS = (7360, 4912)


@dataclass
class Tile:
    position: tuple[float, float]
    image: ndarray
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)


@dataclass
class FireballBox:
    box: tuple[float, float, float, float]  # xyxy
    conf: float

    def __repr__(self) -> str:
        return f"{self.conf} {' '.join(map(str, self.box))}"
    
    def __str__(self) -> str:
        return f"<{self.conf:.2f} ({', '.join(f'{i:.2f}' for i in self.box)})>"
