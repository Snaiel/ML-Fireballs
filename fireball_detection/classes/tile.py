from dataclasses import dataclass, field
from numpy import ndarray


@dataclass
class Tile:

    position: tuple[float, float]
    image: ndarray
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)


    def get_detections(self) -> list[dict]:
        detections = []
        for b, c in zip(self.boxes, self.confidences):
            detections.append({"box": b, "confidence": c})
        return detections


    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "detections": self.get_detections()
        }
