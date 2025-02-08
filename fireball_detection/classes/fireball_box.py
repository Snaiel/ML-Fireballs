from dataclasses import dataclass


@dataclass(frozen=True)
class FireballBox:

    box: tuple[float, float, float, float]  # xyxy
    conf: float


    def __repr__(self) -> str:
        return f"{self.conf} {' '.join(map(str, self.box))}"
    

    def __str__(self) -> str:
        return f"<{self.conf:.2f} ({', '.join(f'{i:.2f}' for i in self.box)})>"
    

    def to_dict(self) -> dict:
        return {"conf": self.conf, "box": self.box}
