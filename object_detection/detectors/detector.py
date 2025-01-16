from abc import ABC, abstractmethod
import numpy as np


class Detector(ABC):

    def __init__(self, path: str, conf: float) -> None:
        self.path = path
        self.conf = conf


    @abstractmethod
    def detect(self, image: np.ndarray) -> tuple:
        """
        input image numpy array and output

        xyxy, conf, cls
        """
        pass