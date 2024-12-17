from abc import ABC, abstractmethod
import numpy as np


class Detector(ABC):

    @abstractmethod
    def detect(self, image: np.ndarray) -> tuple:
        """
        input image numpy array and output

        xyxy, conf, cls
        """
        pass