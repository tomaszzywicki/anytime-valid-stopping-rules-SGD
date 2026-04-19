from abc import ABC, abstractmethod

import numpy as np


class Projector(ABC):
    @abstractmethod
    def project(self, x: np.ndarray) -> np.ndarray:
        "Returns x if x is in feasible set or the closest point"
        "in feasible set otherwise."
        pass


class BallProjector(Projector):
    """Projector to ball"""
    def __init__(self, R: float, center: np.ndarray):
        self.R = R
        self.center = center

    def project(self, x: np.ndarray):
        shifted_x = x - self.center
        norm_x = np.linalg.norm(shifted_x)

        if norm_x <= self.R:
            return x

        # projection on a ball
        return self.center + (shifted_x / norm_x) * self.R


class DummyProjector(Projector):
    def project(self, x: np.ndarray) -> np.ndarray:
        return x
