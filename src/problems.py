import numpy as np
from typing import Tuple, Callable


def _euclidean_projection(x: np.ndarray, center: np.ndarray, R: float) -> np.ndarray:
    diff = x - center
    norm = np.linalg.norm(diff)
    if norm > R:
        return center + diff / norm * R
    return x


def make_quadratic(H: np.ndarray, sigma: float, center: np.ndarray, R=np.inf):
    def quadratic(x: np.ndarray) -> float:
        return 0.5 * x @ H @ x.T

    def true_grad(x: np.ndarray) -> np.ndarray:
        return x @ H

    def stoch_grad(x: np.ndarray) -> np.ndarray:
        gaussian_noise = np.random.normal(0, sigma, size=len(x))
        return true_grad(x) + gaussian_noise

    def proj(x: np.ndarray):
        return _euclidean_projection(x, center, R)

    return quadratic, true_grad, stoch_grad, None if R == np.inf else proj


def make_rastrigin(n: int, A: float = 10, sigma: float = 0.0) -> Tuple[Callable, Callable, Callable, float]:
    def rastrigin(x: np.ndarray) -> float:
        assert x.shape[-1] == n
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

    def true_grad(x: np.ndarray) -> np.ndarray:
        assert x.shape[-1] == n
        return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)

    def stoch_grad(x: np.ndarray) -> np.ndarray:
        gaussian_noise = np.random.normal(0, sigma, size=x.shape)
        return true_grad(x) + gaussian_noise

    L = 2 + 40 * np.pi**2

    return rastrigin, true_grad, stoch_grad, L
