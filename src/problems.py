import numpy as np
from typing import Tuple, Callable
from scipy.special import expit


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


# class LogisticRegression:

#     def __init__(self, X: np.ndarray, y: np.ndarray):
#         self.X = X
#         self.y = y
#         self.coef_ = np.random.normal(0, 0.001, size=X.shape[1])

#     def loss(self):
#         p = expit(self.X.T @ self.coef_)
#         return -np.mean(self.y * np.log(p) + (1 - self.y) * np.log(1 - p))

#     def loss_grad(self):
#         p = expit(self.X.T @ self.coef_)
#         return np.mean(self.X.T @ (p - self.y))


def logistic_regression(X: np.ndarray, y: np.ndarray, R: float, batch_size: int = 16):
    def loss(beta: np.ndarray):
        p = expit(X @ beta)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def true_grad(beta: np.ndarray):
        # Deterministic (full batch)
        p = expit(X @ beta)
        return (X.T @ (p - y)) / X.shape[0]

    def stoch_grad(beta: np.ndarray):
        # Stochastic gradient (Mini-Batch)
        idx = np.random.choice(X.shape[0], size=batch_size, replace=False)
        X_batch = X[idx]
        y_batch = y[idx]
        p = expit(X_batch @ beta)
        return (X_batch.T @ (p - y_batch)) / batch_size

    def proj(beta: np.ndarray):
        center = np.zeros_like(beta)
        return _euclidean_projection(beta, center, R)

    return loss, true_grad, stoch_grad, proj
