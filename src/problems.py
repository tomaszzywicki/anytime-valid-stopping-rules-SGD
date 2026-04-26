import numpy as np
from typing import Tuple, Callable, Optional
from scipy.special import expit
from scipy.stats import ortho_group
from dataclasses import dataclass


@dataclass
class OptimizationProblem:
    fun: Callable[[np.ndarray], float]
    true_grad: Callable[[np.ndarray], np.ndarray]
    stoch_grad: Callable[[np.ndarray], np.ndarray]
    proj: Optional[Callable[[np.ndarray], np.ndarray]] = None
    L: float = 0.0  # Lipschitz
    R: float = np.inf  # Space ball


def _euclidean_projection(x: np.ndarray, center: np.ndarray, R: float) -> np.ndarray:
    diff = x - center
    norm = np.linalg.norm(diff)
    if norm > R:
        return center + diff / norm * R
    return x

def generate_random_hessian(n_dim: int, min_eig: float = 0.1, max_eig: float = 10.0) -> np.ndarray:
    eigenvalues = np.random.uniform(min_eig, max_eig, n_dim)
    Lambda = np.diag(eigenvalues)
    Q = ortho_group.rvs(dim=n_dim)
    return Q @ Lambda @ Q.T


def make_quadratic(H: np.ndarray, sigma: float, center: np.ndarray, R=np.inf):
    def fun(x: np.ndarray) -> float:
        return 0.5 * x @ H @ x.T

    def true_grad(x: np.ndarray) -> np.ndarray:
        return x @ H

    def stoch_grad(x: np.ndarray) -> np.ndarray:
        gaussian_noise = np.random.normal(0, sigma, size=len(x))
        return true_grad(x) + gaussian_noise

    def proj(x: np.ndarray):
        return _euclidean_projection(x, center, R)

    L = np.max(np.linalg.eigvals(H))

    return OptimizationProblem(
        fun=fun, true_grad=true_grad, stoch_grad=stoch_grad, proj=None if R == np.inf else proj, L=L, R=R
    )


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
