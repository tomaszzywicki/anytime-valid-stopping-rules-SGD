import numpy as np


class PenalizedObjective:
    """Class for evaluating penalized objective and its gradient."""
    def __init__(
        self,
        base_func: callable,
        base_grad: callable,
        penalty_func: callable,
        penalty_grad: callable,
    ):
        self.base_func = base_func
        self.base_grad = base_grad
        self.penalty_func = penalty_func
        self.penalty_grad = penalty_grad

        self.penalty_multiplier = 1

    def evaluate(self, x: np.ndarray) -> float:
        """Return value of penalized function in point x."""
        return self.base_func(x) + self.penalty_multiplier * self.penalty_func(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Return value of the gradient of penalized function in point x."""
        return self.base_grad(x) + self.penalty_multiplier * self.penalty_grad(x)

    def update_penalty_multiplier(self, penalty_multiplier: float) -> None:
        self.penalty_multiplier = penalty_multiplier