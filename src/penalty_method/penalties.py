import numpy as np


def make_classic_penalty(
    inequality_constraints: list[tuple[callable, callable]],
    equality_constraints: list[tuple[callable, callable]],
) -> tuple[callable, callable]:
    """Creates a classical penalty for sequential penalty method

    Args:
        inequality_constraints (list[tuple[callable, callable]]): A list of tuples (func, gradient) of inequality constraints.
        equality_constraints (list[tuple[callable, callable]]): A list of tuples (func, gradient) of equality constraints.

    Returns:
        tuple[callable, callable]: Penalty and gradient callables.
    """
    def penalty(x: np.ndarray) -> float:
        total_penalty = 0
        for ineq, _ in inequality_constraints:
            total_penalty += max(0, ineq(x)) ** 2
        for eq, _ in equality_constraints:
            total_penalty += eq(x) ** 2
        return total_penalty

    def grad(x: np.ndarray) -> np.ndarray:
        total_grad = np.zeros_like(x, dtype=np.float64)

        for ineq, ineq_grad in inequality_constraints:
            val = ineq(x)
            if val > 0:
                total_grad += 2 * ineq(x) * ineq_grad(x)

        for eq, eq_grad in equality_constraints:
            total_grad += 2 * eq(x) * eq_grad(x)

        return total_grad

    return penalty, grad
