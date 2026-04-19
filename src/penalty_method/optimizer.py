import numpy as np

from .penalized_objective import PenalizedObjective

from ..optimizer import minimize
from ..stopping_criterion import StoppingCriterion


def sequential_penalty_optimizer(
    objective: PenalizedObjective,
    projector: callable, 
    stopping_criterion: StoppingCriterion,
    x0: np.ndarray,
    initial_penalty: float = 1.0,
    penalty_growth_factor: float = 5.0,
    max_outer_iter: int = 5,
    max_inner_iter: int = 1_000_000,
    eta0: float = 1.0,
    gamma: float = 0.5,
    verbose: bool = True,
    report_interval: int = 100,
): 
    """Performs optimization using sequential penalty method and any time valid stopping criterion,"""
    current_x = x0.copy()
    objective.update_penalty_multiplier(initial_penalty)
    all_results = []

    for outer_t in range(1, max_outer_iter + 1):
        current_penalty_multiplier = objective.penalty_multiplier
        print(
            f"Outer iteration {outer_t} | Penalty multiplier: {current_penalty_multiplier}"
        )

        x_avg, inner_results = minimize(
            fun=objective.evaluate,
            grad=objective.gradient,
            proj=projector,
            x0=current_x,
            eta0=eta0,
            gamma=gamma,
            stopping_criterion=stopping_criterion,
            max_iter=max_inner_iter,
            verbose=verbose,
            report_interval=report_interval,
        )
        stopping_criterion.reset()

        all_results.append(inner_results)
        current_penalty_value = objective.penalty_func(x_avg)
        if current_penalty_value < 1e-5:
            print("Feasible x found")
            return x_avg, all_results

        objective.update_penalty_multiplier(
            current_penalty_multiplier * penalty_growth_factor
        )
        current_x = x_avg

    return current_x, all_results
