from multiprocessing import Value
import numpy as np
from collections import defaultdict

from src.stopping_criterion import StoppingCriterion


def minimize(
    fun: callable,
    grad: callable,
    proj: callable,
    x0: np.ndarray,
    eta0: float,
    gamma: float,
    stopping_criterion: StoppingCriterion,
    max_iter: int = 10_000_000,
    verbose: bool = True,
    report_interval: int = 100,
    true_grad: callable = None
):
    if stopping_criterion.requires_bounded_domain and proj is None:
        raise ValueError("This stopping criterion requires projection function (proj).")

    results = defaultdict(list)

    x = x0

    S_t = 0
    V_t = 0
    eta_x = np.zeros_like(x0, dtype=np.float64)
    eta_f = 0
    eta_g_norm = 0

    for t in range(1, max_iter + 1):
        eta_t = eta0 * 1 / t**gamma

        f = fun(x)
        g = grad(x)

        # Variable update
        S_t += eta_t
        V_t += eta_t**2

        eta_x += eta_t * x
        eta_f += eta_t * f
        if true_grad is not None:
            g_true = true_grad(x)
            eta_g_norm += eta_t * np.linalg.norm(g_true) ** 2
            G_avg = eta_g_norm / S_t
        else:
            G_avg = float("nan")

        # Check stopping criterion
        is_stop, cert_val = stopping_criterion.check(t, S_t, eta_t, g, x)

        # Values for logging and plots
        x_avg_t = eta_x / S_t
        F_avg = eta_f / S_t

        if verbose and t % report_interval == 0:
            print(
                f"[Iter {t:6d}]  "
                f"F(x)={f:8.5f} | "
                f"F_avg(x)={F_avg:8.5f} | "
                f"G_avg={G_avg:8.5} | "
                f"Cert={cert_val:8.5f}"
            )
            results["f"].append(f)
            results["F_avg"].append(F_avg)
            results["G_avg"].append(G_avg)
            results["cert"].append(cert_val)

        if is_stop:
            print(f"Ending at t={t}")
            break

        # SGD step
        x = x - eta_t * g
        if proj is not None:
            x = proj(x)

    return x_avg_t, results
