import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def plot_convergence(
    results: Dict[str, List[float]],
    bound_metric: str,
    report_interval: int,
    f_opt: float = 0.0,
    title: str = "Convergence Analysis",
):
    if bound_metric not in ["F_avg", "G_avg"]:
        raise ValueError("bound_metric musi wynosić 'F_avg' lub 'G_avg'.")

    iterations = (
        np.arange(1, len(results["cert"]) + 1) * report_interval
    )  # Pamiętaj o uwzględnieniu interwału na osi X

    plt.figure(figsize=(10, 6))

    # Rysowanie LUKI suboptymalności
    if bound_metric == "F_avg" and "F_avg" in results:
        f_data = np.array(results["F_avg"]) - f_opt
        f_data = np.maximum(
            f_data, 1e-12
        )  # Zabezpieczenie przed logarytmem z zera/wartości ujemnych przy precyzji zmiennoprzecinkowej
        plt.plot(
            iterations,
            f_data,
            label=r"Suboptimality Gap ($\bar{F}_t - f(x^\star)$)",
            color="blue",
            linewidth=2,
        )

    elif bound_metric == "G_avg" and "G_avg" in results:
        g_data = np.array(results["G_avg"])
        valid_idx = ~np.isnan(g_data)
        if np.any(valid_idx):
            plt.plot(
                iterations[valid_idx],
                g_data[valid_idx],
                label=r"Empirical Stationarity ($\bar{G}_t$)",
                color="green",
                linewidth=2,
            )

    if "f" in results:
        plt.plot(
            iterations, results["f"], label=r"Raw Trajectory $F(x_t)$", color="gray", alpha=0.4, linewidth=1
        )

    if "cert" in results:
        plt.plot(
            iterations,
            results["cert"],
            label=r"Statistical Certificate Bound",
            color="red",
            linestyle="--",
            linewidth=2,
        )

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Iterations (t)", fontsize=12)
    plt.ylabel("Value (Log Scale)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
