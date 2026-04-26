import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Dict, List, Literal
from mpl_toolkits.mplot3d import Axes3D  

def plot_convergence(
    results: Dict[str, List[float | np.ndarray]],
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


def plot_2d_trace(
        results: Dict[str, List[float | np.ndarray]],
        fun: Callable,
        plot_type: Literal["3d", "contours"] = "contours",
        title: str = "Optimization Trajectory on Function Landscape",
        x_min: np.ndarray | None = None
    ):
    assert "x" in results, "Results must contain 'x' for 2D trace plotting."
    assert "x_avg" in results, "Results must contain 'x_avg' for 2D trace plotting."

    assert all(isinstance(x, np.ndarray) and x.ndim == 1 for x in results["x"]), "All x values must be 1D numpy arrays."
    assert all(isinstance(x_avg, np.ndarray) and x_avg.ndim == 1 for x_avg in results["x_avg"]), "All x_avg values must be 1D numpy arrays."

    assert all(x.shape[0] == 2 for x in results["x"]), "All x values must be 2D for trace plotting." #type: ignore
    assert all(x.shape[0] == 2 for x in results["x_avg"]), "All x_avg values must be 2D for trace plotting." #type: ignore 


    sgd_trace = np.array(results["x"])
    certified_trace = np.array(results["x_avg"])

    F_avg = np.array(results["F_avg"])
    f = np.array(results["f"])

    x_min_bound, x_max_bound = np.min(sgd_trace[:, 0]), np.max(sgd_trace[:, 0])
    y_min_bound, y_max_bound = np.min(sgd_trace[:, 1]), np.max(sgd_trace[:, 1])

    x_min_cert, x_max_cert = np.min(certified_trace[:, 0]), np.max(certified_trace[:, 0])
    y_min_cert, y_max_cert = np.min(certified_trace[:, 1]), np.max(certified_trace[:, 1])

    x_min_bound = min(x_min_bound, x_min_cert)
    x_max_bound = max(x_max_bound, x_max_cert)
    y_min_bound = min(y_min_bound, y_min_cert)
    y_max_bound = max(y_max_bound, y_max_cert)

    print(f"SGD Trace X range: [{x_min_bound:.2f}, {x_max_bound:.2f}]")
    print(f"SGD Trace Y range: [{y_min_bound:.2f}, {y_max_bound:.2f}]")
    print(f"Certified Trace X range: [{x_min_cert:.2f}, {x_max_cert:.2f}]")
    print(f"Certified Trace Y range: [{y_min_cert:.2f}, {y_max_cert:.2f}]")

    x_range = x_max_bound - x_min_bound
    y_range = y_max_bound - y_min_bound
    padding = 0.1
    x_min_bound -= padding * x_range
    x_max_bound += padding * x_range
    y_min_bound -= padding * y_range
    y_max_bound += padding * y_range

    x_grid, y_grid = np.meshgrid(np.linspace(x_min_bound, x_max_bound, 100), np.linspace(y_min_bound, y_max_bound, 100))
    z_grid = np.array([[fun(np.array([x, y])) for x in np.linspace(x_min_bound, x_max_bound, 100)] for y in np.linspace(y_min_bound, y_max_bound, 100)])

    plt.figure(figsize=(10, 6))

    if plot_type == "3d":
        ax = plt.subplot(111, projection='3d')
        ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis', alpha=0.8) #type: ignore
        ax.plot(sgd_trace[:, 0], sgd_trace[:, 1], f, label='SGD Trace', color='red')
        ax.plot(certified_trace[:, 0], certified_trace[:, 1], [fun(np.array([x, y])) for x, y in certified_trace], label='x_avg Trace', color='blue')
        
        if x_min is not None:
            ax.scatter(x_min[0], x_min[1], fun(x_min), color='black', marker='*', s=100, label=r'$x^*$')
            
        ax.set_xlabel('x[0]')
        ax.set_ylabel('x[1]')
        ax.set_zlabel('F(x)') #type: ignore

    elif plot_type == "contours":
        plt.contourf(x_grid, y_grid, z_grid, levels=50, cmap='viridis')
        plt.colorbar(label='F(x)')
        plt.plot(sgd_trace[:, 0], sgd_trace[:, 1], label='SGD Trace', color='red')
        plt.plot(certified_trace[:, 0], certified_trace[:, 1], label='x_avg Trace', color='blue')
        
        if x_min is not None:
            plt.scatter(x_min[0], x_min[1], color='black', marker='*', s=150, label=r'$x^*$')
            
        plt.xlabel('x[0]')
        plt.ylabel('x[1]')

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()