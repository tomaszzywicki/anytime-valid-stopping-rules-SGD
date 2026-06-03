from copy import deepcopy
from typing import Dict, Any, Iterable, Iterator
from tqdm import tqdm
import numpy as np
from abc import ABC, abstractmethod
import csv
import os

from .penalty_method.projection import BallProjector
from .problems import make_quadratic, generate_random_hessian, make_rastrigin
from .stopping_criterion import ExactConvexCertificate
from .optimizer import minimize
import time

class Saver(ABC):
    def __init__(self, path: str) -> None:
        self.path = path

    @abstractmethod
    def visit(self, experiment: 'Experiment') -> None:
        pass

class CsvSaver(Saver):
    def visit(self, experiment: 'Experiment') -> None:
        results = experiment.summary()
        
        if not results:
            return

        file_exists = os.path.isfile(self.path)
        
        with open(self.path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            
            if not file_exists:
                writer.writeheader()
                
            writer.writerow(results)

class Experiment(ABC):
    def __init__(self, results_saver: Saver, save_path: str = "") -> None:
        self.results_saver = results_saver
        self.save_path = save_path or self.results_saver.path

    @abstractmethod
    def conduct(self, **kwargs) -> None:
        pass

    @abstractmethod
    def summary(self, **kwargs) -> Dict[str, Any]:
        pass

    def visit_saver(self) -> None:
        self.results_saver.visit(self)


class QuadraticFormLipschitzVSEtaExperiment(Experiment):
    def __init__(
        self, 
        results_saver, 
        n_dim: int, 
        eta_multiplier: float, 
        H: np.ndarray, 
        max_iter: int, 
        num_starts: int = 5,
        eps: float = 1e-1,
        alpha: float = 0.05
    ):
        super().__init__(results_saver)
        self.n_dim = n_dim
        self.eta_multiplier = eta_multiplier
        self.max_iter = max_iter
        self.num_starts = num_starts
        self.eps = eps
        self.alpha= alpha
        
        self.H = H
        self.L = np.max(np.linalg.eigvalsh(self.H))
        self.effective_eta = self.eta_multiplier / self.L
        
        self.iterations_list = []
        self.avg_iters = 0.0
        self.std_iters = 0.0

    def conduct(self, **kwargs) -> None:
        x_min = np.zeros(self.n_dim)
        sigma = 0.5
        R_domain = 10.0
        
        problem = make_quadratic(self.H, sigma, center=x_min, R=R_domain)
        
        for _ in range(self.num_starts):
            x0 = np.random.uniform(-R_domain/2, R_domain/2, self.n_dim)
            criterion = ExactConvexCertificate(x_min, sigma, alpha=self.alpha, eps=self.alpha)
            
            x_avg_t, results = minimize(
                fun=problem.fun,
                grad=problem.stoch_grad,
                proj=problem.proj,
                x0=x0,
                eta0=self.effective_eta,
                gamma=0.5,
                stopping_criterion=criterion,
                max_iter=self.max_iter,
                verbose=False,       
                report_interval=100, 
                true_grad=problem.true_grad,
                save_trace=False 
            )
            
            iters = results.get('iters', len(results.get('cert', [])) * 100)
            self.iterations_list.append(iters)
            
        self.avg_iters = float(np.mean(self.iterations_list))
        self.std_iters = float(np.std(self.iterations_list))

    def summary(self, **kwargs) -> Dict[str, Any]:
        return {
            "eta_multiplier": self.eta_multiplier,
            "effective_eta": self.effective_eta,
            "L_constant": self.L,
            "n_dim": self.n_dim,
            "num_starts": self.num_starts,
            "avg_iters": self.avg_iters,
            "std_iters": self.std_iters
        }
    
class RastriginLipschitzVSEtaExperiment(Experiment):
    def __init__(
        self, 
        results_saver, 
        n_dim: int, 
        eta_multiplier: float, 
        max_iter: int, 
        A: float = 10.0,
        num_starts: int = 5,
        eps: float = 1e-1,
        alpha: float = 0.05
    ):
        super().__init__(results_saver)
        self.n_dim = n_dim
        self.eta_multiplier = eta_multiplier
        self.max_iter = max_iter
        self.A = A
        self.num_starts = num_starts
        self.eps = eps
        self.alpha = alpha
        
        self.L = 2.0 + 4.0 * self.A * np.pi**2
        self.effective_eta = self.eta_multiplier / self.L
        
        self.iterations_list = []
        self.avg_iters = 0.0
        self.std_iters = 0.0

    def conduct(self, **kwargs) -> None:
        x_min = np.zeros(self.n_dim)
        sigma = 0.5
        R_domain = 10.24  
        R_radius = R_domain / 2.0 
        
        fun, true_grad, stoch_grad, _ = make_rastrigin(n=self.n_dim, A=self.A, sigma=sigma)
        projector = BallProjector(R=R_radius, center=x_min)
        
        for _ in range(self.num_starts):
            x0 = np.random.uniform(-R_radius, R_radius, self.n_dim)
            criterion = ExactConvexCertificate(x_min, sigma, alpha=self.alpha, eps=self.eps)
            
            x_avg_t, results = minimize(
                fun=fun,
                grad=stoch_grad,
                proj=projector.project,
                x0=x0,
                eta0=self.effective_eta,
                gamma=0.5,
                stopping_criterion=criterion,
                max_iter=self.max_iter,
                verbose=False,       
                report_interval=100, 
                true_grad=true_grad,
                save_trace=False 
            )
            
            iters = results.get('iters', len(results.get('cert', [])) * 100)
            self.iterations_list.append(iters)
            
        self.avg_iters = float(np.mean(self.iterations_list))
        self.std_iters = float(np.std(self.iterations_list))

    def summary(self, **kwargs) -> Dict[str, Any]:
        return {
            "eta_multiplier": self.eta_multiplier,
            "effective_eta": self.effective_eta,
            "L_constant": self.L,
            "A_param": self.A,
            "n_dim": self.n_dim,
            "num_starts": self.num_starts,
            "avg_iters": self.avg_iters,
            "std_iters": self.std_iters
        }

class ExperimentHandler:
    def __init__(self, experiments: Iterable) -> None:
        self._iterator: Iterator = iter(experiments)

    def run_next(self, verbose: bool = False) -> bool:
        try:
            experiment = next(self._iterator)
            
            if verbose:
                tqdm.write(">>> Starting a new experiment...")
            
            experiment.conduct()
            experiment.visit_saver()
            
            if verbose:
                tqdm.write("<<< Experiment completed and saved.\n")
            
            return True
        
        except StopIteration:
            if verbose:
                tqdm.write("Experiment queue is empty.")
            return False

    def run_all(self) -> None:
        tqdm.write("--- Starting experiment queue ---")
        with tqdm(desc="Progress") as pbar:
            while self.run_next():
                pbar.update(1)
        tqdm.write("--- All experiments completed ---")

