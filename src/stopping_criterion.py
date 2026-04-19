import numpy as np
from abc import ABC, abstractmethod


class StoppingCriterion(ABC):
    @property
    @abstractmethod
    def requires_bounded_domain(self) -> bool:
        pass

    @abstractmethod
    def check(self, t, S_t, eta_t, g_t, x_t):
        pass


class ExactConvexCertificate(StoppingCriterion):
    @property
    def requires_bounded_domain(self) -> bool:
        return False

    def __init__(self, x_min: np.ndarray, sigma: float, alpha: float, eps: float):
        self.x_min = x_min
        self.sigma = sigma
        self.alpha = alpha
        self.eps = eps

        self.Z1 = None
        self.Sigma_t = 0.0
        self.tel_sum = 0.0

    def check(self, t, S_t, eta_t, g_t, x_t):
        if t == 1:
            self.Z1 = np.linalg.norm(self.x_min - x_t) ** 2

        Z_t = np.linalg.norm(self.x_min - x_t) ** 2
        sigma_t = self.sigma * eta_t**2 * Z_t
        self.Sigma_t += sigma_t
        Sigma_eff_t = np.maximum(self.Sigma_t, 2 * (np.log(2 / self.alpha) + 1))

        self.tel_sum += eta_t**2 * np.linalg.norm(g_t) ** 2

        U_t = (
            1
            / (2 * S_t)
            * (
                7 * np.sqrt(Sigma_eff_t * (np.log(2 / self.alpha) + np.log(np.log(np.e + Sigma_eff_t))))
                + self.Z1
                + self.tel_sum
            )
        )

        is_stop = U_t < self.eps

        return is_stop, U_t


class ObservableConvexCertificate(StoppingCriterion):
    @property
    def requires_bounded_domain(self) -> bool:
        return True

    def __init__(self, R: float, sigma: float, alpha: float, eps):
        self.R = R
        self.sigma = sigma
        self.alpha = alpha
        self.eps = eps

        self.Sigma_t = 0.0
        self.tel_sum = 0.0

    def check(self, t, S_t, eta_t, g_t, x_t):
        sigma_t = self.sigma * eta_t**2 * self.R**2
        self.Sigma_t += sigma_t
        Sigma_eff_t = np.maximum(self.Sigma_t, 2 * (np.log(2 / self.alpha) + 1))

        self.tel_sum += eta_t**2 * np.linalg.norm(g_t) ** 2

        U_t = (
            1
            / (2 * S_t)
            * (
                7 * np.sqrt(Sigma_eff_t * (np.log(2 / self.alpha) + np.log(np.log(np.e + Sigma_eff_t))))
                + self.R**2
                + self.tel_sum
            )
        )

        is_stop = U_t < self.eps

        return is_stop, U_t
    
    def reset(self):
        self.Sigma_t = 0.0
        self.tel_sum = 0.0



class ExactNonconvexCertificate(StoppingCriterion):
    @property
    def requires_bounded_domain(self) -> bool:
        return False

    def __init__(self, fun: callable, sigma: float, alpha: float, eps: float, L: float, true_grad: callable):
        self.fun = fun
        self.sigma = sigma
        self.alpha = alpha
        self.eps = eps
        self.L = L
        self.true_grad = true_grad

        self.f1 = None
        self.Gamma_t = 0.0
        self.tel_sum = 0.0

    def check(self, t, S_t, eta_t, g_t, x_t):
        if t == 1:
            self.f1 = self.fun(x_t)

        v_t = self.sigma * eta_t**2 * np.linalg.norm(self.true_grad(x_t)) ** 2
        self.Gamma_t += v_t
        Gamma_eff_t = np.maximum(self.Gamma_t, 2 * (np.log(2 / self.alpha) + 1))

        self.tel_sum += eta_t**2 * np.linalg.norm(g_t) ** 2

        W_t = (
            1
            / S_t
            * (
                4 * np.sqrt(Gamma_eff_t * (np.log(2 / self.alpha) + np.log(np.log(np.e + Gamma_eff_t))))
                + self.f1
                + self.L / 2 * self.tel_sum
            )
        )

        is_stop = W_t < self.eps

        return is_stop, W_t


class ObservableNonconvexCertificate(StoppingCriterion):
    @property
    def requires_bounded_domain(self) -> bool:
        return False

    def __init__(self, fun: callable, sigma: float, alpha: float, eps: float, L: float, G: float):
        self.fun = fun
        self.sigma = sigma
        self.alpha = alpha
        self.eps = eps
        self.L = L
        self.G = G

        self.f1 = None
        self.Gamma_t = 0.0
        self.tel_sum = 0.0

    def check(self, t, S_t, eta_t, g_t, x_t):
        if t == 1:
            self.f1 = self.fun(x_t)

        v_t = self.sigma * eta_t**2 * self.G**2
        self.Gamma_t += v_t
        Gamma_eff_t = np.maximum(self.Gamma_t, 2 * (np.log(2 / self.alpha) + 1))

        self.tel_sum += eta_t**2 * np.linalg.norm(g_t) ** 2

        W_t = (
            1
            / S_t
            * (
                4 * np.sqrt(Gamma_eff_t * (np.log(2 / self.alpha) + np.log(np.log(np.e + Gamma_eff_t))))
                + self.f1
                + self.L / 2 * self.tel_sum
            )
        )

        is_stop = W_t < self.eps

        return is_stop, W_t
