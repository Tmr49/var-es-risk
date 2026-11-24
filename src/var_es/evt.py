"""Extreme value theory: Peaks-over-Threshold with a Generalised Pareto tail."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from ._types import RiskEstimate, as_1d, check_alpha


@dataclass(frozen=True)
class GPDTail:
    threshold: float
    xi: float  # shape; > 0 means a power-law (fat) tail
    beta: float  # scale
    n: int  # sample size
    n_exceed: int

    def var(self, alpha: float) -> float:
        """VaR_a = u + beta/xi * [((n/N_u)(1-a))^(-xi) - 1]  (McNeil, Frey & Embrechts)."""
        p = self.n / self.n_exceed * (1 - alpha)
        if abs(self.xi) < 1e-8:
            return self.threshold - self.beta * np.log(p)
        return self.threshold + self.beta / self.xi * (p ** (-self.xi) - 1)

    def es(self, alpha: float) -> float:
        """ES_a = (VaR_a + beta - xi u) / (1 - xi), finite for xi < 1."""
        if self.xi >= 1:
            raise ValueError("ES is infinite for xi >= 1")
        return (self.var(alpha) + self.beta - self.xi * self.threshold) / (1 - self.xi)


def fit_pot(returns: ArrayLike, threshold_quantile: float = 0.90) -> GPDTail:
    """Fit a GPD to losses above the ``threshold_quantile`` loss quantile."""
    losses = -as_1d(returns)
    u = float(np.quantile(losses, threshold_quantile))
    excess = losses[losses > u] - u
    if excess.size < 20:
        raise ValueError(f"only {excess.size} exceedances; lower the threshold")
    xi, _, beta = stats.genpareto.fit(excess, floc=0)
    return GPDTail(u, float(xi), float(beta), losses.size, int(excess.size))


def evt_pot_var_es(
    returns: ArrayLike, alpha: float = 0.99, threshold_quantile: float = 0.90, horizon: int = 1
) -> RiskEstimate:
    """POT VaR/ES. Multi-day horizons use square-root-of-time scaling."""
    check_alpha(alpha)
    if alpha <= threshold_quantile:
        raise ValueError("alpha must lie above the threshold quantile")
    tail = fit_pot(returns, threshold_quantile)
    s = np.sqrt(horizon)
    return RiskEstimate(float(tail.var(alpha) * s), float(tail.es(alpha) * s), alpha, horizon)
