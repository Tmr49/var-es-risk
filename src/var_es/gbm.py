"""Geometric Brownian motion Monte Carlo."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._types import RiskEstimate, as_1d, check_alpha, tail_mean


def fit_gbm(returns: ArrayLike) -> tuple[float, float]:
    """Estimate GBM drift and volatility (per day) from simple returns via log returns:
    sigma = sd(log r), mu = mean(log r) + sigma^2 / 2."""
    log_r = np.log1p(as_1d(returns))
    sigma = float(log_r.std(ddof=1))
    return float(log_r.mean() + 0.5 * sigma**2), sigma


def gbm_mc_var_es(
    returns: ArrayLike,
    alpha: float = 0.99,
    horizon: int = 1,
    n_sims: int = 100_000,
    seed: int | None = None,
) -> RiskEstimate:
    """Simulate S_T / S_0 = exp((mu - sigma^2/2) T + sigma sqrt(T) Z) and read VaR/ES off
    the simulated simple-return losses. This is the light-tailed, constant-volatility
    benchmark the other models are measured against."""
    check_alpha(alpha)
    mu, sigma = fit_gbm(returns)
    z = np.random.default_rng(seed).standard_normal(n_sims)
    growth = np.exp((mu - 0.5 * sigma**2) * horizon + sigma * np.sqrt(horizon) * z)
    losses = -(growth - 1.0)
    var = float(np.quantile(losses, alpha))
    return RiskEstimate(var, tail_mean(losses, var), alpha, horizon)
