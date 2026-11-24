"""Monte Carlo VaR/ES: multivariate Normal / Student-t portfolios and GARCH(1,1) paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from ._types import RiskEstimate, as_1d, check_alpha, tail_mean


def _estimate(pnl: NDArray[np.float64], alpha: float, horizon: int) -> RiskEstimate:
    losses = -pnl
    var = float(np.quantile(losses, alpha))
    return RiskEstimate(var, tail_mean(losses, var), alpha, horizon)


def mc_portfolio_var_es(
    asset_returns: ArrayLike,
    weights: ArrayLike,
    alpha: float = 0.99,
    horizon: int = 1,
    dist: str = "normal",
    nu: float = 5.0,
    n_sims: int = 100_000,
    seed: int | None = None,
) -> RiskEstimate:
    """Simulate horizon-day portfolio returns from a fitted multivariate distribution.

    ``asset_returns`` is (T, N) of daily simple returns; ``weights`` is length N.
    Daily draws are i.i.d.; horizon returns are the sum of daily draws.
    For ``dist="t"`` the covariance is matched by scaling with (nu-2)/nu.
    """
    check_alpha(alpha)
    x = np.asarray(asset_returns, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    w = np.asarray(weights, dtype=float)
    if w.shape != (x.shape[1],):
        raise ValueError("weights must have one entry per asset")
    rng = np.random.default_rng(seed)
    mu, cov = x.mean(axis=0), np.cov(x, rowvar=False).reshape(x.shape[1], x.shape[1])
    chol = np.linalg.cholesky(cov + 1e-14 * np.eye(len(w)))

    z = rng.standard_normal((n_sims, horizon, len(w)))
    if dist == "t":
        if nu <= 2:
            raise ValueError("nu must exceed 2")
        g = rng.chisquare(nu, size=(n_sims, horizon, 1)) / nu
        z = z / np.sqrt(g) * np.sqrt((nu - 2) / nu)
    elif dist != "normal":
        raise ValueError(f"unknown dist {dist!r}")
    daily = mu + z @ chol.T
    pnl = (daily @ w).sum(axis=1)
    return _estimate(pnl, alpha, horizon)


@dataclass(frozen=True)
class Garch11:
    omega: float
    alpha: float
    beta: float
    mu: float
    last_var: float  # conditional variance forecast for the next day

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_vol(self) -> float:
        return float(np.sqrt(self.omega / (1 - self.persistence)))


def _garch_filter(params: NDArray[np.float64], e: NDArray[np.float64]) -> NDArray[np.float64]:
    omega, a, b = params
    h = np.empty(e.size + 1)
    h[0] = e.var()
    for t in range(e.size):
        h[t + 1] = omega + a * e[t] ** 2 + b * h[t]
    return h


def fit_garch11(returns: ArrayLike) -> Garch11:
    """Gaussian quasi-MLE for GARCH(1,1) with constant mean."""
    r = as_1d(returns)
    mu = float(r.mean())
    e = r - mu
    v = e.var()

    def nll(p: NDArray[np.float64]) -> float:
        if p[0] <= 0 or p[1] < 0 or p[2] < 0 or p[1] + p[2] >= 0.9999:
            return 1e10
        h = _garch_filter(p, e)[:-1]
        return float(0.5 * np.sum(np.log(h) + e**2 / h))

    x0 = np.array([v * 0.05, 0.08, 0.90])
    res = optimize.minimize(
        nll, x0, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-8, "maxiter": 5000}
    )
    omega, a, b = res.x
    h = _garch_filter(res.x, e)
    return Garch11(float(omega), float(a), float(b), mu, float(h[-1]))


def garch_mc_var_es(
    returns: ArrayLike,
    alpha: float = 0.99,
    horizon: int = 10,
    n_sims: int = 50_000,
    seed: int | None = None,
    model: Garch11 | None = None,
) -> RiskEstimate:
    """Simulate GARCH(1,1) paths with bootstrapped standardised residuals (FHS-style),
    so horizon risk reflects volatility clustering instead of sqrt-of-time."""
    check_alpha(alpha)
    r = as_1d(returns)
    m = model or fit_garch11(r)
    e = r - m.mu
    std_resid = e / np.sqrt(_garch_filter(np.array([m.omega, m.alpha, m.beta]), e)[:-1])
    rng = np.random.default_rng(seed)
    h = np.full(n_sims, m.last_var)
    total = np.zeros(n_sims)
    for _ in range(horizon):
        eps = rng.choice(std_resid, size=n_sims) * np.sqrt(h)
        total += m.mu + eps
        h = m.omega + m.alpha * eps**2 + m.beta * h
    return _estimate(total, alpha, horizon)
