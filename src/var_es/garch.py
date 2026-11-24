"""GARCH(1,1) with standardised Student-t innovations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats
from scipy.special import gammaln

from ._types import RiskEstimate, as_1d, check_alpha, tail_mean


@dataclass(frozen=True)
class GarchT:
    mu: float
    omega: float
    alpha: float
    beta: float
    nu: float
    next_var: float  # conditional variance forecast for the day after the sample

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_vol(self) -> float:
        return float(np.sqrt(self.omega / (1 - self.persistence)))


def garch_variance(
    eps: NDArray[np.float64], omega: float, alpha: float, beta: float
) -> NDArray[np.float64]:
    """h_t = omega + alpha eps_{t-1}^2 + beta h_{t-1}. Returns len(eps)+1 values; the last
    is the one-step-ahead forecast. Initialised at the sample variance."""
    h = np.empty(eps.size + 1)
    h[0] = eps.var()
    for t in range(eps.size):
        h[t + 1] = omega + alpha * eps[t] ** 2 + beta * h[t]
    return h


def _std_t_logpdf(z2_over_h: NDArray[np.float64], h: NDArray[np.float64], nu: float):
    """Log density of eps with variance h under a unit-variance Student-t."""
    return (
        gammaln((nu + 1) / 2)
        - gammaln(nu / 2)
        - 0.5 * np.log(np.pi * (nu - 2) * h)
        - (nu + 1) / 2 * np.log1p(z2_over_h / (nu - 2))
    )


def fit_garch_t(returns: ArrayLike) -> GarchT:
    """Maximum likelihood for GARCH(1,1)-t with a constant mean."""
    r = as_1d(returns)
    v = r.var()

    def nll(p: NDArray[np.float64]) -> float:
        mu, omega, a, b, nu = p
        if omega <= 0 or a < 0 or b < 0 or a + b >= 0.9999 or not 2.05 < nu < 200:
            return 1e10
        eps = r - mu
        h = garch_variance(eps, omega, a, b)[:-1]
        return float(-np.sum(_std_t_logpdf(eps**2 / h, h, nu)))

    x0 = np.array([r.mean(), 0.05 * v, 0.08, 0.90, 8.0])
    res = optimize.minimize(
        nll, x0, method="Nelder-Mead", options={"maxiter": 8000, "xatol": 1e-9, "fatol": 1e-9}
    )
    mu, omega, a, b, nu = res.x
    h = garch_variance(r - mu, omega, a, b)
    return GarchT(float(mu), float(omega), float(a), float(b), float(nu), float(h[-1]))


def t_var_es(mu: float, sigma: float, nu: float, alpha: float) -> tuple[float, float]:
    """VaR/ES of a loss -X where X = mu + sigma * T_nu scaled to unit variance."""
    s = sigma * np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(alpha, nu)
    var = -mu + s * q
    es = -mu + s * stats.t.pdf(q, nu) / (1 - alpha) * (nu + q**2) / (nu - 1)
    return float(var), float(es)


def garch_t_var_es(
    returns: ArrayLike,
    alpha: float = 0.99,
    horizon: int = 1,
    model: GarchT | None = None,
    n_sims: int = 50_000,
    seed: int | None = None,
) -> RiskEstimate:
    """One-day VaR/ES in closed form from the conditional t distribution. Longer horizons
    simulate GARCH-t paths, so volatility clustering compounds over the horizon."""
    check_alpha(alpha)
    r = as_1d(returns)
    m = model or fit_garch_t(r)
    if model is not None:  # re-filter so next_var reflects the latest data
        h = garch_variance(r - m.mu, m.omega, m.alpha, m.beta)
        m = GarchT(m.mu, m.omega, m.alpha, m.beta, m.nu, float(h[-1]))
    if horizon == 1:
        var, es = t_var_es(m.mu, np.sqrt(m.next_var), m.nu, alpha)
        return RiskEstimate(var, es, alpha, 1)

    rng = np.random.default_rng(seed)
    scale = np.sqrt((m.nu - 2) / m.nu)
    h = np.full(n_sims, m.next_var)
    total = np.zeros(n_sims)
    for _ in range(horizon):
        eps = rng.standard_t(m.nu, n_sims) * scale * np.sqrt(h)
        total += m.mu + eps
        h = m.omega + m.alpha * eps**2 + m.beta * h
    losses = -total
    var = float(np.quantile(losses, alpha))
    return RiskEstimate(var, tail_mean(losses, var), alpha, horizon)
