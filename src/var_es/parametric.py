"""Parametric (variance-covariance) VaR/ES: Normal, Student-t and Cornish-Fisher."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate, stats

from ._types import RiskEstimate, as_1d, check_alpha


def normal_var_es(mu: float, sigma: float, alpha: float = 0.99, horizon: int = 1) -> RiskEstimate:
    """Closed form for Normal returns: VaR = -mu + sigma z, ES = -mu + sigma phi(z)/(1-alpha)."""
    check_alpha(alpha)
    mu_h, sig_h = mu * horizon, sigma * np.sqrt(horizon)
    z = stats.norm.ppf(alpha)
    var = -mu_h + sig_h * z
    es = -mu_h + sig_h * stats.norm.pdf(z) / (1 - alpha)
    return RiskEstimate(float(var), float(es), alpha, horizon)


def student_t_var_es(
    mu: float, sigma: float, nu: float, alpha: float = 0.99, horizon: int = 1
) -> RiskEstimate:
    """Student-t returns with standard deviation ``sigma`` (not scale) and nu > 2 dof."""
    check_alpha(alpha)
    if nu <= 2:
        raise ValueError("nu must exceed 2 for finite variance")
    return student_t_scale_var_es(mu, sigma * np.sqrt((nu - 2) / nu), nu, alpha, horizon)


def student_t_scale_var_es(
    loc: float, scale: float, nu: float, alpha: float = 0.99, horizon: int = 1
) -> RiskEstimate:
    """Student-t parameterised by location/scale (as returned by MLE). Needs nu > 1 for ES.
    Works for 1 < nu <= 2 where the variance is infinite but ES is still finite."""
    check_alpha(alpha)
    if nu <= 1:
        raise ValueError("nu must exceed 1 for finite ES")
    loc_h, scale_h = loc * horizon, scale * np.sqrt(horizon)
    q = stats.t.ppf(alpha, nu)
    var = -loc_h + scale_h * q
    es = -loc_h + scale_h * stats.t.pdf(q, nu) / (1 - alpha) * (nu + q**2) / (nu - 1)
    return RiskEstimate(float(var), float(es), alpha, horizon)


def _cf_quantile(u: float, skew: float, exkurt: float) -> float:
    z = stats.norm.ppf(u)
    return (
        z + (z**2 - 1) * skew / 6 + (z**3 - 3 * z) * exkurt / 24 - (2 * z**3 - 5 * z) * skew**2 / 36
    )


def cornish_fisher_var_es(
    mu: float,
    sigma: float,
    skew: float,
    exkurt: float,
    alpha: float = 0.99,
    horizon: int = 1,
) -> RiskEstimate:
    """Cornish-Fisher expansion of the loss quantile. ES integrates the CF quantile
    over the tail, ES = 1/(1-alpha) * int_alpha^1 VaR_u du. ``skew`` and ``exkurt``
    describe the *loss* distribution."""
    check_alpha(alpha)
    mu_h, sig_h = mu * horizon, sigma * np.sqrt(horizon)
    var = -mu_h + sig_h * _cf_quantile(alpha, skew, exkurt)
    tail, _ = integrate.quad(_cf_quantile, alpha, 1, args=(skew, exkurt), limit=200)
    es = -mu_h + sig_h * tail / (1 - alpha)
    return RiskEstimate(float(var), float(max(es, var)), alpha, horizon)


def fit_and_estimate(
    returns: ArrayLike, method: str = "normal", alpha: float = 0.99, horizon: int = 1
) -> RiskEstimate:
    """Fit a parametric model to a return sample and return VaR/ES."""
    r = as_1d(returns)
    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    if method == "normal":
        return normal_var_es(mu, sigma, alpha, horizon)
    if method == "t":
        # Use the MLE scale directly: rescaling by the sample s.d. breaks when the
        # fitted nu is close to 2 (variance -> infinity, implied scale -> 0).
        nu, loc, scale = stats.t.fit(r)
        return student_t_scale_var_es(loc, scale, max(nu, 1.5), alpha, horizon)
    if method == "cornish_fisher":
        losses = -r
        return cornish_fisher_var_es(
            mu, sigma, float(stats.skew(losses)), float(stats.kurtosis(losses)), alpha, horizon
        )
    raise ValueError(f"unknown method {method!r}")
