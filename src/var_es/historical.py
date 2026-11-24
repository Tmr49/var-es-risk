"""Historical simulation and filtered historical simulation (FHS)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._types import RiskEstimate, as_1d, check_alpha, tail_mean


def historical_var_es(returns: ArrayLike, alpha: float = 0.99, horizon: int = 1) -> RiskEstimate:
    """Plain historical simulation. Multi-day horizons use sqrt-of-time scaling."""
    check_alpha(alpha)
    losses = -as_1d(returns)
    var = float(np.quantile(losses, alpha))
    es = tail_mean(losses, var)
    scale = np.sqrt(horizon)
    return RiskEstimate(var * scale, es * scale, alpha, horizon)


def ewma_volatility(returns: ArrayLike, lam: float = 0.94) -> NDArray[np.float64]:
    """RiskMetrics EWMA volatility. Element t is the forecast for day t using data up to t-1;
    the returned array has len(returns) + 1 entries, the last being the next-day forecast."""
    if not 0 < lam < 1:
        raise ValueError("lam must be in (0, 1)")
    r = as_1d(returns)
    var = np.empty(r.size + 1)
    var[0] = r.var()
    for t in range(r.size):
        var[t + 1] = lam * var[t] + (1 - lam) * r[t] ** 2
    return np.sqrt(var)


def filtered_historical_var_es(
    returns: ArrayLike, alpha: float = 0.99, lam: float = 0.94, horizon: int = 1
) -> RiskEstimate:
    """Filtered HS (Hull-White / Barone-Adesi): devolatilise past returns with EWMA,
    then rescale the standardised residuals by tomorrow's volatility forecast."""
    check_alpha(alpha)
    r = as_1d(returns)
    sigma = ewma_volatility(r, lam)
    z = r / sigma[:-1]
    losses = -z * sigma[-1]
    var = float(np.quantile(losses, alpha))
    es = tail_mean(losses, var)
    scale = np.sqrt(horizon)
    return RiskEstimate(var * scale, es * scale, alpha, horizon)
