"""Historical simulation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._types import RiskEstimate, as_1d, check_alpha, tail_mean


def historical_var_es(returns: ArrayLike, alpha: float = 0.99, horizon: int = 1) -> RiskEstimate:
    """Plain historical simulation: the empirical loss quantile and the mean loss beyond it.
    Multi-day horizons use square-root-of-time scaling."""
    check_alpha(alpha)
    losses = -as_1d(returns)
    var = float(np.quantile(losses, alpha))
    es = tail_mean(losses, var)
    scale = np.sqrt(horizon)
    return RiskEstimate(var * scale, es * scale, alpha, horizon)
