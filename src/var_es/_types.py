"""Shared types and helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class RiskEstimate:
    """VaR and ES expressed as positive losses (fraction of portfolio value)."""

    var: float
    es: float
    alpha: float
    horizon: int = 1

    def __post_init__(self) -> None:
        if self.es < self.var - 1e-12:
            raise ValueError(f"ES ({self.es}) must be >= VaR ({self.var})")


def as_1d(x: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("empty input")
    if not np.all(np.isfinite(arr)):
        raise ValueError("input contains NaN or inf")
    return arr


def check_alpha(alpha: float) -> None:
    if not 0.5 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0.5, 1), got {alpha}")


def tail_mean(losses: NDArray[np.float64], var: float) -> float:
    """Mean of losses at or beyond VaR (empirical ES)."""
    tail = losses[losses >= var]
    return float(tail.mean()) if tail.size else float(var)
