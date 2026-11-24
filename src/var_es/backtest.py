"""Rolling-window VaR backtesting: Kupiec POF and Christoffersen tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from scipy.special import xlogy

from ._types import RiskEstimate


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float

    def reject(self, level: float = 0.05) -> bool:
        return self.p_value < level


def _breaches(x: ArrayLike) -> NDArray[np.int_]:
    b = np.asarray(x).astype(int).ravel()
    if not np.isin(b, [0, 1]).all():
        raise ValueError("breaches must be 0/1")
    return b


def _bern_ll(n0: float, n1: float, p: float) -> float:
    return float(xlogy(n0, 1 - p) + xlogy(n1, p))


def kupiec_pof(breaches: ArrayLike, alpha: float) -> TestResult:
    """Proportion-of-failures LR test: is the breach rate equal to 1 - alpha?  LR ~ chi2(1)."""
    b = _breaches(breaches)
    n, x = b.size, int(b.sum())
    p = 1 - alpha
    lr = -2 * (_bern_ll(n - x, x, p) - _bern_ll(n - x, x, x / n))
    return TestResult(lr, float(stats.chi2.sf(lr, 1)))


def christoffersen_independence(breaches: ArrayLike) -> TestResult:
    """Markov test that breaches don't cluster (today's breach independent of yesterday's)."""
    b = _breaches(breaches)
    prev, curr = b[:-1], b[1:]
    n00 = np.sum((prev == 0) & (curr == 0))
    n01 = np.sum((prev == 0) & (curr == 1))
    n10 = np.sum((prev == 1) & (curr == 0))
    n11 = np.sum((prev == 1) & (curr == 1))
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll_null = _bern_ll(n00 + n10, n01 + n11, pi)
    ll_alt = _bern_ll(n00, n01, pi0) + _bern_ll(n10, n11, pi1)
    lr = max(-2 * (ll_null - ll_alt), 0.0)
    return TestResult(lr, float(stats.chi2.sf(lr, 1)))


def christoffersen_cc(breaches: ArrayLike, alpha: float) -> TestResult:
    """Conditional coverage: Kupiec + independence, LR ~ chi2(2)."""
    lr = kupiec_pof(breaches, alpha).statistic + christoffersen_independence(breaches).statistic
    return TestResult(lr, float(stats.chi2.sf(lr, 2)))


def rolling_backtest(
    returns: pd.Series,
    model: Callable[[NDArray[np.float64]], RiskEstimate],
    window: int = 500,
) -> pd.DataFrame:
    """Walk forward: fit on the trailing ``window`` days, forecast VaR/ES for the next day."""
    r = returns.to_numpy(dtype=float)
    rows = []
    for t in range(window, r.size):
        est = model(r[t - window : t])
        rows.append((est.var, est.es, r[t]))
    out = pd.DataFrame(rows, columns=["var", "es", "realized"], index=returns.index[window:])
    out["breach"] = (out["realized"] < -out["var"]).astype(int)
    return out


def summarize(bt: pd.DataFrame, alpha: float) -> dict[str, float | str]:
    b = bt["breach"].to_numpy()
    tail = bt.loc[bt["breach"] == 1]
    return {
        "n_obs": int(b.size),
        "breaches": int(b.sum()),
        "expected": round(b.size * (1 - alpha), 1),
        "breach_rate": float(b.mean()),
        "kupiec_p": kupiec_pof(b, alpha).p_value,
        "christoffersen_ind_p": christoffersen_independence(b).p_value,
        "cc_p": christoffersen_cc(b, alpha).p_value,
        # ES check: on breach days, realised loss vs forecast ES (ratio ~1 if ES is right)
        "loss_to_es_on_breach": float((-tail["realized"] / tail["es"]).mean())
        if len(tail)
        else float("nan"),
    }
