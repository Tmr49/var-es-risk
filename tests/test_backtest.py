import numpy as np
import pandas as pd
import pytest

import var_es as ve


def test_kupiec_correct_rate_not_rejected():
    b = np.zeros(1000, dtype=int)
    b[::100] = 1  # exactly 1%
    res = ve.kupiec_pof(b, 0.99)
    assert res.statistic == pytest.approx(0.0, abs=1e-9)
    assert not res.reject()


def test_kupiec_too_many_breaches_rejected():
    b = np.zeros(1000, dtype=int)
    b[::25] = 1  # 4%
    assert ve.kupiec_pof(b, 0.99).reject()


def test_kupiec_zero_breaches_finite():
    res = ve.kupiec_pof(np.zeros(500, dtype=int), 0.99)
    assert np.isfinite(res.statistic)


def test_christoffersen_detects_clustering():
    b = np.zeros(1000, dtype=int)
    b[500:510] = 1  # same count as 1%, all in a row
    assert ve.christoffersen_independence(b).reject()
    spread = np.zeros(1000, dtype=int)
    spread[::100] = 1
    assert not ve.christoffersen_independence(spread).reject()


def test_rolling_backtest_shapes_and_coverage():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, 3000))
    bt = ve.rolling_backtest(r, lambda w: ve.historical_var_es(w, 0.99), window=500)
    assert len(bt) == 2500
    assert set(bt.columns) == {"var", "es", "realized", "breach"}
    s = ve.summarize(bt, 0.99)
    assert 0.005 < s["breach_rate"] < 0.02
