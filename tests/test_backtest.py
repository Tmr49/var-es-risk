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


@pytest.mark.parametrize(
    "n,zone", [(0, "green"), (4, "green"), (5, "yellow"), (9, "yellow"), (10, "red")]
)
def test_basel_zones(n, zone):
    assert ve.basel_traffic_light(n) == zone


def test_acerbi_z2_near_zero_for_correct_model():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200_000)
    est = ve.normal_var_es(0, 1, 0.975)
    z2 = ve.acerbi_szekely_z2(x, np.full(x.size, est.var), np.full(x.size, est.es), 0.975)
    assert abs(z2) < 0.03
    under = ve.acerbi_szekely_z2(
        x, np.full(x.size, est.var / 2), np.full(x.size, est.es / 2), 0.975
    )
    assert under < -0.5


def test_rolling_backtest_shapes_and_coverage():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, 3000))
    bt = ve.rolling_backtest(r, lambda w: ve.historical_var_es(w, 0.99), window=500)
    assert len(bt) == 2500
    assert set(bt.columns) == {"var", "es", "realized", "breach"}
    s = ve.summarize(bt, 0.99)
    assert 0.005 < s["breach_rate"] < 0.02
