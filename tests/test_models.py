import numpy as np
import pytest
from scipy import stats

import var_es as ve

RNG = np.random.default_rng(0)
NORMAL = RNG.normal(0.0005, 0.01, 400_000)


def test_normal_closed_form_99():
    est = ve.normal_var_es(0.0, 1.0, 0.99)
    assert est.var == pytest.approx(2.326348, abs=1e-6)
    assert est.es == pytest.approx(2.665214, abs=1e-6)


def test_normal_horizon_scaling():
    one = ve.normal_var_es(0.0, 0.01, 0.99, horizon=1)
    ten = ve.normal_var_es(0.0, 0.01, 0.99, horizon=10)
    assert ten.var == pytest.approx(one.var * np.sqrt(10))


def test_t_converges_to_normal_for_large_nu():
    n = ve.normal_var_es(0.0, 1.0, 0.975)
    t = ve.student_t_var_es(0.0, 1.0, 1e6, 0.975)
    assert t.var == pytest.approx(n.var, rel=1e-4)
    assert t.es == pytest.approx(n.es, rel=1e-4)


def test_t_es_matches_numerical_integral():
    nu, alpha = 4.0, 0.99
    est = ve.student_t_var_es(0.0, 1.0, nu, alpha)
    scale = np.sqrt((nu - 2) / nu)
    draws = stats.t.rvs(nu, size=2_000_000, random_state=1) * scale
    losses = -draws
    q = np.quantile(losses, alpha)
    assert est.var == pytest.approx(q, rel=0.01)
    assert est.es == pytest.approx(losses[losses >= q].mean(), rel=0.02)


def test_cornish_fisher_reduces_to_normal():
    n = ve.normal_var_es(0.0, 1.0, 0.99)
    cf = ve.cornish_fisher_var_es(0.0, 1.0, 0.0, 0.0, 0.99)
    assert cf.var == pytest.approx(n.var, rel=1e-9)
    assert cf.es == pytest.approx(n.es, rel=1e-6)


def test_cornish_fisher_fat_tails_increase_var():
    base = ve.cornish_fisher_var_es(0.0, 1.0, 0.0, 0.0, 0.99)
    fat = ve.cornish_fisher_var_es(0.0, 1.0, 0.5, 3.0, 0.99)
    assert fat.var > base.var and fat.es > base.es


def test_historical_matches_normal_on_large_sample():
    hs = ve.historical_var_es(NORMAL, 0.99)
    n = ve.normal_var_es(0.0005, 0.01, 0.99)
    assert hs.var == pytest.approx(n.var, rel=0.02)
    assert hs.es == pytest.approx(n.es, rel=0.02)


def test_fhs_scales_with_current_vol():
    calm = RNG.normal(0, 0.01, 1000)
    stressed = np.concatenate([calm, RNG.normal(0, 0.04, 40)])
    assert ve.filtered_historical_var_es(stressed).var > 2 * ve.historical_var_es(calm).var


def test_mc_normal_matches_closed_form():
    x = RNG.multivariate_normal([0, 0], [[1e-4, 5e-5], [5e-5, 4e-4]], 200_000)
    w = np.array([0.6, 0.4])
    mc = ve.mc_portfolio_var_es(x, w, 0.99, n_sims=400_000, seed=1)
    port = x @ w
    n = ve.normal_var_es(port.mean(), port.std(), 0.99)
    assert mc.var == pytest.approx(n.var, rel=0.02)
    assert mc.es == pytest.approx(n.es, rel=0.02)


def test_mc_t_fatter_than_normal():
    x = RNG.normal(0, 0.01, (5000, 2))
    w = [0.5, 0.5]
    n = ve.mc_portfolio_var_es(x, w, 0.995, seed=2)
    t = ve.mc_portfolio_var_es(x, w, 0.995, dist="t", nu=3.5, seed=2)
    assert t.es > n.es


def test_garch_recovers_parameters():
    omega, a, b, n = 2e-6, 0.08, 0.9, 6000
    rng = np.random.default_rng(3)
    h, r = omega / (1 - a - b), np.empty(n)
    for i in range(n):
        r[i] = np.sqrt(h) * rng.standard_normal()
        h = omega + a * r[i] ** 2 + b * h
    m = ve.fit_garch11(r)
    assert m.alpha == pytest.approx(a, abs=0.03)
    assert m.beta == pytest.approx(b, abs=0.04)
    est = ve.garch_mc_var_es(r, 0.99, horizon=10, n_sims=20_000, seed=4, model=m)
    assert est.es > est.var > 0


def test_rejects_bad_alpha():
    with pytest.raises(ValueError):
        ve.historical_var_es(NORMAL, 0.3)


def test_t_fit_handles_nu_near_two():
    # Very heavy tails: fitted nu can approach 2, which must not collapse VaR to ~0.
    r = stats.t.rvs(2.2, scale=0.005, size=3000, random_state=7)
    t = ve.fit_and_estimate(r, "t", 0.99)
    hs = ve.historical_var_es(r, 0.99)
    assert t.var == pytest.approx(hs.var, rel=0.3)
