import numpy as np
import pytest
from scipy import stats

import var_es as ve

RNG = np.random.default_rng(0)


def test_historical_matches_normal_quantile_on_large_sample():
    r = RNG.normal(0.0005, 0.01, 400_000)
    hs = ve.historical_var_es(r, 0.99)
    z = stats.norm.ppf(0.99)
    assert hs.var == pytest.approx(-0.0005 + 0.01 * z, rel=0.02)
    assert hs.es == pytest.approx(-0.0005 + 0.01 * stats.norm.pdf(z) / 0.01, rel=0.02)


def test_historical_horizon_scaling():
    r = RNG.normal(0, 0.01, 5000)
    assert ve.historical_var_es(r, 0.99, 10).var == pytest.approx(
        ve.historical_var_es(r, 0.99).var * np.sqrt(10)
    )


def test_gbm_matches_lognormal_closed_form():
    mu, sigma = 0.0004, 0.012
    log_r = RNG.normal(mu - 0.5 * sigma**2, sigma, 200_000)
    r = np.expm1(log_r)
    est = ve.gbm_mc_var_es(r, 0.99, horizon=10, n_sims=400_000, seed=1)
    m, s = ve.fit_gbm(r)
    # loss = 1 - exp(N((m - s^2/2) h, s^2 h)), so VaR = 1 - exp(drift - s sqrt(h) z)
    exact = 1 - np.exp((m - 0.5 * s**2) * 10 - s * np.sqrt(10) * stats.norm.ppf(0.99))
    assert est.var == pytest.approx(exact, rel=0.01)
    assert est.es > est.var


def _simulate_garch_t(n, mu, omega, a, b, nu, seed):
    rng = np.random.default_rng(seed)
    scale = np.sqrt((nu - 2) / nu)
    h, r = omega / (1 - a - b), np.empty(n)
    for i in range(n):
        eps = np.sqrt(h) * rng.standard_t(nu) * scale
        r[i] = mu + eps
        h = omega + a * eps**2 + b * h
    return r


def test_garch_t_recovers_parameters():
    r = _simulate_garch_t(6000, 0.0003, 2e-6, 0.08, 0.90, 6.0, seed=3)
    m = ve.fit_garch_t(r)
    assert m.alpha == pytest.approx(0.08, abs=0.03)
    assert m.beta == pytest.approx(0.90, abs=0.04)
    assert 4.0 < m.nu < 9.0


def test_garch_t_one_day_matches_closed_form_t():
    r = _simulate_garch_t(3000, 0.0, 2e-6, 0.08, 0.90, 5.0, seed=4)
    m = ve.fit_garch_t(r)
    est = ve.garch_t_var_es(r, 0.99, model=m)
    sim = (
        -np.sqrt(m.next_var)
        * np.sqrt((m.nu - 2) / m.nu)
        * stats.t.rvs(m.nu, size=2_000_000, random_state=5)
        - m.mu
    )
    q = np.quantile(sim, 0.99)
    assert est.var == pytest.approx(q, rel=0.01)
    assert est.es == pytest.approx(sim[sim >= q].mean(), rel=0.02)


def test_garch_t_multi_day_is_simulated_and_ordered():
    r = _simulate_garch_t(3000, 0.0, 2e-6, 0.08, 0.90, 5.0, seed=6)
    m = ve.fit_garch_t(r)
    one = ve.garch_t_var_es(r, 0.99, model=m)
    ten = ve.garch_t_var_es(r, 0.99, horizon=10, model=m, n_sims=40_000, seed=7)
    assert ten.es > ten.var > one.var


def test_evt_recovers_gpd_tail():
    # Exceedances of a GPD over any threshold are GPD with the same shape (threshold
    # stability), so POT should recover xi and the true quantile.
    losses = stats.genpareto.rvs(0.25, scale=0.01, size=200_000, random_state=8)
    tail = ve.fit_pot(-losses, 0.95)
    assert tail.xi == pytest.approx(0.25, abs=0.04)
    est = ve.evt_pot_var_es(-losses, 0.999, 0.95)
    assert est.var == pytest.approx(stats.genpareto.ppf(0.999, 0.25, scale=0.01), rel=0.05)
    exact_es = stats.genpareto.expect(
        lambda x: x, args=(0.25,), scale=0.01, lb=est.var, conditional=True
    )
    assert est.es == pytest.approx(exact_es, rel=0.05)


def test_evt_es_formula_matches_numeric_tail_mean():
    tail = ve.GPDTail(threshold=0.02, xi=0.2, beta=0.008, n=1000, n_exceed=100)
    us = np.linspace(0.99, 1 - 1e-9, 400_001)
    numeric = np.trapezoid([tail.var(u) for u in us[::100]], us[::100]) / 0.01
    assert tail.es(0.99) == pytest.approx(numeric, rel=0.02)


def test_evt_needs_alpha_above_threshold():
    with pytest.raises(ValueError):
        ve.evt_pot_var_es(RNG.normal(size=1000), 0.9, 0.95)


def test_rejects_bad_alpha():
    with pytest.raises(ValueError):
        ve.historical_var_es(RNG.normal(size=100), 0.3)
