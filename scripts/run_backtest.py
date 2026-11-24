"""Rolling-window backtest of four tail-risk models on S&P 500 daily returns.

Each day t the models are fitted on the previous WINDOW returns and forecast 1-day VaR/ES
for day t. Writes results/summary_<level>.csv/.md and charts to results/.
Run: python scripts/run_backtest.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import var_es as ve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
WINDOW, GARCH_REFIT, LEVELS = 500, 20, (0.99, 0.975)


class RollingGarchT:
    """Refit GARCH(1,1)-t parameters every ``refit`` days; filter variance daily."""

    def __init__(self, alpha: float, refit: int = GARCH_REFIT) -> None:
        self.alpha, self.refit, self.calls, self.model = alpha, refit, 0, None

    def __call__(self, window: np.ndarray) -> ve.RiskEstimate:
        if self.model is None or self.calls % self.refit == 0:
            self.model = ve.fit_garch_t(window)
        self.calls += 1
        return ve.garch_t_var_es(window, self.alpha, model=self.model)


def models(alpha: float) -> dict:
    return {
        "GBM Monte Carlo": lambda w: ve.gbm_mc_var_es(w, alpha, n_sims=20_000, seed=0),
        "Historical Simulation": lambda w: ve.historical_var_es(w, alpha),
        "GARCH(1,1)-t": RollingGarchT(alpha),
        "EVT POT (GPD)": lambda w: ve.evt_pot_var_es(w, alpha, threshold_quantile=0.90),
    }


def fmt_table(summary: pd.DataFrame) -> str:
    t = summary.copy()
    t["breach_rate"] = (t["breach_rate"].astype(float) * 100).map("{:.2f}%".format)
    for c in ["kupiec_p", "christoffersen_ind_p", "cc_p", "loss_to_es_on_breach"]:
        t[c] = t[c].astype(float).map("{:.3f}".format)
    return t.to_markdown()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    px = pd.read_csv(ROOT / "data" / "sp500.csv", index_col=0, parse_dates=True)["close"]
    r = px.pct_change().dropna()

    all_bts = {}
    for alpha in LEVELS:
        bts = {name: ve.rolling_backtest(r, f, WINDOW) for name, f in models(alpha).items()}
        all_bts[alpha] = bts
        summary = pd.DataFrame({k: ve.summarize(v, alpha) for k, v in bts.items()}).T
        tag = f"{alpha * 1000:.0f}"
        summary.to_csv(OUT / f"summary_{tag}.csv")
        (OUT / f"summary_{tag}.md").write_text(fmt_table(summary))
        print(f"\n{alpha:.1%} VaR\n{fmt_table(summary)}")

        # Breaches by calendar year: where do the models fail?
        yearly = pd.DataFrame({k: v["breach"].groupby(v.index.year).sum() for k, v in bts.items()})
        yearly.to_csv(OUT / f"breaches_by_year_{tag}.csv")

    bts = all_bts[0.99]
    colors = {
        "GBM Monte Carlo": "C1",
        "Historical Simulation": "C0",
        "GARCH(1,1)-t": "C3",
        "EVT POT (GPD)": "C2",
    }

    fig, ax = plt.subplots(figsize=(11, 5))
    base = bts["Historical Simulation"]
    ax.bar(base.index, base["realized"] * 100, width=1.5, color="0.75", label="Daily return")
    for name, bt in bts.items():
        ax.plot(bt.index, -bt["var"] * 100, lw=0.9, color=colors[name], label=f"-VaR {name}")
    ax.set_ylabel("Return (%)")
    ax.set_ylim(-13, 12)
    ax.set_title("S&P 500: 1-day 99% VaR forecasts vs realised returns (500-day window)")
    ax.legend(loc="lower left", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "var_vs_returns.png", dpi=130)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name, bt in bts.items():
        ax.plot(bt.index, bt["breach"].rolling(250).sum(), lw=1.1, color=colors[name], label=name)
    ax.axhline(2.5, color="k", ls="--", lw=0.8, label="Expected (2.5)")
    ax.set_ylabel("Breaches, trailing 250 days")
    ax.set_title("Rolling 99% VaR breach count")
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(OUT / "rolling_breaches.png", dpi=130)

    fig, ax = plt.subplots(figsize=(11, 4))
    covid = slice("2020-02-01", "2020-06-30")
    sub = bts["Historical Simulation"].loc[covid]
    ax.bar(sub.index, sub["realized"] * 100, color="0.75", label="Daily return")
    for name, bt in bts.items():
        ax.plot(
            bt.loc[covid].index, -bt.loc[covid, "var"] * 100, color=colors[name], lw=1.4, label=name
        )
    ax.set_ylabel("Return (%)")
    ax.set_title("Zoom: COVID crash, Feb-Jun 2020 (99% VaR)")
    ax.legend(fontsize=8, ncol=5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "covid_zoom.png", dpi=130)


if __name__ == "__main__":
    main()
