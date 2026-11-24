"""Walk-forward 1-day 99% VaR/ES backtest of every model on the demo portfolio.

Writes results/summary.csv, results/summary.md and charts to results/.
Run: python scripts/run_backtest.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import var_es as ve
from var_es.montecarlo import _garch_filter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
ALPHA, WINDOW, GARCH_REFIT = 0.99, 500, 20
WEIGHTS = pd.Series({"SPY": 0.40, "TLT": 0.30, "GLD": 0.15, "EFA": 0.15})


def garch_fhs_backtest(r: pd.Series) -> pd.DataFrame:
    """GARCH(1,1) filtered HS, refitting parameters every GARCH_REFIT days."""
    x = r.to_numpy()
    rows, model = [], None
    for t in range(WINDOW, x.size):
        w = x[t - WINDOW : t]
        if model is None or (t - WINDOW) % GARCH_REFIT == 0:
            model = ve.fit_garch11(w)
        e = w - model.mu
        h = _garch_filter(np.array([model.omega, model.alpha, model.beta]), e)
        losses = -(model.mu + e / np.sqrt(h[:-1]) * np.sqrt(h[-1]))
        var = float(np.quantile(losses, ALPHA))
        rows.append((var, float(losses[losses >= var].mean()), x[t]))
    bt = pd.DataFrame(rows, columns=["var", "es", "realized"], index=r.index[WINDOW:])
    bt["breach"] = (bt["realized"] < -bt["var"]).astype(int)
    return bt


def main() -> None:
    OUT.mkdir(exist_ok=True)
    px = pd.read_csv(ROOT / "data" / "prices.csv", index_col=0, parse_dates=True)
    rets = px[WEIGHTS.index].pct_change().dropna()
    port = rets @ WEIGHTS  # daily-rebalanced portfolio return

    models = {
        "Historical": lambda w: ve.historical_var_es(w, ALPHA),
        "Filtered HS (EWMA)": lambda w: ve.filtered_historical_var_es(w, ALPHA),
        "Normal": lambda w: ve.fit_and_estimate(w, "normal", ALPHA),
        "Student-t": lambda w: ve.fit_and_estimate(w, "t", ALPHA),
        "Cornish-Fisher": lambda w: ve.fit_and_estimate(w, "cornish_fisher", ALPHA),
    }
    bts = {name: ve.rolling_backtest(port, f, WINDOW) for name, f in models.items()}
    bts["GARCH(1,1) FHS"] = garch_fhs_backtest(port)

    summary = pd.DataFrame({k: ve.summarize(v, ALPHA) for k, v in bts.items()}).T
    summary.to_csv(OUT / "summary.csv")
    fmt = summary.copy()
    fmt["breach_rate"] = (fmt["breach_rate"].astype(float) * 100).map("{:.2f}%".format)
    for c in ["kupiec_p", "christoffersen_ind_p", "cc_p", "acerbi_z2"]:
        fmt[c] = fmt[c].astype(float).map("{:.3f}".format)
    (OUT / "summary.md").write_text(fmt.to_markdown())
    print(fmt.to_string())

    # Chart 1: realised P&L vs VaR for three contrasting models
    fig, ax = plt.subplots(figsize=(11, 5))
    base = bts["Historical"]
    ax.bar(base.index, base["realized"] * 100, width=1.5, color="0.7", label="Daily P&L")
    for name, c in [("Historical", "C0"), ("Normal", "C1"), ("GARCH(1,1) FHS", "C3")]:
        ax.plot(bts[name].index, -bts[name]["var"] * 100, lw=0.9, color=c, label=f"-VaR {name}")
    hits = base[base["breach"] == 1]
    ax.scatter(
        hits.index, hits["realized"] * 100, s=8, color="C0", zorder=3, label="Historical breaches"
    )
    ax.set_ylabel("Return (%)")
    ax.set_title(f"1-day {ALPHA:.0%} VaR vs realised P&L, 40/30/15/15 SPY/TLT/GLD/EFA")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "var_vs_pnl.png", dpi=130)

    # Chart 2: rolling 250-day breach count with Basel zones
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, bt in bts.items():
        ax.plot(bt.index, bt["breach"].rolling(250).sum(), lw=1, label=name)
    ax.axhspan(0, 4.5, color="green", alpha=0.07)
    ax.axhspan(4.5, 9.5, color="gold", alpha=0.12)
    ax.axhspan(9.5, ax.get_ylim()[1] + 5, color="red", alpha=0.07)
    ax.set_ylabel("Breaches, trailing 250d")
    ax.set_title("Rolling breach count vs Basel traffic-light zones")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "rolling_breaches.png", dpi=130)


if __name__ == "__main__":
    main()
