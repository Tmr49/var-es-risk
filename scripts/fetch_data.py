"""Download adjusted closes for the demo portfolio and cache them to data/prices.csv.

The CSV is committed so results are reproducible without network access.
Run: python scripts/fetch_data.py
"""

from pathlib import Path

import yfinance as yf

TICKERS = ["SPY", "TLT", "GLD", "EFA"]
START, END = "2005-01-01", "2026-11-01"
OUT = Path(__file__).resolve().parents[1] / "data" / "prices.csv"

if __name__ == "__main__":
    px = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)["Close"]
    px = px[TICKERS].dropna()
    OUT.parent.mkdir(exist_ok=True)
    px.round(4).to_csv(OUT)
    print(f"saved {px.shape[0]} rows, {px.index[0].date()} to {px.index[-1].date()} -> {OUT}")
