"""Download S&P 500 index closes and cache them to data/sp500.csv.

The CSV is committed so results reproduce without network access.
Run: python scripts/fetch_data.py   (needs: pip install yfinance)
"""

from pathlib import Path

import yfinance as yf

TICKER, START, END = "^GSPC", "2005-01-01", "2026-11-01"
OUT = Path(__file__).resolve().parents[1] / "data" / "sp500.csv"

if __name__ == "__main__":
    px = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)["Close"]
    px = px.squeeze().rename("close").dropna()
    OUT.parent.mkdir(exist_ok=True)
    px.round(4).to_csv(OUT)
    print(f"saved {px.size} rows, {px.index[0].date()} to {px.index[-1].date()} -> {OUT}")
