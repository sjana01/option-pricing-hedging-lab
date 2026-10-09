"""
REAL-DATA SCRIPT (needs internet unless a price CSV already exists; not run in the build environment).

Delta-hedge a short at-the-money 60-trading-day call on REAL historical prices. A new position is
opened every 5 trading days; each is hedged daily with Black-Scholes delta using the trailing
30-day realised volatility known at the start date as the hedging volatility. Outputs the P&L
distribution of those positions. Compare with the simulated GBM/jump results in output/tables.

    python scripts/real_data/hedge_historical_path.py --ticker SPY --years 10 --rate 0.03

Caveats: overlapping windows are strongly dependent, so the effective sample size is much smaller than
the number of positions; no dividends; flat rate; daily closes only; zero transaction costs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import delta_hedging as dh  # noqa: E402

DAYS, STRIDE, LOOKBACK = 60, 5, 30


def load_prices(ticker, years):
    csv = ROOT / "data" / "real" / f"prices_{ticker}.csv"
    if csv.exists():
        return pd.read_csv(csv, index_col=0, parse_dates=True)["Close"].dropna()
    import yfinance as yf
    close = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)["Close"]
    close = close.squeeze().dropna()
    csv.parent.mkdir(parents=True, exist_ok=True)
    close.rename("Close").to_frame().to_csv(csv)
    return close


def windows(close):
    px = close.to_numpy(dtype=float)
    starts = range(LOOKBACK, len(px) - DAYS, STRIDE)
    paths = np.array([px[i:i + DAYS + 1] for i in starts])
    log_ret = np.diff(np.log(px))
    trailing_vol = np.array([log_ret[i - LOOKBACK:i].std(ddof=1) * np.sqrt(252) for i in starts])
    realised_vol = np.array([np.diff(np.log(px[i:i + DAYS + 1])).std(ddof=1) * np.sqrt(252) for i in starts])
    dates = close.index[list(starts)]
    return paths, trailing_vol, realised_vol, dates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--rate", type=float, default=0.03)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    close = load_prices(args.ticker, args.years)
    paths, hedge_vol, realised, dates = windows(close)
    T = DAYS / 252
    pnl, _ = dh.hedge_pnl(paths, paths[:, 0], T, args.rate, hedge_vol)       # strike = spot (ATM)

    out = pd.DataFrame(dict(start=dates, hedge_vol=hedge_vol, realised_vol=realised, pnl_per_option=pnl,
                            pnl_pct_of_spot=pnl / paths[:, 0] * 100))
    tdir = ROOT / "output" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(tdir / f"hedge_real_{args.ticker}.csv", index=False)
    summary = dict(positions=len(out), mean_pct=out.pnl_pct_of_spot.mean(), std_pct=out.pnl_pct_of_spot.std(),
                   p1_pct=out.pnl_pct_of_spot.quantile(0.01), worst_pct=out.pnl_pct_of_spot.min(),
                   corr_pnl_vs_vol_gap=np.corrcoef(pnl, hedge_vol - realised)[0, 1])
    print(pd.Series(summary).round(4).to_string())

    if not args.no_plot:
        from _common import plt, save_fig, BLUE, ORANGE
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
        axes[0].hist(out.pnl_pct_of_spot, bins=40, color=BLUE, alpha=0.85)
        axes[0].set(xlabel="Hedging P&L per option sold (% of spot, PV)", ylabel="Positions",
                    title=f"{args.ticker}: daily delta-hedge outcomes")
        axes[1].scatter((hedge_vol - realised) * 100, out.pnl_pct_of_spot, s=10, color=ORANGE, alpha=0.7)
        axes[1].set(xlabel="Hedging vol minus subsequently realised vol (vol points)",
                    ylabel="P&L (% of spot, PV)", title="P&L is driven by the volatility gap")
        fig.text(0.01, 0.005, f"Data: {args.ticker} daily closes, {close.index[0].date()} to {close.index[-1].date()}",
                 fontsize=8, ha="left", va="bottom")
        save_fig(fig, f"hedge_real_{args.ticker}.png")


if __name__ == "__main__":
    main()
