"""
REAL-DATA SCRIPT (needs internet; not run in the build environment).

Pull a live option chain from Yahoo Finance via yfinance, back out Black-Scholes implied volatility
for every liquid contract, and save data/real/implied_vols.csv. Then re-run
`python scripts/03_implied_vol_smile.py` and it will also draw the market smile
(figures/vol_smile_market.png, vol_term_structure_market.png).

    python scripts/real_data/fetch_option_chain.py --ticker SPY --rate 0.04 --expiries 6

--rate is the continuously-compounded risk-free rate (use a current Treasury yield).
Caveats: yfinance data is delayed/indicative; SPY pays dividends, which this no-dividend model
ignores (it biases call vs put implied vols slightly); one flat rate is used for all expiries.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import implied_vol as iv  # noqa: E402


def spot_price(tkr):
    return float(tkr.history(period="1d")["Close"].iloc[-1])


def years_to_expiry(expiry_str, now=None):
    now = now or datetime.now()
    return (datetime.strptime(expiry_str, "%Y-%m-%d") - now).days / 365.0


def build_chain(tkr, symbol, rate, n_expiries, min_days=7, max_spread_ratio=0.25):
    S = spot_price(tkr)
    asof = datetime.now().strftime("%Y-%m-%d")
    rows = []
    for expiry in tkr.options:
        T = years_to_expiry(expiry)
        if T * 365 < min_days:
            continue
        if len({r["expiry"] for r in rows}) >= n_expiries:
            break
        chain = tkr.option_chain(expiry)
        for kind, df in (("call", chain.calls), ("put", chain.puts)):
            for _, row in df.iterrows():
                bid, ask = float(row["bid"]), float(row["ask"])
                if bid <= 0 or ask <= bid:
                    continue                                  # no live two-sided market
                mid = 0.5 * (bid + ask)
                if (ask - bid) / mid > max_spread_ratio:
                    continue                                  # too wide to trust
                vol = iv.implied_volatility(mid, S, float(row["strike"]), T, rate, kind)
                if vol is None:
                    continue
                rows.append(dict(ticker=symbol, asof=asof, expiry=expiry, T_years=T, option_type=kind,
                                 strike=float(row["strike"]), underlying_price=S, mid_price=mid,
                                 implied_vol=vol, moneyness=float(row["strike"]) / S))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--rate", type=float, default=0.04, help="risk-free rate, e.g. 0.04")
    ap.add_argument("--expiries", type=int, default=6)
    args = ap.parse_args()

    df = build_chain(yf.Ticker(args.ticker), args.ticker, args.rate, args.expiries)
    out = ROOT / "data" / "real" / "implied_vols.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} contracts across {df['expiry'].nunique()} expiries to {out}")


if __name__ == "__main__":
    main()
