# Real market data (not included)

The build environment had no access to Yahoo Finance, so every result in this repo uses
**model-generated data** (Merton jump-diffusion / geometric Brownian motion), labelled as such.

To add real data, run from the repo root with internet access:

    python scripts/real_data/fetch_option_chain.py --ticker SPY --rate 0.04 --expiries 6
    python scripts/03_implied_vol_smile.py        # now also draws the market smile
    python scripts/real_data/hedge_historical_path.py --ticker SPY --years 10 --rate 0.03

Files written here (git-ignored): `implied_vols.csv`, `prices_<TICKER>.csv`.
Outputs go to `output/figures/` (`vol_smile_market.png`, `vol_term_structure_market.png`,
`hedge_real_<TICKER>.png`) and `output/tables/hedge_real_<TICKER>.csv`.
