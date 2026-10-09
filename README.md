# Option Pricing and Delta-Hedging Study

Black-Scholes pricing and Greeks, cross-checked against Monte Carlo and binomial-tree pricers, implied-volatility smiles, and a delta-hedging simulator that shows where the textbook model breaks (discrete trading, wrong volatility, jumps, transaction costs).

> **Data note.** The smile and hedging results use **model-generated data** (Merton jump-diffusion and geometric Brownian motion), not market prices. The environment that built this had no market-data access. Scripts for real option chains and real price histories are in `scripts/real_data/`; they were tested only against mock data. Read `EXECUTIVE_SUMMARY.md` first (one page).

## Headline results
| Result | Number |
|---|---|
| ATM call (S=K=100, T=1, r=5%, vol=20%) | 10.4506 |
| Binomial (2,000 steps) max error, 6 contracts | 0.0010 |
| Monte Carlo (1M paths), max error | 0.34 standard errors |
| Put-call parity error | 2.8e-14 |
| American vs European ATM put | 6.089 vs 5.570 |
| Model-smile skew (90%-110%), 1M to 12M | 7.49 to 1.52 vol points |
| Hedge-error std, 3 / 12 / 60 / 240 rebalances | 1.805 / 0.944 / 0.432 / 0.218 |
| Hedge-error scaling with rebalances (log-log slope) | -0.479 (theory -0.5) |
| 1st-percentile P&L, smooth vs jumps | -1.16 vs -12.2 |

## Structure
```
OptionsProject/
├── README.md
├── EXECUTIVE_SUMMARY.md        one-page summary (PDF copy in docs/)
├── requirements.txt
├── src/                        the library
│   ├── black_scholes.py        price + Greeks (closed form)
│   ├── monte_carlo.py          risk-neutral Monte Carlo
│   ├── binomial_tree.py        CRR tree, European and American
│   ├── implied_vol.py          Brent solver for implied volatility
│   ├── jump_diffusion.py       Merton price + path simulation
│   └── delta_hedging.py        hedging P&L, P&L-attribution formula
├── tests/                      34 pytest tests
├── scripts/
│   ├── run_all.py              tests + steps 1-4 (use --skip-tests to skip tests)
│   ├── 01_pricing_validation.py
│   ├── 02_convergence.py
│   ├── 03_implied_vol_smile.py
│   ├── 04_delta_hedging.py
│   └── real_data/              needs internet (yfinance)
│       ├── fetch_option_chain.py
│       └── hedge_historical_path.py
├── notebooks/results.ipynb     executed walkthrough with plots
├── output/
│   ├── figures/                12 PNG charts
│   └── tables/                 CSVs + key_results.json (every number quoted in the docs)
├── data/real/                  empty; see its README
└── docs/
    ├── Executive_Summary.pdf
    ├── research_note.md        methods, results, limitations
    ├── Options_BlackScholes_Background.pdf        plain-language primer
    └── MonteCarlo_BinomialTree_Background.pdf     how the two pricers and their code work
```

## Quick start
```
pip install -r requirements.txt
python scripts/run_all.py            # ~40 s: tests, then all figures and tables
jupyter notebook notebooks/results.ipynb
```

## The four steps
1. **Black-Scholes and Greeks** (`01`): pricer, Greeks, put-call parity and finite-difference checks, intuition charts.
2. **Monte Carlo and binomial tree** (`02`): independent pricers, convergence rates, American options.
3. **Implied volatility and smile** (`03`): invert Black-Scholes on jump-model prices; smile and term structure. If `data/real/implied_vols.csv` exists, market versions are drawn too.
4. **Delta hedging** (`04`): four experiments: rebalancing frequency, volatility mismatch, jumps, transaction costs. Simulator is validated against theory.

## Run on real data
```
python scripts/real_data/fetch_option_chain.py --ticker SPY --rate 0.04 --expiries 6
python scripts/03_implied_vol_smile.py
python scripts/real_data/hedge_historical_path.py --ticker SPY --years 10 --rate 0.03
```
Caveats: yfinance quotes are delayed; the model has no dividends and uses one flat rate; overlapping hedge windows are dependent, so the effective sample is smaller than the position count.

## Limitations
Model-generated data; no dividends; constant-parameter jumps; no stochastic volatility; one seed (2026); the model smile's rising ATM term structure is an artefact of the Merton parameters, not a realistic feature.

## Method notes
- Hedging P&L is a present value per option sold. Costs are discounted from the date paid.
- Smiles use out-of-the-money contracts only (puts below spot, calls above).
- Jump-case mean P&L is positive (+0.355) because variance-matched Black-Scholes overprices the ATM option relative to Merton; the tail is what matters.
