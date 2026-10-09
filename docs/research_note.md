# Research note: pricing and hedging options under Black-Scholes and jump-diffusion

*All smile and hedging results below are from model-generated data (Merton jump-diffusion / geometric Brownian motion). See README for the data note.*

## 1. Question
How accurate is Black-Scholes as a pricing and hedging tool, and how does its hedge degrade when its assumptions (continuous trading, constant volatility, no jumps, no costs) are relaxed?

## 2. Methods
- **Pricing.** Closed-form Black-Scholes with analytic Greeks (`src/black_scholes.py`); risk-neutral Monte Carlo (`monte_carlo.py`); Cox-Ross-Rubinstein binomial tree with optional American exercise (`binomial_tree.py`).
- **Implied volatility.** Brent root-finding on the Black-Scholes price; returns `None` for quotes violating no-arbitrage bounds (`implied_vol.py`). Smiles use out-of-the-money contracts only.
- **Jump model.** Merton (1976): closed-form price as a Poisson-weighted sum of Black-Scholes prices, plus exact path simulation (`jump_diffusion.py`). Parameters: diffusion vol 15%, 0.5 jumps per year, mean log-jump -10%, jump st.dev. 15% (total vol 19.7%).
- **Hedging.** Sell a 60-day ATM call at its Black-Scholes price, hold delta shares, cash earns the risk-free rate, rebalance at discrete times; P&L is the present value at expiry per option (`delta_hedging.py`). 40,000 paths, seed 2026, spot 100, rate 3%, realised vol 20%.

## 3. Results
**Pricing validation.** ATM call 10.4506; binomial (2,000 steps) maximum error 0.0010 across six contracts; Monte Carlo (1M paths) within 0.34 standard errors; put-call parity error 2.8e-14; analytic vs numerical delta difference 8e-11.

**Convergence.** Binomial error 0.197 at 10 steps, 0.0004 at 5,000. Monte Carlo error 0.539 at 1,000 paths, 0.0026 at 1M; standard-error slope -0.482.

**American options.** ATM put 6.089 (American) vs 5.570 (European), premium 0.519; at strike 130 the premium is 4.701.

**Smile (model).** ATM implied vol 16.98 / 17.92 / 18.50 / 18.97% at 1 / 3 / 6 / 12 months. The 90%-110% skew is 7.49 / 3.97 / 2.55 / 1.52 points. The skew flattens with maturity, as in real markets; the rising ATM level is specific to this model.

**Hedging.**

| Rebalances | Std of P&L |
|---|---|
| 3 (monthly) | 1.805 |
| 12 (weekly) | 0.944 |
| 60 (daily) | 0.432 |
| 240 (4 per day) | 0.218 |

Log-log slope -0.479. At matched volatility the mean P&L is -0.004 (s.e. 0.002). Volatility mismatch: hedging at 25% gives +0.964 (theory +0.967), at 15% gives -0.970 (theory -0.967).

*Jumps (same 19.7% total vol, daily hedging):*

| | Smooth | Jumps |
|---|---|---|
| Std | 0.426 | 2.403 |
| 1st percentile | -1.163 | -12.221 |
| Worst path | -2.724 | -43.728 |
| Excess kurtosis | 1.55 | 40.14 |

The jump-case mean P&L is slightly positive (+0.355) because a variance-matched Black-Scholes price is above the Merton price for ATM options. The mean therefore hides the real risk, which sits in the left tail.

*Costs (5 bp):* daily hedging costs 0.149 per option with std 0.440; weekly costs 0.079 with std 0.950.

## 4. Limitations
Model-generated data; no dividends; flat rate; Merton jumps are i.i.d. with constant parameters; hedging vol is held constant rather than re-estimated; zero bid-ask beyond the proportional cost; results fixed to one seed.

## 5. Next steps
1. Run `scripts/real_data/fetch_option_chain.py` and `hedge_historical_path.py` on SPY and compare with the model results.
2. Add stochastic volatility (Heston) and calibrate to a real chain.
3. Optimise hedging frequency or add a no-trade band under costs.
