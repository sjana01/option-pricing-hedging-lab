# Executive Summary: Option Pricing and Delta-Hedging Study

**Author:** Santanil Jana  |  **Code:** `OptionsProject/`  |  **Date:** October 2026

## What this is
A from-scratch Python study of how options are priced and how well they can be hedged. It builds the Black-Scholes model and its sensitivities (the "Greeks"), checks it against two independent methods, backs out implied volatility, and then stress-tests a delta hedge to see where the textbook model breaks. 34 automated tests pass, and one command rebuilds every table and figure.

## Important data note
All smile and hedging results use **model-generated data** (a jump-diffusion model and simulated stock paths), not market prices. The build environment could not reach market-data providers. Ready-to-run scripts for real option chains and real price histories are included (`scripts/real_data/`) and were tested only on mock data. No result here should be read as a statement about real markets.

## Key findings
1. **Three pricing methods agree.** For a 1-year at-the-money call (stock 100, strike 100, rate 5%, vol 20%), Black-Scholes gives 10.4506. A 2,000-step binomial tree is within 0.001 of that, and a 1-million-path Monte Carlo is within 0.34 standard errors for every contract tested. Put-call parity holds to 3e-14.
2. **Convergence behaves as theory predicts.** Binomial error falls from 0.197 (10 steps) to 0.0004 (5,000 steps). Monte Carlo error falls from 0.539 (1,000 paths) to 0.0026 (1 million), with a fitted slope of -0.48 against the theoretical -0.5.
3. **American puts are worth more.** At the money, early exercise adds 0.52 (6.09 vs 5.57); for a deep in-the-money strike (130) it adds 4.70. Only the tree can price this.
4. **Jumps create a volatility skew.** Pricing from a jump-diffusion model and inverting Black-Scholes gives implied vols that rise as strikes fall: the 90%-vs-110% strike skew is 7.5 volatility points at 1 month and 1.5 at 12 months.
5. **Hedge error shrinks with trading frequency, slowly.** Over a 60-day call, the standard deviation of hedging P&L is 1.81 with 3 rebalances, 0.94 with 12, 0.43 with daily (60) and 0.22 with 240. It falls like 1/sqrt(n) (fitted slope -0.48): halving the error costs four times the trading.
6. **A wrong volatility input produces a predictable P&L.** Hedging at 25% when the stock realises 20% gives +0.964 per option (formula: +0.967); at 15% it gives -0.970 (formula: -0.967). The simulator matches theory, and at matched volatility the mean P&L is zero within noise (-0.004, s.e. 0.002).
7. **Jumps wreck the tails even when the average looks fine.** At the same total volatility (19.7%), the worst 1% of outcomes is -1.16 under smooth paths and -12.2 under jumps; the single worst path is -2.7 versus -43.7. Rebalancing more often does not fix gap risk.
8. **Costs set a limit.** With 5 basis-point trading costs, daily hedging costs 0.149 per option versus 0.079 for weekly, while cutting risk from a standard deviation of 0.95 to 0.44. The best frequency depends on how much risk you will accept per unit of cost.

## Limitations
- Model-generated data only; real-data scripts untested on live feeds.
- No dividends, a flat interest rate, and constant-parameter jump model; no stochastic volatility.
- The model smile's rising ATM term structure is an artefact of the jump model, not realistic.
- Simulation results depend on a fixed seed and 40,000 paths (standard errors are reported in `output/tables/`).

## Next steps
Run `scripts/real_data/` on SPY to compare the model smile and hedge outcomes with real ones; add stochastic volatility (Heston) and calibrate it to a real chain.
