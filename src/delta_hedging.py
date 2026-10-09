"""
Delta-hedging simulator for a short European option position.

The experiment: sell one option at its Black-Scholes price (computed with the
"hedging volatility" sigma_h), then dynamically hold delta shares of the stock so
the position is locally risk-free, rebalancing at discrete times. Whatever is
left at expiry is the hedging P&L. In a perfect Black-Scholes world (continuous
trading, constant volatility = sigma_h, no costs) it is exactly zero. This module
measures how it deviates when those assumptions are relaxed.

All P&L is reported as a present value (discounted to time 0) per 1 option sold.
"""

import numpy as np

import black_scholes as bs


def _price_fn(option_type):
    return bs.call_price if option_type == "call" else bs.put_price


def hedge_pnl(paths, K, T, r, sigma_hedge, option_type="call", cost_rate=0.0):
    """
    Hedging P&L (present value) for each path, for a SHORT option position.

    paths:      array (n_paths, n_steps + 1) of stock prices at equally spaced times
    sigma_hedge: volatility used both to price the option sold and to compute deltas
    cost_rate:  proportional transaction cost, as a fraction of the traded notional
                (0.0005 = 5 basis points)

    Returns (pnl, total_cost): arrays of length n_paths.
    """
    n_paths, n_cols = paths.shape
    n_steps = n_cols - 1
    dt = T / n_steps

    S0 = paths[:, 0]
    premium = _price_fn(option_type)(S0, K, T, r, sigma_hedge)
    delta = bs.delta(S0, K, T, r, sigma_hedge, option_type)

    # Sell the option (receive premium), buy `delta` shares, financed from cash.
    cost = cost_rate * np.abs(delta) * S0
    cash = premium - delta * S0 - cost
    total_cost_pv = cost.copy()                           # paid at t=0, so PV = cost

    for i in range(1, n_steps):
        cash = cash * np.exp(r * dt)                      # cash earns the risk-free rate
        S = paths[:, i]
        new_delta = bs.delta(S, K, T - i * dt, r, sigma_hedge, option_type)
        trade = new_delta - delta
        step_cost = cost_rate * np.abs(trade) * S
        cash = cash - trade * S - step_cost
        total_cost_pv += np.exp(-r * i * dt) * step_cost  # discount from the time it was paid
        delta = new_delta

    cash = cash * np.exp(r * dt)
    S_T = paths[:, -1]
    payoff = np.maximum(S_T - K, 0) if option_type == "call" else np.maximum(K - S_T, 0)

    pnl_at_T = cash + delta * S_T - payoff                # liquidate stock, pay option holder
    return np.exp(-r * T) * pnl_at_T, total_cost_pv


def subsample_paths(paths, factor):
    """Rebalance less often: keep every `factor`-th column (path is unchanged)."""
    n_steps = paths.shape[1] - 1
    if n_steps % factor != 0:
        raise ValueError("factor must divide the number of simulated steps")
    return paths[:, ::factor]


def theoretical_pnl_attribution(paths, K, T, r, sigma_hedge, sigma_realised):
    """
    Pathwise P&L predicted by the classic Black-Scholes P&L-attribution result
    (Carr / El Karoui et al.): hedging a short option at volatility sigma_h while the
    stock realises volatility sigma yields, in present value,

        PnL = integral_0^T exp(-r t) * 0.5 * S_t^2 * Gamma_t(sigma_h) * (sigma_h^2 - sigma^2) dt

    This is exact under continuous hedging; with discrete hedging it is a close
    approximation, which makes it a useful independent check on the simulator.
    """
    n_steps = paths.shape[1] - 1
    dt = T / n_steps
    total = np.zeros(paths.shape[0])
    for i in range(n_steps):
        t = i * dt
        S = paths[:, i]
        gamma = bs.gamma(S, K, T - t, r, sigma_hedge)
        total += np.exp(-r * t) * 0.5 * S**2 * gamma * (sigma_hedge**2 - sigma_realised**2) * dt
    return total
