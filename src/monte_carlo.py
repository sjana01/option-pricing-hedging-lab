"""
Monte Carlo option pricing: simulate many random stock price paths under
the same risk-neutral random-walk assumption Black-Scholes is built on,
average the discounted payoffs, and see if it converges to the closed-form
Black-Scholes price.

The stock is simulated as geometric Brownian motion:
    S_T = S_0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
where Z is a standard normal random draw. This is the exact same
assumption about how prices move that underlies the Black-Scholes formula
itself -- which is exactly why the two should agree.
"""

import numpy as np


def simulate_terminal_prices(S, T, r, sigma, n_simulations, seed=None):
    """Simulate n_simulations random final stock prices at time T."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_simulations)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    return S_T


def monte_carlo_price(S, K, T, r, sigma, n_simulations=100_000, option_type="call", seed=None):
    """
    Price a European option via Monte Carlo. Returns (price, standard_error)
    -- the standard error tells you how precise this estimate is, which is
    the whole point of also reporting a confidence interval.
    """
    S_T = simulate_terminal_prices(S, T, r, sigma, n_simulations, seed)

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - S_T, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted_payoffs = np.exp(-r * T) * payoffs

    price = discounted_payoffs.mean()
    standard_error = discounted_payoffs.std(ddof=1) / np.sqrt(n_simulations)

    return price, standard_error
