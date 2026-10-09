"""
Back out implied volatility from a real market option price.

Black-Scholes has no algebraic inverse for sigma, so this solves for it
numerically: find the sigma that makes the Black-Scholes price match the
observed market price, using Brent's method (a robust, standard root-finder
that doesn't require derivatives and is guaranteed to converge as long as
the answer lies inside the given bracket).
"""

import numpy as np
from scipy.optimize import brentq
import black_scholes as bs


def implied_volatility(market_price, S, K, T, r, option_type="call",
                        vol_lower=1e-4, vol_upper=5.0):
    """
    Solve for the volatility that makes the Black-Scholes price equal to
    market_price. Returns None if no solution exists in the search range
    (this happens for prices that violate no-arbitrage bounds -- e.g. a
    quoted price below intrinsic value, usually from stale/bad data).
    """
    if option_type == "call":
        price_fn = bs.call_price
        # A call's price must exceed its intrinsic value; below that, or
        # above the stock price itself, no volatility can explain the quote.
        intrinsic = max(S - K * np.exp(-r * T), 0)
    else:
        price_fn = bs.put_price
        intrinsic = max(K * np.exp(-r * T) - S, 0)

    if market_price < intrinsic or market_price <= 0:
        return None

    def price_difference(sigma):
        return price_fn(S, K, T, r, sigma) - market_price

    try:
        # Brent's method needs the function to have opposite signs at the
        # two ends of the bracket -- that's what guarantees a root exists
        # somewhere in between.
        if price_difference(vol_lower) * price_difference(vol_upper) > 0:
            return None
        return brentq(price_difference, vol_lower, vol_upper, xtol=1e-6)
    except ValueError:
        return None
