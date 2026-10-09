"""
Black-Scholes pricing and Greeks, implemented directly from the closed-form
equations -- no library shortcut that hides the math.

Inputs used throughout (matching the background note):
    S       current stock price
    K       strike price
    T       time to expiry, in years
    r       risk-free interest rate (annualized, e.g. 0.05 for 5%)
    sigma   volatility (annualized standard deviation of returns, e.g. 0.2 for 20%)
"""

import numpy as np
from scipy.stats import norm


def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def call_price(S, K, T, r, sigma):
    D1, D2 = d1(S, K, T, r, sigma), d2(S, K, T, r, sigma)
    return S * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)


def put_price(S, K, T, r, sigma):
    D1, D2 = d1(S, K, T, r, sigma), d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-D2) - S * norm.cdf(-D1)


# ---------------------------------------------------------------------
# Greeks -- each implemented from its own closed-form derivative of the
# pricing formula above, not derived numerically.
# ---------------------------------------------------------------------

def delta(S, K, T, r, sigma, option_type="call"):
    D1 = d1(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(D1)
    elif option_type == "put":
        return norm.cdf(D1) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def gamma(S, K, T, r, sigma):
    # Same formula for calls and puts -- gamma doesn't depend on option type.
    D1 = d1(S, K, T, r, sigma)
    return norm.pdf(D1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    # Same formula for calls and puts. Note: this is per 1.00 (100%) change
    # in volatility -- conventionally divided by 100 to express "per 1% vol move".
    D1 = d1(S, K, T, r, sigma)
    return S * norm.pdf(D1) * np.sqrt(T)


def theta(S, K, T, r, sigma, option_type="call"):
    D1, D2 = d1(S, K, T, r, sigma), d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(D1) * sigma) / (2 * np.sqrt(T))

    if option_type == "call":
        term2 = -r * K * np.exp(-r * T) * norm.cdf(D2)
        return term1 + term2
    elif option_type == "put":
        term2 = r * K * np.exp(-r * T) * norm.cdf(-D2)
        return term1 + term2
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def rho(S, K, T, r, sigma, option_type="call"):
    D2 = d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(D2)
    elif option_type == "put":
        return -K * T * np.exp(-r * T) * norm.cdf(-D2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
