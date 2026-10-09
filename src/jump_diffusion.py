"""
Merton (1976) jump-diffusion model.

Why this module exists: Black-Scholes assumes the stock moves continuously
with constant volatility. Real markets show sudden jumps, and options markets
price that in (the implied-volatility smile/skew). Merton's model keeps the
Black-Scholes diffusion but adds random Poisson jumps, which gives a market-like
world where the Black-Scholes assumptions are known to fail. It has a closed-form
European option price, so it can serve as a controlled stand-in for market prices
and as a path generator for stress-testing hedges.

Risk-neutral dynamics:
    dS/S = (r - lam*kappa) dt + sigma dW + (J - 1) dN
    N ~ Poisson(lam),  ln J ~ Normal(mu_j, sigma_j^2),  kappa = E[J - 1] = exp(mu_j + sigma_j^2/2) - 1

Closed-form call price (Merton 1976): a Poisson-weighted sum of Black-Scholes prices,
each conditioned on k jumps having occurred.
"""

import numpy as np
from scipy.stats import poisson

import black_scholes as bs


def total_volatility(sigma, lam, mu_j, sigma_j):
    """Annualised volatility of log-returns including jumps (variance-matched)."""
    return np.sqrt(sigma**2 + lam * (mu_j**2 + sigma_j**2))


def merton_call_price(S, K, T, r, sigma, lam, mu_j, sigma_j, n_terms=60):
    kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    lam_p = lam * (1 + kappa)
    price = 0.0
    for k in range(n_terms):
        weight = poisson.pmf(k, lam_p * T)
        sigma_k = np.sqrt(sigma**2 + k * sigma_j**2 / T)
        r_k = r - lam * kappa + k * np.log(1 + kappa) / T
        price += weight * bs.call_price(S, K, T, r_k, sigma_k)
    return price


def merton_put_price(S, K, T, r, sigma, lam, mu_j, sigma_j, n_terms=60):
    # Put-call parity is model-independent (it only needs no-arbitrage).
    call = merton_call_price(S, K, T, r, sigma, lam, mu_j, sigma_j, n_terms)
    return call - S + K * np.exp(-r * T)


def simulate_gbm_paths(S0, T, r, sigma, n_steps, n_paths, seed=None):
    """Risk-neutral geometric Brownian motion, shape (n_paths, n_steps + 1)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_incr = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_incr, axis=1)], axis=1)
    return S0 * np.exp(log_paths)


def simulate_merton_paths(S0, T, r, sigma, lam, mu_j, sigma_j, n_steps, n_paths, seed=None):
    """Risk-neutral Merton jump-diffusion paths, shape (n_paths, n_steps + 1)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1

    z = rng.standard_normal((n_paths, n_steps))
    n_jumps = rng.poisson(lam * dt, size=(n_paths, n_steps))
    z_jump = rng.standard_normal((n_paths, n_steps))
    # Sum of n i.i.d. Normal(mu_j, sigma_j^2) jumps is Normal(n*mu_j, n*sigma_j^2).
    jump_sum = n_jumps * mu_j + np.sqrt(n_jumps) * sigma_j * z_jump

    drift = (r - lam * kappa - 0.5 * sigma**2) * dt
    log_incr = drift + sigma * np.sqrt(dt) * z + jump_sum
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_incr, axis=1)], axis=1)
    return S0 * np.exp(log_paths)
