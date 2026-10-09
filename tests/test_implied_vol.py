"""Implied-volatility solver: must recover a known sigma, and reject impossible prices."""
import numpy as np
import pytest

import black_scholes as bs
import implied_vol as iv


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_recovers_known_volatility(option_type):
    price_fn = bs.call_price if option_type == "call" else bs.put_price
    S, T, r = 100.0, 0.5, 0.05
    for sigma in (0.08, 0.2, 0.35, 0.6):
        for K in (70, 90, 100, 110, 130):
            market = price_fn(S, K, T, r, sigma)
            if market < 1e-8:
                continue                       # vega ~ 0: volatility is not identifiable
            got = iv.implied_volatility(market, S, K, T, r, option_type)
            assert got == pytest.approx(sigma, abs=1e-4), (sigma, K)


def test_price_below_intrinsic_returns_none():
    S, K, T, r = 100.0, 100.0, 0.5, 0.05
    intrinsic = S - K * np.exp(-r * T)
    assert iv.implied_volatility(intrinsic - 1.0, S, K, T, r, "call") is None


def test_nonpositive_price_returns_none():
    assert iv.implied_volatility(0.0, 100, 100, 0.5, 0.05, "put") is None
