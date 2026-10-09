"""Black-Scholes pricer and Greeks: edge cases, put-call parity, finite-difference checks."""
import numpy as np
import pytest

import black_scholes as bs

BASE = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


def test_deep_itm_call_matches_intrinsic():
    p = bs.call_price(S=200, K=50, T=1.0, r=0.05, sigma=0.2)
    assert p == pytest.approx(200 - 50 * np.exp(-0.05), abs=1e-3)


def test_deep_otm_call_is_near_zero():
    assert bs.call_price(S=50, K=200, T=1.0, r=0.05, sigma=0.2) < 1e-6


def test_near_expiry_converges_to_intrinsic():
    assert bs.call_price(S=110, K=100, T=1e-4, r=0.05, sigma=0.2) == pytest.approx(10.0, abs=0.01)


def test_put_call_parity_random_parameters():
    rng = np.random.default_rng(0)
    for _ in range(200):
        S, K = rng.uniform(50, 150, 2)
        T, r, sigma = rng.uniform(0.05, 3), rng.uniform(0, 0.08), rng.uniform(0.05, 0.8)
        lhs = bs.call_price(S, K, T, r, sigma) - bs.put_price(S, K, T, r, sigma)
        assert lhs == pytest.approx(S - K * np.exp(-r * T), abs=1e-9)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_all_greeks_match_finite_differences(option_type):
    price = bs.call_price if option_type == "call" else bs.put_price
    S, K, T, r, sigma = (BASE[k] for k in ("S", "K", "T", "r", "sigma"))

    def central(f, x, h):
        return (f(x + h) - f(x - h)) / (2 * h)

    num_delta = central(lambda s: price(s, K, T, r, sigma), S, 1e-3)
    num_gamma = (price(S + 1e-2, K, T, r, sigma) - 2 * price(S, K, T, r, sigma)
                 + price(S - 1e-2, K, T, r, sigma)) / 1e-4
    num_vega = central(lambda v: price(S, K, T, r, v), sigma, 1e-5)
    num_theta = -central(lambda t: price(S, K, t, r, sigma), T, 1e-5)  # theta = -dV/dT
    num_rho = central(lambda rr: price(S, K, T, rr, sigma), r, 1e-6)

    assert bs.delta(S, K, T, r, sigma, option_type) == pytest.approx(num_delta, abs=1e-6)
    assert bs.gamma(S, K, T, r, sigma) == pytest.approx(num_gamma, abs=1e-5)
    assert bs.vega(S, K, T, r, sigma) == pytest.approx(num_vega, abs=1e-4)
    assert bs.theta(S, K, T, r, sigma, option_type) == pytest.approx(num_theta, abs=1e-4)
    assert bs.rho(S, K, T, r, sigma, option_type) == pytest.approx(num_rho, abs=1e-4)


def test_put_delta_is_call_delta_minus_one():
    args = (BASE["S"], BASE["K"], BASE["T"], BASE["r"], BASE["sigma"])
    assert bs.delta(*args, "put") == pytest.approx(bs.delta(*args, "call") - 1)


def test_price_increases_with_volatility_and_time():
    vols = [bs.call_price(100, 100, 1.0, 0.05, v) for v in (0.1, 0.2, 0.4)]
    times = [bs.call_price(100, 100, t, 0.05, 0.2) for t in (0.25, 1.0, 2.0)]
    assert vols == sorted(vols) and times == sorted(times)
