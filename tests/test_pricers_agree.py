"""Three independent pricing methods must agree: closed form, binomial tree, Monte Carlo."""
import pytest

import binomial_tree as bt
import black_scholes as bs
import monte_carlo as mc

S, K, T, r, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_binomial_converges_to_black_scholes(option_type):
    exact = (bs.call_price if option_type == "call" else bs.put_price)(S, K, T, r, SIGMA)
    gaps = [abs(bt.binomial_tree_price(S, K, T, r, SIGMA, n, option_type) - exact)
            for n in (50, 500, 2000)]
    assert gaps[-1] < 2e-3
    assert gaps[-1] < gaps[0]


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_monte_carlo_within_four_standard_errors(option_type):
    exact = (bs.call_price if option_type == "call" else bs.put_price)(S, K, T, r, SIGMA)
    price, se = mc.monte_carlo_price(S, K, T, r, SIGMA, 400_000, option_type, seed=7)
    assert abs(price - exact) < 4 * se


def test_monte_carlo_error_shrinks_like_inverse_sqrt_n():
    se_small = mc.monte_carlo_price(S, K, T, r, SIGMA, 10_000, "call", seed=1)[1]
    se_large = mc.monte_carlo_price(S, K, T, r, SIGMA, 1_000_000, "call", seed=1)[1]
    assert se_small / se_large == pytest.approx(10.0, rel=0.1)   # 100x samples -> 10x smaller


def test_american_put_worth_more_than_european_put():
    euro = bt.binomial_tree_price(S, K, T, r, SIGMA, 1000, "put", "european")
    amer = bt.binomial_tree_price(S, K, T, r, SIGMA, 1000, "put", "american")
    assert amer > euro + 0.1          # real early-exercise premium


def test_american_call_equals_european_call_without_dividends():
    euro = bt.binomial_tree_price(S, K, T, r, SIGMA, 1000, "call", "european")
    amer = bt.binomial_tree_price(S, K, T, r, SIGMA, 1000, "call", "american")
    assert amer == pytest.approx(euro, abs=1e-8)
