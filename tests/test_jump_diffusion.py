"""Merton jump-diffusion: closed form vs. Black-Scholes limit vs. simulated paths."""
import numpy as np
import pytest

import black_scholes as bs
import jump_diffusion as jd

P = dict(S=100.0, K=100.0, T=0.5, r=0.03, sigma=0.15, lam=0.5, mu_j=-0.10, sigma_j=0.15)


def price(kind, **kw):
    a = {**P, **kw}
    fn = jd.merton_call_price if kind == "call" else jd.merton_put_price
    return fn(a["S"], a["K"], a["T"], a["r"], a["sigma"], a["lam"], a["mu_j"], a["sigma_j"])


def test_reduces_to_black_scholes_without_jumps():
    assert price("call", lam=0.0) == pytest.approx(
        bs.call_price(P["S"], P["K"], P["T"], P["r"], P["sigma"]), abs=1e-10)


def test_put_call_parity_holds():
    lhs = price("call") - price("put")
    assert lhs == pytest.approx(P["S"] - P["K"] * np.exp(-P["r"] * P["T"]), abs=1e-10)


@pytest.mark.parametrize("K", [85.0, 100.0, 115.0])
def test_simulated_paths_reproduce_closed_form_price(K):
    """Validates the path generator and the closed form against each other."""
    paths = jd.simulate_merton_paths(P["S"], P["T"], P["r"], P["sigma"], P["lam"],
                                     P["mu_j"], P["sigma_j"], n_steps=50, n_paths=400_000, seed=3)
    payoff = np.exp(-P["r"] * P["T"]) * np.maximum(paths[:, -1] - K, 0)
    se = payoff.std(ddof=1) / np.sqrt(len(payoff))
    assert abs(payoff.mean() - price("call", K=K)) < 4 * se


def test_risk_neutral_mean_of_terminal_price():
    paths = jd.simulate_merton_paths(P["S"], P["T"], P["r"], P["sigma"], P["lam"],
                                     P["mu_j"], P["sigma_j"], n_steps=50, n_paths=400_000, seed=4)
    se = paths[:, -1].std(ddof=1) / np.sqrt(paths.shape[0])
    assert abs(paths[:, -1].mean() - P["S"] * np.exp(P["r"] * P["T"])) < 4 * se


def test_negative_jump_mean_creates_downside_skew():
    import implied_vol as iv
    put_otm = price("put", K=90.0)
    call_otm = price("call", K=110.0)
    iv_put = iv.implied_volatility(put_otm, P["S"], 90.0, P["T"], P["r"], "put")
    iv_call = iv.implied_volatility(call_otm, P["S"], 110.0, P["T"], P["r"], "call")
    assert iv_put > iv_call
