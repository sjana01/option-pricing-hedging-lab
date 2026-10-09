"""Delta-hedging simulator: classical results every correct implementation must reproduce."""
import numpy as np
import pytest

import delta_hedging as dh
import jump_diffusion as jd

S0, K, T, R, SIGMA = 100.0, 100.0, 0.25, 0.03, 0.2


@pytest.fixture(scope="module")
def gbm_paths():
    return jd.simulate_gbm_paths(S0, T, R, SIGMA, n_steps=252, n_paths=40_000, seed=11)


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_matched_vol_hedge_has_zero_mean_pnl(gbm_paths, option_type):
    pnl, _ = dh.hedge_pnl(gbm_paths, K, T, R, SIGMA, option_type)
    assert abs(pnl.mean()) < 4 * se(pnl)


def test_hedge_error_std_scales_like_inverse_sqrt_rebalances(gbm_paths):
    ns, stds = [], []
    for f in (1, 2, 4, 6, 9, 14, 21, 36, 63):
        p = dh.subsample_paths(gbm_paths, f)
        ns.append(p.shape[1] - 1)
        stds.append(dh.hedge_pnl(p, K, T, R, SIGMA)[0].std())
    slope = np.polyfit(np.log(ns), np.log(stds), 1)[0]
    assert -0.58 < slope < -0.42        # theory: -0.5


@pytest.mark.parametrize("sigma_hedge, sign", [(0.25, +1), (0.15, -1)])
def test_vol_mismatch_pnl_sign_and_magnitude(gbm_paths, sigma_hedge, sign):
    """Selling at a volatility above realised wins; below realised loses; size matches theory."""
    pnl, _ = dh.hedge_pnl(gbm_paths, K, T, R, sigma_hedge)
    theory = dh.theoretical_pnl_attribution(gbm_paths, K, T, R, sigma_hedge, SIGMA)
    assert np.sign(pnl.mean()) == sign
    assert abs(pnl.mean() - theory.mean()) < 0.01 * abs(theory.mean()) + 4 * se(pnl)


def test_transaction_costs_grow_with_rebalancing_frequency(gbm_paths):
    costs = [dh.hedge_pnl(dh.subsample_paths(gbm_paths, f), K, T, R, SIGMA, cost_rate=0.0005)[1].mean()
             for f in (63, 9, 1)]
    assert costs == sorted(costs)


def test_costs_reduce_pnl_one_for_one(gbm_paths):
    free, _ = dh.hedge_pnl(gbm_paths, K, T, R, SIGMA, cost_rate=0.0)
    paid, cost = dh.hedge_pnl(gbm_paths, K, T, R, SIGMA, cost_rate=0.0005)
    # Costs are charged to cash, so the P&L reduction equals the PV of the costs exactly.
    assert np.allclose(free - paid, cost, atol=1e-9)


def test_jumps_fatten_the_left_tail_of_hedge_errors():
    sig_d, lam, mu_j, sig_j = 0.15, 0.5, -0.10, 0.15
    total_vol = jd.total_volatility(sig_d, lam, mu_j, sig_j)
    gbm = jd.simulate_gbm_paths(S0, T, R, total_vol, 63, 40_000, seed=5)
    jump = jd.simulate_merton_paths(S0, T, R, sig_d, lam, mu_j, sig_j, 63, 40_000, seed=5)
    p_gbm, _ = dh.hedge_pnl(gbm, K, T, R, total_vol)
    p_jump, _ = dh.hedge_pnl(jump, K, T, R, total_vol)
    assert np.percentile(p_jump, 0.5) < np.percentile(p_gbm, 0.5)    # worse worst-case
