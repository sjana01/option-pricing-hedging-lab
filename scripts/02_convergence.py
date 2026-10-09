"""
Step 2 - Monte Carlo and binomial tree converge to Black-Scholes; early exercise adds value.

Outputs: figures/binomial_convergence.png, monte_carlo_convergence.png, american_vs_european_put.png
         tables/convergence.csv, tables/key_results.json (section 'convergence')
"""
import numpy as np
import pandas as pd

from _common import *  # noqa: F401,F403
import binomial_tree as bt
import black_scholes as bs
import monte_carlo as mc

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20
EXACT = bs.call_price(S0, K, T, R, SIGMA)


def binomial():
    steps = np.unique(np.logspace(0.5, 3.7, 45).astype(int))
    prices = np.array([bt.binomial_tree_price(S0, K, T, R, SIGMA, n, "call") for n in steps])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(steps, prices, color=BLUE, label="Binomial tree")
    ax.axhline(EXACT, color=ORANGE, ls="--", label=f"Black-Scholes ({EXACT:.4f})")
    ax.set(xscale="log", xlabel="Number of time steps (log scale)", ylabel="Call price",
           title="Binomial tree converges to Black-Scholes")
    ax.legend()
    save_fig(fig, "binomial_convergence.png")
    return steps, prices


def monte_carlo():
    counts = np.unique(np.logspace(2, 6, 25).astype(int))
    est, se = zip(*[mc.monte_carlo_price(S0, K, T, R, SIGMA, n, "call", seed=42) for n in counts])
    est, se = np.array(est), np.array(se)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.plot(counts, est, color=BLUE, label="Monte Carlo estimate")
    ax.fill_between(counts, est - 1.96 * se, est + 1.96 * se, color=BLUE, alpha=0.15, lw=0,
                    label="95% confidence band")
    ax.axhline(EXACT, color=ORANGE, ls="--", label="Black-Scholes")
    ax.set(xscale="log", xlabel="Simulations (log scale)", ylabel="Call price",
           title="Estimate settles on the exact price")
    ax.legend()
    ax = axes[1]
    ax.plot(counts, np.abs(est - EXACT), color=BLUE, marker="o", ms=3, lw=1, label="Actual error")
    ax.plot(counts, 1.96 * se, color=ORANGE, label="95% half-width (1.96 x std. error)")
    ax.set(xscale="log", yscale="log", xlabel="Simulations (log scale)", ylabel="Absolute error (log scale)",
           title="Error falls like 1/sqrt(N)")
    ax.legend()
    save_fig(fig, "monte_carlo_convergence.png")
    slope = np.polyfit(np.log(counts), np.log(se), 1)[0]
    return counts, est, se, slope


def american():
    strikes = np.linspace(70, 130, 31)
    euro = np.array([bt.binomial_tree_price(S0, k, T, R, SIGMA, 500, "put", "european") for k in strikes])
    amer = np.array([bt.binomial_tree_price(S0, k, T, R, SIGMA, 500, "put", "american") for k in strikes])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(strikes, euro, color=BLUE, label="European put")
    ax.plot(strikes, amer, color=ORANGE, label="American put")
    ax.set(xlabel="Strike (stock price = 100)", ylabel="Put price",
           title="Early-exercise premium grows as the put goes in the money")
    ax.legend()
    save_fig(fig, "american_vs_european_put.png")
    return strikes, euro, amer


if __name__ == "__main__":
    print("Step 2: convergence")
    steps, tree = binomial()
    counts, est, se, slope = monte_carlo()
    strikes, euro, amer = american()

    pd.DataFrame({"binomial_steps": steps, "binomial_price": tree, "abs_err": np.abs(tree - EXACT)}
                 ).to_csv(TABLE_DIR / "convergence_binomial.csv", index=False, float_format="%.6f")
    pd.DataFrame({"simulations": counts, "mc_price": est, "std_error": se, "abs_err": np.abs(est - EXACT)}
                 ).to_csv(TABLE_DIR / "convergence_monte_carlo.csv", index=False, float_format="%.6f")
    i100 = list(strikes).index(100.0)
    record("convergence", black_scholes_atm_call=EXACT,
           binomial_err_10_steps=abs(bt.binomial_tree_price(S0, K, T, R, SIGMA, 10, "call") - EXACT),
           binomial_err_5000_steps=abs(bt.binomial_tree_price(S0, K, T, R, SIGMA, 5000, "call") - EXACT),
           mc_err_1k=float(abs(est[np.argmin(abs(counts - 1000))] - EXACT)),
           mc_err_1m=float(abs(est[-1] - EXACT)), mc_stderr_slope_vs_logN=float(slope),
           american_put_atm=float(amer[i100]), european_put_atm=float(euro[i100]),
           early_exercise_premium_atm=float(amer[i100] - euro[i100]),
           early_exercise_premium_K130=float(amer[-1] - euro[-1]))
    print(f"  slope of std error vs N (log-log): {slope:.3f} (theory -0.5)")
