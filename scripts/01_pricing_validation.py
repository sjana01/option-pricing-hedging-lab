"""
Step 1 - Black-Scholes pricer and Greeks: validation numbers and intuition charts.

Outputs: figures/price_vs_stock.png, greeks_vs_stock.png, price_vs_vol_and_time.png
         tables/pricing_methods_comparison.csv, tables/key_results.json (section 'pricing')
"""
import numpy as np
import pandas as pd

from _common import *  # noqa: F401,F403
import binomial_tree as bt
import black_scholes as bs
import monte_carlo as mc

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20


def charts():
    stocks = np.linspace(50, 150, 300)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(stocks, bs.call_price(stocks, K, T, R, SIGMA), color=BLUE)
    ax.plot(stocks, bs.put_price(stocks, K, T, R, SIGMA), color=ORANGE)
    ax.axvline(K, color=MUTED, ls="--", lw=0.8)
    ax.text(150, bs.call_price(150, K, T, R, SIGMA) - 4, "Call", color=INK, ha="right")
    ax.text(52, bs.put_price(52, K, T, R, SIGMA) - 5, "Put", color=INK)
    ax.text(K + 1, 2, "Strike", color=MUTED)
    ax.set(xlabel="Stock price", ylabel="Option price",
           title="Option value vs. stock price (K=100, T=1y, r=5%, vol=20%)")
    ax.legend(["Call", "Put"], loc="upper left", bbox_to_anchor=(0.30, 1.0))
    save_fig(fig, "price_vs_stock.png")

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    series = {"Delta": bs.delta(stocks, K, T, R, SIGMA, "call"), "Gamma": bs.gamma(stocks, K, T, R, SIGMA),
              "Vega": bs.vega(stocks, K, T, R, SIGMA), "Theta (per year)": bs.theta(stocks, K, T, R, SIGMA, "call")}
    for ax, (name, y) in zip(axes.flat, series.items()):
        ax.plot(stocks, y, color=BLUE)
        ax.axvline(K, color=MUTED, ls="--", lw=0.8)
        ax.set(title=name, xlabel="Stock price")
    fig.suptitle("Call-option Greeks vs. stock price (K=100, T=1y, r=5%, vol=20%)", x=0.02, ha="left",
                 fontsize=11, fontweight="bold")
    save_fig(fig, "greeks_vs_stock.png")

    vols, times = np.linspace(0.05, 0.6, 100), np.linspace(0.02, 3, 100)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for T_, c in zip((0.25, 1.0, 2.0), (BLUE, ORANGE, AQUA)):
        axes[0].plot(vols * 100, bs.call_price(100, 100, T_, R, vols), color=c)
        axes[0].text(vols[-1] * 100, bs.call_price(100, 100, T_, R, vols[-1]), f" {T_:g}y", color=INK, va="center")
    for s_, c in zip((0.1, 0.2, 0.4), (BLUE, ORANGE, AQUA)):
        axes[1].plot(times, bs.call_price(100, 100, times, R, s_), color=c)
        axes[1].text(times[-1], bs.call_price(100, 100, times[-1], R, s_), f" {s_:.0%}", color=INK, va="center")
    axes[0].set(title="Price rises with volatility", xlabel="Volatility (%)", ylabel="At-the-money call price")
    axes[1].set(title="Price rises with time to expiry", xlabel="Years to expiry")
    for ax in axes:
        ax.margins(x=0.12)
    save_fig(fig, "price_vs_vol_and_time.png")


def method_comparison():
    rows = []
    for option_type in ("call", "put"):
        fn = bs.call_price if option_type == "call" else bs.put_price
        for strike in (90.0, 100.0, 110.0):
            exact = fn(S0, strike, T, R, SIGMA)
            tree = bt.binomial_tree_price(S0, strike, T, R, SIGMA, 2000, option_type)
            mcp, se = mc.monte_carlo_price(S0, strike, T, R, SIGMA, 1_000_000, option_type, seed=42)
            rows.append(dict(option=option_type, strike=strike, black_scholes=exact,
                             binomial_2000=tree, binomial_abs_err=abs(tree - exact),
                             monte_carlo_1m=mcp, mc_std_error=se, mc_abs_err=abs(mcp - exact),
                             mc_err_in_std_errors=abs(mcp - exact) / se))
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "pricing_methods_comparison.csv", index=False, float_format="%.6f")

    # Model-free identities, checked on random parameter sets
    rng = np.random.default_rng(0)
    parity_err = 0.0
    for _ in range(1000):
        s, k = rng.uniform(50, 150, 2)
        t, r, v = rng.uniform(0.05, 3), rng.uniform(0, 0.08), rng.uniform(0.05, 0.8)
        parity_err = max(parity_err, abs(bs.call_price(s, k, t, r, v) - bs.put_price(s, k, t, r, v)
                                         - (s - k * np.exp(-r * t))))
    h = 1e-3
    num_delta = (bs.call_price(S0 + h, K, T, R, SIGMA) - bs.call_price(S0 - h, K, T, R, SIGMA)) / (2 * h)
    record("pricing", put_call_parity_max_abs_error=parity_err,
           delta_analytic_vs_numeric_abs_diff=abs(num_delta - bs.delta(S0, K, T, R, SIGMA)),
           binomial_max_abs_err_2000_steps=df["binomial_abs_err"].max(),
           mc_max_err_in_std_errors=df["mc_err_in_std_errors"].max(),
           atm_call_price=bs.call_price(S0, K, T, R, SIGMA))
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    print("Step 1: pricing validation")
    charts()
    method_comparison()
