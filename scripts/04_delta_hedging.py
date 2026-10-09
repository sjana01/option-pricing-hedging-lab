"""
Step 4 - Delta hedging: where Black-Scholes breaks in practice.

Experiments (all: sell 1 at-the-money 60-trading-day call, hedge with Black-Scholes delta):
  A  rebalancing frequency   - hedge error shrinks like 1/sqrt(rebalances)
  B  volatility mismatch     - sell at a volatility above/below what is realised; compare with theory
  C  jumps                   - same total variance, but with jumps: fatter left tail
  D  transaction costs       - more rebalancing = less risk but more cost

Simulated paths are model-generated (GBM / Merton jump-diffusion), not market data.
Outputs: figures/hedge_*.png, tables/hedge_*.csv, tables/key_results.json (section 'hedging')
"""
import numpy as np
import pandas as pd

from _common import *  # noqa: F401,F403
import delta_hedging as dh
import jump_diffusion as jd

S0, K, R, SIGMA = 100.0, 100.0, 0.03, 0.20
DAYS = 60
T = DAYS / 252
STEPS_PER_DAY = 4
FINE = DAYS * STEPS_PER_DAY            # 240 fine steps; daily rebalancing = every 4th step
N_PATHS = 40_000
SEED = 2026
DAILY, WEEKLY, MONTHLY = 4, 20, 80     # subsample factors


def reb(f):
    return FINE // f


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


def exp_a(gbm):
    factors = [f for f in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 40, 48, 60, 80, 120) if FINE % f == 0]
    rows = []
    for f in factors:
        pnl, _ = dh.hedge_pnl(dh.subsample_paths(gbm, f), K, T, R, SIGMA)
        rows.append(dict(rebalances=reb(f), mean_pnl=pnl.mean(), std_pnl=pnl.std(), mean_se=se(pnl)))
    df = pd.DataFrame(rows).sort_values("rebalances")
    df.to_csv(TABLE_DIR / "hedge_A_rebalancing.csv", index=False, float_format="%.5f")
    slope, intercept = np.polyfit(np.log(df["rebalances"]), np.log(df["std_pnl"]), 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.plot(df["rebalances"], df["std_pnl"], color=BLUE, marker="o", ms=4, label="Simulated")
    ref = df["std_pnl"].iloc[-1] * np.sqrt(df["rebalances"].iloc[-1] / df["rebalances"])
    ax.plot(df["rebalances"], ref, color=ORANGE, ls="--", label="1/sqrt(n) reference")
    ax.set(xscale="log", yscale="log", xlabel="Rebalances over the option's life (log scale)",
           ylabel="Std. dev. of hedging P&L ($, log scale)", title=f"Hedge error vs. rebalancing (fitted slope {slope:.2f})")
    ax.legend()
    ax = axes[1]
    bins = np.linspace(-3, 3, 80)
    for f, c, name in ((MONTHLY, AQUA, "Monthly (3)"), (WEEKLY, ORANGE, "Weekly (12)"), (DAILY, BLUE, "Daily (60)")):
        pnl, _ = dh.hedge_pnl(dh.subsample_paths(gbm, f), K, T, R, SIGMA)
        ax.hist(pnl, bins=bins, density=True, histtype="step", lw=2, color=c, label=name)
    ax.set(xlabel="Hedging P&L per option sold ($, present value)", ylabel="Density",
           title="Distribution tightens as hedging gets more frequent")
    ax.legend(title="Rebalancing")
    save_fig(fig, "hedge_A_rebalancing.png")
    return df, slope


def exp_b(gbm):
    sigmas_h = np.round(np.arange(0.10, 0.301, 0.025), 3)
    p = dh.subsample_paths(gbm, DAILY)
    rows = []
    for sh in sigmas_h:
        pnl, _ = dh.hedge_pnl(p, K, T, R, sh)
        theory = dh.theoretical_pnl_attribution(gbm, K, T, R, sh, SIGMA)
        rows.append(dict(hedge_vol=sh, mean_pnl=pnl.mean(), se=se(pnl), std_pnl=pnl.std(), theory_mean=theory.mean()))
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "hedge_B_vol_mismatch.csv", index=False, float_format="%.5f")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(SIGMA * 100, color=MUTED, ls=":", lw=0.8)
    ax.errorbar(df["hedge_vol"] * 100, df["mean_pnl"], yerr=2 * df["se"], color=BLUE, marker="o", ms=5,
                capsize=3, lw=1.2, label="Simulated mean (+/- 2 std. errors)", zorder=3)
    ax.plot(df["hedge_vol"] * 100, df["theory_mean"], color=ORANGE, ls="--", lw=2.2,
            label="Theory (P&L attribution formula)", zorder=5)
    ax.text(SIGMA * 100 + 0.3, ax.get_ylim()[0] + 0.1, "realised volatility = 20%", color=MUTED)
    ax.set(xlabel="Volatility used to sell and hedge (%)", ylabel="Mean hedging P&L ($ per option sold)",
           title="Sell above realised volatility and you win; below and you lose")
    ax.legend(loc="upper left")
    save_fig(fig, "hedge_B_vol_mismatch.png")
    return df


def exp_c():
    sig_d, lam, mu_j, sig_j = 0.15, 0.5, -0.10, 0.15
    total = jd.total_volatility(sig_d, lam, mu_j, sig_j)
    gbm = jd.simulate_gbm_paths(S0, T, R, total, FINE, N_PATHS, seed=SEED + 1)
    jump = jd.simulate_merton_paths(S0, T, R, sig_d, lam, mu_j, sig_j, FINE, N_PATHS, seed=SEED + 1)
    out, pnls = [], {}
    for name, paths in (("Continuous (GBM)", gbm), ("With jumps (Merton)", jump)):
        pnl, _ = dh.hedge_pnl(dh.subsample_paths(paths, DAILY), K, T, R, total)
        pnls[name] = pnl
        out.append(dict(model=name, total_vol=total, mean=pnl.mean(), std=pnl.std(),
                        p1=np.percentile(pnl, 1), p5=np.percentile(pnl, 5), worst=pnl.min(),
                        skew=float(pd.Series(pnl).skew()), excess_kurtosis=float(pd.Series(pnl).kurt())))
    df = pd.DataFrame(out)
    df.to_csv(TABLE_DIR / "hedge_C_jumps.csv", index=False, float_format="%.4f")

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    bins = np.linspace(-6, 4, 110)
    for (name, pnl), c in zip(pnls.items(), (BLUE, ORANGE)):
        ax.hist(np.clip(pnl, bins[0], bins[-1] - 1e-9), bins=bins, density=True, histtype="step", lw=2,
                color=c, label=name)
    ax.set_yscale("log")
    ax.set(xlabel="Hedging P&L per option sold ($, PV; losses beyond -6 are stacked in the -6 bin)",
           ylabel="Density (log scale)",
           title="Same total volatility, very different tail risk (daily hedging)")
    ax.legend(loc="upper left")
    save_fig(fig, "hedge_C_jumps.png")
    return df, total


def exp_d(gbm):
    cost_rate = 0.0005   # 5 basis points of traded notional
    rows = []
    for f in (MONTHLY, 40, WEEKLY, 10, DAILY, 2, 1):
        pnl, cost = dh.hedge_pnl(dh.subsample_paths(gbm, f), K, T, R, SIGMA, cost_rate=cost_rate)
        rows.append(dict(rebalances=reb(f), mean_net_pnl=pnl.mean(), std_net_pnl=pnl.std(), mean_cost=cost.mean()))
    df = pd.DataFrame(rows).sort_values("rebalances")
    df.to_csv(TABLE_DIR / "hedge_D_costs.csv", index=False, float_format="%.5f")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    axes[0].plot(df["rebalances"], df["mean_cost"], color=ORANGE, marker="o", ms=5)
    axes[0].set(xscale="log", xlabel="Rebalances (log scale)", ylabel="Expected transaction cost ($)",
                title="Cost rises with more rebalancing")
    axes[1].plot(df["rebalances"], df["std_net_pnl"], color=BLUE, marker="o", ms=5)
    axes[1].set(xscale="log", xlabel="Rebalances (log scale)", ylabel="Std. dev. of net P&L ($)",
                title="Risk falls with more rebalancing")
    fig.suptitle("Transaction cost of 5 bps of traded notional", x=0.02, ha="left", fontsize=9, color=MUTED)
    save_fig(fig, "hedge_D_costs.png")
    return df


if __name__ == "__main__":
    print(f"Step 4: delta hedging ({N_PATHS:,} paths, {DAYS} trading days, seed {SEED})")
    gbm = jd.simulate_gbm_paths(S0, T, R, SIGMA, FINE, N_PATHS, seed=SEED)

    a, slope = exp_a(gbm)
    b = exp_b(gbm)
    c, total = exp_c()
    d = exp_d(gbm)

    pick = lambda df, n: df[df["rebalances"] == n].iloc[0]
    rb = lambda sh: b[np.isclose(b["hedge_vol"], sh)].iloc[0]
    gb, jp = c.iloc[0], c.iloc[1]
    record("hedging",
           config=dict(paths=N_PATHS, trading_days=DAYS, spot=S0, strike=K, rate=R, realised_vol=SIGMA, seed=SEED),
           std_monthly_3=float(pick(a, 3).std_pnl), std_weekly_12=float(pick(a, 12).std_pnl),
           std_daily_60=float(pick(a, 60).std_pnl), std_four_per_day_240=float(pick(a, 240).std_pnl),
           std_vs_rebalances_loglog_slope=float(slope),
           matched_vol_daily_mean_pnl=float(pick(a, 60).mean_pnl), matched_vol_daily_mean_se=float(pick(a, 60).mean_se),
           mismatch_sell_at_25=dict(sim=float(rb(0.25).mean_pnl), se=float(rb(0.25).se), theory=float(rb(0.25).theory_mean)),
           mismatch_sell_at_15=dict(sim=float(rb(0.15).mean_pnl), se=float(rb(0.15).se), theory=float(rb(0.15).theory_mean)),
           jumps=dict(total_vol=float(total), gbm_std=float(gb["std"]), jump_std=float(jp["std"]),
                      gbm_p1=float(gb.p1), jump_p1=float(jp.p1), gbm_worst=float(gb.worst), jump_worst=float(jp.worst),
                      jump_excess_kurtosis=float(jp.excess_kurtosis), gbm_excess_kurtosis=float(gb.excess_kurtosis)),
           costs_5bps=dict(daily_cost=float(pick(d, 60).mean_cost), weekly_cost=float(pick(d, 12).mean_cost),
                           daily_std=float(pick(d, 60).std_net_pnl), weekly_std=float(pick(d, 12).std_net_pnl)))
    print(a.round(4).to_string(index=False)); print(b.round(4).to_string(index=False))
    print(c.round(3).to_string(index=False)); print(d.round(4).to_string(index=False))
