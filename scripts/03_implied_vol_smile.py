"""
Step 3 - Implied volatility and the volatility smile.

DATA SOURCE: by default the "market" option prices come from the Merton jump-diffusion model
(closed-form), because this build environment has no market-data access. Black-Scholes implied
volatility is then backed out contract by contract, exactly as it would be from quoted prices.
If data/real/implied_vols.csv exists (produced by scripts/real_data/fetch_option_chain.py) it is
plotted as well, with its own label.

Outputs: figures/vol_smile_model.png, vol_term_structure_model.png (+ *_market.png if real data)
         tables/model_implied_vols.csv, tables/key_results.json (section 'smile')
"""
import numpy as np
import pandas as pd

from _common import *  # noqa: F401,F403
import implied_vol as iv
import jump_diffusion as jd

S0, R = 100.0, 0.04
SIGMA, LAM, MU_J, SIGMA_J = 0.15, 0.5, -0.10, 0.15
EXPIRIES = {"1M": 1 / 12, "3M": 0.25, "6M": 0.5, "12M": 1.0}


def blue_ramp(n):
    """Dark -> light blue, one shade per expiry (expiry is ordered, so a single-hue ramp)."""
    dark, light = np.array([0x14, 0x44, 0x7F]), np.array([0xA9, 0xC9, 0xF3])
    return ["#%02x%02x%02x" % tuple((dark + (light - dark) * t).astype(int)) for t in np.linspace(0, 1, max(n, 2))[:n]]


def otm_only(df):
    """Keep out-of-the-money contracts only (puts below spot, calls at/above spot)."""
    if df["option_type"].nunique() < 2:
        return df
    keep = ((df["strike"] < df["underlying_price"]) & (df["option_type"] == "put")) | \
           ((df["strike"] >= df["underlying_price"]) & (df["option_type"] == "call"))
    return df[keep]


def iv_at(d, m):
    """Implied vol at moneyness m by linear interpolation; NaN if the chain does not span m."""
    d = d.sort_values("moneyness")
    if d["moneyness"].min() > m or d["moneyness"].max() < m:
        return np.nan
    return float(np.interp(m, d["moneyness"], d["implied_vol"]))


def build_model_chain():
    rows = []
    for label, T in EXPIRIES.items():
        for K in np.arange(80, 121, 1.0):
            # Standard convention: use the out-of-the-money option (puts below spot, calls above).
            kind = "put" if K < S0 else "call"
            price_fn = jd.merton_put_price if kind == "put" else jd.merton_call_price
            price = price_fn(S0, K, T, R, SIGMA, LAM, MU_J, SIGMA_J)
            vol = iv.implied_volatility(price, S0, K, T, R, kind)
            if vol is not None:
                rows.append(dict(expiry=label, T_years=T, option_type=kind, strike=K,
                                 underlying_price=S0, mid_price=price, implied_vol=vol, moneyness=K / S0))
    return pd.DataFrame(rows)


def plot_smile(df, source_label, fname):
    df = otm_only(df)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    labels = list(dict.fromkeys(df.sort_values("T_years")["expiry"]))
    colours = dict(zip(labels, blue_ramp(len(labels))))
    for lab in labels:
        d = df[df["expiry"] == lab].sort_values("moneyness")
        ax.plot(d["moneyness"] * 100, d["implied_vol"] * 100, color=colours[lab], label=lab)
    ref = labels[1] if len(labels) > 1 else labels[0]
    flat = iv_at(df[df["expiry"] == ref], 1.0)
    if not np.isnan(flat):
        ax.axhline(flat * 100, color=ORANGE, ls="--", lw=1.4,
                   label=f"Constant volatility ({ref} at-the-money level)")
    ax.axvline(100, color=MUTED, ls=":", lw=0.8)
    ax.set(xlabel="Strike as % of spot (puts below 100, calls above 100)",
           ylabel="Black-Scholes implied volatility (%)", title="Implied-volatility skew by expiry")
    ax.margins(x=0.03)
    ax.legend(loc="upper right", title="Expiry")
    fig.text(0.01, 0.005, f"Data: {source_label}", fontsize=8, color=MUTED, ha="left", va="bottom")
    save_fig(fig, fname)


def plot_term(df, source_label, fname):
    df = otm_only(df)
    rows = []
    for (label, T), d in df.groupby(["expiry", "T_years"]):
        rows.append(dict(expiry=label, T_years=T, atm=iv_at(d, 1.0),
                         skew=(iv_at(d, 0.90) - iv_at(d, 1.10)) * 100))
    t = pd.DataFrame(rows).sort_values("T_years")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ok = t.dropna(subset=["atm"])
    axes[0].plot(ok["T_years"] * 12, ok["atm"] * 100, color=BLUE, marker="o", ms=5)
    axes[0].set(title="At-the-money implied volatility", xlabel="Months to expiry", ylabel="Implied volatility (%)")
    ok = t.dropna(subset=["skew"])
    axes[1].plot(ok["T_years"] * 12, ok["skew"], color=BLUE, marker="o", ms=5)
    axes[1].set(title="Skew: IV(90% strike) minus IV(110% strike)", xlabel="Months to expiry",
                ylabel="Percentage points")
    fig.text(0.01, 0.005, f"Data: {source_label}", fontsize=8, color=MUTED, ha="left", va="bottom")
    save_fig(fig, fname)
    return t


if __name__ == "__main__":
    print("Step 3: implied-volatility smile (model-generated prices)")
    chain = build_model_chain()
    chain.to_csv(TABLE_DIR / "model_implied_vols.csv", index=False, float_format="%.6f")
    label = "model-generated (Merton jump-diffusion), NOT market data"
    plot_smile(chain, label, "vol_smile_model.png")
    term = plot_term(chain, label, "vol_term_structure_model.png")

    total = jd.total_volatility(SIGMA, LAM, MU_J, SIGMA_J)
    record("smile", data_source="Merton jump-diffusion (model-generated, not market data)",
           diffusion_vol=SIGMA, jump_intensity=LAM, jump_mean=MU_J, jump_std=SIGMA_J, total_vol=total,
           atm_iv_pct={r.expiry: float(r.atm * 100) for r in term.itertuples()},
           skew_pts_90_minus_110={r.expiry: float(r.skew) for r in term.itertuples()},
           contracts_inverted=int(len(chain)))
    shown = term.copy()
    shown["atm"] = (shown["atm"] * 100).round(2)
    shown["skew"] = shown["skew"].round(2)
    print(shown.to_string(index=False))

    real = DATA_DIR / "real" / "implied_vols.csv"
    if real.exists():
        mk = pd.read_csv(real)
        mk["expiry"] = mk["expiry"].astype(str)
        asof = mk["asof"].iloc[0] if "asof" in mk else "date unknown"
        sym = mk["ticker"].iloc[0] if "ticker" in mk else "market"
        plot_smile(mk, f"{sym} market data, {asof}", "vol_smile_market.png")
        plot_term(mk, f"{sym} market data, {asof}", "vol_term_structure_market.png")
        print(f"  market-data charts written for {sym} ({asof})")
    else:
        print("  (no data/real/implied_vols.csv found - market-data charts skipped)")
