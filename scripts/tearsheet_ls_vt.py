#!/usr/bin/env python3
import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd


def cumcurve(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0.0)).cumprod()


def drawdown_curve(r: pd.Series) -> pd.Series:
    wealth = cumcurve(r)
    peak = wealth.cummax()
    return wealth / peak - 1.0


def rolling_alpha(y: pd.Series, x: pd.Series, window: int = 36) -> pd.Series:
    import statsmodels.api as sm
    out = []
    idx = y.index
    for i in range(len(idx)):
        if i + 1 < window:
            out.append(np.nan)
            continue
        sl = slice(i + 1 - window, i + 1)
        df = pd.concat([y.iloc[sl], x.iloc[sl]], axis=1)
        df.columns = ["y", "x"]
        df = df.dropna()
        if len(df) < max(12, int(window * 0.6)):
            out.append(np.nan)
            continue
        X = sm.add_constant(df["x"].values)
        res = sm.OLS(df["y"].values, X).fit()
        out.append(res.params[0])
    return pd.Series(out, index=idx)


def rolling_sharpe(y: pd.Series, rf: Optional[pd.Series], window: int = 12) -> pd.Series:
    x = y.astype(float).copy()
    if rf is not None:
        x = x.align(rf.astype(float), join="left")[0] - rf
    mu = x.rolling(window=window, min_periods=max(3, window // 3)).mean()
    sd = x.rolling(window=window, min_periods=max(3, window // 3)).std()
    return (mu / sd * np.sqrt(12)).where(sd > 0)


def main():
    p = argparse.ArgumentParser(description="Tearsheet for LS_VT (vol-targeted) strategy")
    p.add_argument("--results-path", default=os.path.join(os.getcwd(), "results", "momentum_br_timeseries.csv"))
    p.add_argument("--out-dir", default=os.path.join(os.getcwd(), "results"))
    p.add_argument("--rolling-3m", type=int, default=3)
    p.add_argument("--vol-window", type=int, default=12)
    p.add_argument("--alpha-window", type=int, default=36)
    args = p.parse_args()

    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8")

    ts = pd.read_csv(args.results_path, parse_dates=["month_end"]).set_index("month_end").sort_index()

    # Prefer LS_vt if present; otherwise derive from vt_scale or fallback to LS
    if "LS_vt" in ts.columns:
        r = ts["LS_vt"].astype(float)
        vt_label = "LS_VT"
    elif "vt_scale" in ts.columns and "LS" in ts.columns:
        r = (ts["LS"].astype(float) * ts["vt_scale"].astype(float))
        vt_label = "LS * vt_scale"
    else:
        r = ts["LS"].astype(float)
        vt_label = "LS (no VT available)"

    # Benchmarks
    cdi = ts["CDI"].astype(float) if "CDI" in ts.columns else pd.Series(index=ts.index, dtype=float)
    bm = ts["BOVA11"].astype(float) if "BOVA11" in ts.columns else pd.Series(index=ts.index, dtype=float)

    os.makedirs(args.out_dir, exist_ok=True)

    # Compute series for plots
    cum_vt = cumcurve(r)
    cum_cdi = cumcurve(cdi) if not cdi.empty else None
    cum_bm = cumcurve(bm) if not bm.empty else None

    roll3 = (1.0 + r).rolling(window=args.rolling_3m, min_periods=args.rolling_3m).apply(np.prod, raw=True) - 1.0
    dd = drawdown_curve(r)
    vol12 = r.rolling(window=args.vol_window, min_periods=max(3, args.vol_window // 3)).std() * np.sqrt(12)
    alpha36 = rolling_alpha(r, bm, window=args.alpha_window) if not bm.empty else pd.Series(index=r.index, dtype=float)
    sharpe12 = rolling_sharpe(r, cdi, window=args.vol_window) if not cdi.empty else rolling_sharpe(r, None, window=args.vol_window)

    # Summary metrics
    years = len(r.dropna()) / 12.0 if len(r.dropna()) else 0.0
    cagr = (cum_vt.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    ann_vol = r.std(ddof=1) * np.sqrt(12) if r.std(ddof=1) > 0 else np.nan
    ex = (r - cdi).dropna() if not cdi.empty else r.copy()
    sharpe = np.sqrt(12) * ex.mean() / ex.std(ddof=1) if ex.std(ddof=1) > 0 else np.nan
    mdd = dd.min()

    # Multi-panel tearsheet
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]
    ax5, ax6 = axes[2]

    # Equity curve vs CDI & BOVA11
    ax1.plot(cum_vt.index, cum_vt.values, label=vt_label, color="#1f77b4")
    if cum_cdi is not None:
        ax1.plot(cum_cdi.index, cum_cdi.values, label="CDI", linestyle="--")
    if cum_bm is not None:
        ax1.plot(cum_bm.index, cum_bm.values, label="BOVA11", linestyle=":")
    ax1.set_title("Cumulative Growth of 1: LS_VT vs Benchmarks")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Rolling 3-month return
    ax2.plot(roll3.index, roll3.values, label=f"Rolling {args.rolling_3m}m Return")
    ax2.axhline(0, color="k", lw=1, alpha=0.6)
    ax2.set_title(f"Rolling {args.rolling_3m}-Month Return (compounded)")
    ax2.grid(True, alpha=0.3)

    # Drawdown chart
    ax3.plot(dd.index, dd.values, label="Drawdown", color="#d62728")
    ax3.set_title("Drawdown (monthly)")
    ax3.grid(True, alpha=0.3)

    # Rolling 12-month volatility
    ax4.plot(vol12.index, vol12.values, label=f"Rolling {args.vol_window}m Vol (ann)")
    ax4.set_title(f"Rolling {args.vol_window}-Month Volatility (annualized)")
    ax4.grid(True, alpha=0.3)

    # Rolling 36m CAPM alpha vs BOVA11 (bps/mo)
    if not alpha36.empty:
        ax5.plot(alpha36.index, (alpha36 * 1e4).values, label="Rolling 36m CAPM alpha", color="#2ca02c")
        ax5.axhline(0, color="k", lw=1, alpha=0.6)
        ax5.set_title("Rolling 36m CAPM Alpha vs BOVA11 (bps/month)")
        ax5.grid(True, alpha=0.3)
    else:
        ax5.axis('off')

    # Rolling 12m Sharpe (excess vs CDI if available)
    ax6.plot(sharpe12.index, sharpe12.values, label="Rolling Sharpe (12m)")
    ax6.axhline(0, color="k", lw=1, alpha=0.6)
    ax6.set_title("Rolling 12m Sharpe" + (" (excess vs CDI)" if not cdi.empty else ""))
    ax6.grid(True, alpha=0.3)

    # Super-title with headline metrics
    ttl = f"LS_VT Tearsheet — CAGR={cagr:.2%} | Vol={ann_vol:.2%} | Sharpe={sharpe:.2f} | MaxDD={mdd:.2%}"
    fig.suptitle(ttl, y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0.0, 1, 0.97])
    out_png = os.path.join(args.out_dir, "tearsheet_ls_vt.png")
    fig.savefig(out_png, dpi=150)
    try:
        fig.savefig(os.path.join(args.out_dir, "tearsheet_ls_vt.pdf"))
    except Exception:
        pass
    
    # Also save a small summary json
    summary = {
        "CAGR": float(cagr) if cagr == cagr else None,
        "AnnVol": float(ann_vol) if ann_vol == ann_vol else None,
        "Sharpe12": float(sharpe) if sharpe == sharpe else None,
        "MaxDD": float(mdd) if mdd == mdd else None,
        "Rolling3m_last": float(roll3.dropna().iloc[-1]) if len(roll3.dropna()) else None,
    }
    pd.Series(summary).to_json(os.path.join(args.out_dir, "tearsheet_ls_vt_summary.json"))


if __name__ == "__main__":
    main()

