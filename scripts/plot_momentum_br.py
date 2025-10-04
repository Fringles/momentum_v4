#!/usr/bin/env python3
import argparse
import os
from typing import List

import numpy as np
import pandas as pd


def compute_net_series(ts: pd.DataFrame, label: str, turnover_label: str, bps: float) -> pd.Series:
    per_side = (bps / 10000.0) / 2.0
    return ts[label].astype(float) - per_side * ts[turnover_label].astype(float)


def cumcurve(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0)).cumprod()


def drawdown_curve(r: pd.Series) -> pd.Series:
    wealth = cumcurve(r)
    peak = wealth.cummax()
    return wealth / peak - 1.0


def rolling_capm_alpha(y: pd.Series, x: pd.Series, window: int = 36) -> pd.Series:
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
        if len(df) < window * 0.6:
            out.append(np.nan)
            continue
        X = sm.add_constant(df["x"].values)
        model = sm.OLS(df["y"].values, X)
        res = model.fit()
        out.append(res.params[0])
    return pd.Series(out, index=idx)


def main():
    p = argparse.ArgumentParser(description="Plots for Brazil Momentum results")
    p.add_argument("--results-path", default=os.path.join(os.getcwd(), "results", "momentum_br_timeseries.csv"))
    p.add_argument("--out-dir", default=os.path.join(os.getcwd(), "results"))
    p.add_argument("--cost-bps", nargs="*", type=int, default=[20, 50, 100])
    p.add_argument("--target-vol", type=float, default=0.10)
    p.add_argument("--vt-bps", type=int, default=50, help="Cost level to vol-target")
    p.add_argument("--rolling-window", type=int, default=12, help="Rolling window (months) for Sharpe/vol")
    p.add_argument("--excess-vs-cdi", action="store_true", help="Compute rolling Sharpe vs CDI excess")
    args = p.parse_args()

    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8")

    ts = pd.read_csv(args.results_path, parse_dates=["month_end"]).set_index("month_end").sort_index()

    # Compute net series for LS and D10 at requested costs
    net_series = {}
    for b in args.cost_bps:
        net_series[f"LS_net_{b}"] = compute_net_series(ts, "LS", "turnover_LS", b)
        net_series[f"D10_net_{b}"] = compute_net_series(ts, "D10", "turnover_D10", b)

    # Vol-target LS at vt-bps using 36m rolling vol of gross LS
    ls_gross = ts["LS"].astype(float)
    ls_roll_vol_ann = ls_gross.rolling(window=36, min_periods=12).std() * np.sqrt(12.0)
    scale = (args.target_vol / ls_roll_vol_ann).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    ls_net_vt = net_series[f"LS_net_{args.vt_bps}"].astype(float) * scale

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Cumulative returns plot
    fig, ax = plt.subplots(figsize=(10, 6))
    cum_d10 = cumcurve(ts["D10"]) 
    cum_ls = cumcurve(ts["LS"]) 
    ax.plot(cum_d10.index, cum_d10.values, label="D10 gross")
    ax.plot(cum_ls.index, cum_ls.values, label="LS gross (beta-neutral)")
    for b in args.cost_bps:
        ax.plot(cumcurve(net_series[f"LS_net_{b}"]))
    # Re-plot with labels
    for b in args.cost_bps:
        s = cumcurve(net_series[f"LS_net_{b}"])
        ax.plot(s.index, s.values, label=f"LS net {b}bps")
    # Vol-targeted
    s_vt = cumcurve(ls_net_vt)
    ax.plot(s_vt.index, s_vt.values, label=f"LS net {args.vt_bps}bps, VT 10%")
    # Benchmarks
    if "CDI" in ts.columns:
        s_cdi = cumcurve(ts["CDI"]) 
        ax.plot(s_cdi.index, s_cdi.values, label="CDI", linestyle="--", alpha=0.8)
    if "BOVA11" in ts.columns:
        s_bm = cumcurve(ts["BOVA11"]) 
        ax.plot(s_bm.index, s_bm.values, label="BOVA11", linestyle=":", alpha=0.8)
    ax.set_title("Cumulative Growth of 1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "plot_cum_returns.png"), dpi=150)
    plt.close(fig)

    # 2) Drawdowns (LS variants)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(drawdown_curve(ts["LS"]), label="LS gross")
    for b in args.cost_bps:
        ax.plot(drawdown_curve(net_series[f"LS_net_{b}"]), label=f"LS net {b}bps")
    ax.plot(drawdown_curve(ls_net_vt), label=f"LS net {args.vt_bps}bps, VT 10%")
    ax.set_title("Drawdowns (monthly)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "plot_drawdowns.png"), dpi=150)
    plt.close(fig)

    # 3) Rolling 36m CAPM alpha (bps/month) for LS net vt-bps and VT
    alpha_ls = rolling_capm_alpha(net_series[f"LS_net_{args.vt_bps}"], ts["BOVA11"], window=36) * 1e4
    alpha_vt = rolling_capm_alpha(ls_net_vt, ts["BOVA11"], window=36) * 1e4
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(alpha_ls.index, alpha_ls.values, label=f"LS net {args.vt_bps}bps")
    ax.plot(alpha_vt.index, alpha_vt.values, label=f"LS net {args.vt_bps}bps, VT 10%")
    ax.axhline(0, color='k', lw=1, alpha=0.6)
    ax.set_title("Rolling 36m CAPM Alpha (bps/month)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "plot_rolling_alpha.png"), dpi=150)
    plt.close(fig)

    # 4) Turnover time series for LS
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts.index, ts["turnover_LS"].values, label="LS turnover")
    ax.set_title("Monthly Turnover (LS)")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "plot_turnover_ls.png"), dpi=150)
    plt.close(fig)

    # 5) Cumulative vs Benchmarks (focused)
    fig, ax = plt.subplots(figsize=(10, 6))
    if f"LS_net_{args.vt_bps}" in net_series:
        ax.plot(cumcurve(net_series[f"LS_net_{args.vt_bps}"]), label=f"LS net {args.vt_bps}bps")
    ax.plot(cumcurve(ls_net_vt), label=f"LS net {args.vt_bps}bps, VT 10%")
    if "CDI" in ts.columns:
        ax.plot(cumcurve(ts["CDI"]), label="CDI", linestyle="--", alpha=0.8)
    if "BOVA11" in ts.columns:
        ax.plot(cumcurve(ts["BOVA11"]), label="BOVA11", linestyle=":", alpha=0.8)
    ax.set_title("Cumulative: Strategy vs CDI and BOVA11")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "plot_cum_vs_benchmarks.png"), dpi=150)
    plt.close(fig)

    # 6) Cumulative Excess vs CDI and vs BOVA11
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    if f"LS_net_{args.vt_bps}" in net_series:
        ex_cdi = net_series[f"LS_net_{args.vt_bps}"] - ts["CDI"]
        ex_bm = net_series[f"LS_net_{args.vt_bps}"] - ts["BOVA11"]
        ax1.plot(cumcurve(ex_cdi), label=f"LS net {args.vt_bps}bps - CDI")
        ax2.plot(cumcurve(ex_bm), label=f"LS net {args.vt_bps}bps - BOVA11")
    ex_cdi_vt = ls_net_vt - ts["CDI"]
    ex_bm_vt = ls_net_vt - ts["BOVA11"]
    ax1.plot(cumcurve(ex_cdi_vt), label="LS VT10 - CDI")
    ax2.plot(cumcurve(ex_bm_vt), label="LS VT10 - BOVA11")
    ax1.set_title("Cumulative Excess vs CDI")
    ax2.set_title("Cumulative Excess vs BOVA11")
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax1.legend(); ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "plot_cum_excess_vs_cdi_bm.png"), dpi=150)
    plt.close(fig)

    # 7) Rolling 12m Sharpe (annualized). Optionally vs CDI excess
    def rolling_sharpe(r: pd.Series, rf, window: int) -> pd.Series:
        x = r.astype(float).copy()
        if rf is not None:
            x = x.align(rf.astype(float), join="left")[0] - rf
        mu = x.rolling(window=window, min_periods=max(3, window // 3)).mean()
        sd = x.rolling(window=window, min_periods=max(3, window // 3)).std()
        out = np.where(sd.values > 0, (mu / sd * np.sqrt(12)).values, np.nan)
        return pd.Series(out, index=x.index)

    rf_series = ts["CDI"] if args.excess_vs_cdi and "CDI" in ts.columns else None
    sharpe_ls_net = rolling_sharpe(net_series[f"LS_net_{args.vt_bps}"], rf_series, args.rolling_window)
    sharpe_ls_vt = rolling_sharpe(ls_net_vt, rf_series, args.rolling_window)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sharpe_ls_net.index, sharpe_ls_net.values, label=f"LS net {args.vt_bps}bps")
    ax.plot(sharpe_ls_vt.index, sharpe_ls_vt.values, label=f"LS net {args.vt_bps}bps, VT 10%", alpha=0.8)
    ax.axhline(0, color='k', lw=1, alpha=0.6)
    ax.set_title(f"Rolling {args.rolling_window}m Sharpe" + (" (excess vs CDI)" if rf_series is not None else ""))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "plot_rolling_sharpe.png"), dpi=150)
    plt.close(fig)

    # 8) Rolling 12m volatility (annualized)
    def rolling_vol_ann(r: pd.Series, window: int) -> pd.Series:
        return r.astype(float).rolling(window=window, min_periods=max(3, window // 3)).std() * np.sqrt(12)

    vol_ls_net = rolling_vol_ann(net_series[f"LS_net_{args.vt_bps}"], args.rolling_window)
    vol_ls_vt = rolling_vol_ann(ls_net_vt, args.rolling_window)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(vol_ls_net.index, vol_ls_net.values, label=f"LS net {args.vt_bps}bps")
    ax.plot(vol_ls_vt.index, vol_ls_vt.values, label=f"LS net {args.vt_bps}bps, VT 10%", alpha=0.8)
    ax.set_title(f"Rolling {args.rolling_window}m Volatility (annualized)")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "plot_rolling_vol.png"), dpi=150)
    plt.close(fig)

    print("Saved plots:")
    for f in [
        "plot_cum_returns.png",
        "plot_drawdowns.png",
        "plot_rolling_alpha.png",
        "plot_turnover_ls.png",
        "plot_cum_vs_benchmarks.png",
        "plot_cum_excess_vs_cdi_bm.png",
        "plot_rolling_sharpe.png",
        "plot_rolling_vol.png",
    ]:
        print(os.path.join(args.out_dir, f))


if __name__ == "__main__":
    main()
