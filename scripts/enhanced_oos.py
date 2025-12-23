#!/usr/bin/env python3
"""Extended OOS analysis and robustness diagnostics for combined LS momentum + value sleeve."""

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_PARAMS = {
    "band_keep": 0.81393,
    "band_add": 0.90307,
    "sector_tol": 0.09218,
    "exit_cap_frac": 0.07257,
    "ls_turnover_budget": 0.32496,
    "overlay_band": 0.11293,
    "micro_add_frac": 0.01774,
}


@dataclass
class RunPaths:
    momentum: Path
    value: Path
    combined: Path


def run_command(cmd: Sequence[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    else:
        sys.stdout.write(proc.stdout)
        sys.stdout.flush()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def combine_timeseries(momentum_ts: Path, value_ts: Path, out_path: Path) -> pd.DataFrame:
    cmd = [
        sys.executable,
        "scripts/combine_ls_sleeves.py",
        "--momentum",
        str(momentum_ts),
        "--value",
        str(value_ts),
        "--out",
        str(out_path),
    ]
    run_command(cmd, Path.cwd())
    df = pd.read_csv(out_path, parse_dates=["month_end"]).set_index("month_end").sort_index()
    return df


def load_combined(base_dir: Path) -> pd.DataFrame:
    combo_path = base_dir / "combined_timeseries.csv"
    if not combo_path.exists():
        raise FileNotFoundError(f"Missing combined timeseries at {combo_path}")
    return pd.read_csv(combo_path, parse_dates=["month_end"]).set_index("month_end").sort_index()


def compute_metrics(series: pd.Series, rf: Optional[pd.Series] = None) -> Dict[str, float]:
    series = series.dropna()
    n = len(series)
    if n == 0:
        return {"months": 0, "mean": np.nan, "cagr": np.nan, "vol": np.nan, "sharpe": np.nan, "hit": np.nan, "max_dd": np.nan}
    months = n
    years = months / 12.0
    gross = (1.0 + series).prod()
    cagr = gross ** (1.0 / years) - 1.0 if years > 0 else np.nan
    vol = series.std(ddof=0) * math.sqrt(12.0)
    mean = series.mean()
    if rf is not None:
        rf_series = rf.reindex(series.index).fillna(0.0)
        excess = series - rf_series
        sharpe = excess.mean() / excess.std(ddof=0) * math.sqrt(12.0) if excess.std(ddof=0) > 0 else np.nan
    else:
        sharpe = mean / series.std(ddof=0) * math.sqrt(12.0) if series.std(ddof=0) > 0 else np.nan
    wealth = (1.0 + series).cumprod()
    max_dd = (wealth / wealth.cummax() - 1.0).min()
    hit = (series > 0).mean()
    return {
        "months": months,
        "mean": mean,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "hit": hit,
        "max_dd": max_dd,
        "skew": series.skew(),
        "kurt": series.kurtosis(),
    }


def walk_forward_metrics(df: pd.DataFrame, rf: pd.Series, start_year: int = 2016, end_year: int = 2025) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for test_year in range(start_year, end_year + 1):
        train_start = df.index.min()
        train_end = pd.Timestamp(f"{test_year-1}-12-31")
        train_mask = (df.index >= train_start) & (df.index <= train_end)
        test_mask = (df.index.year == test_year)
        if df.loc[train_mask].empty or df.loc[test_mask].empty:
            continue
        train_metrics = compute_metrics(df.loc[train_mask, "combo"], rf)
        test_metrics = compute_metrics(df.loc[test_mask, "combo"], rf)
        results.append(
            {
                "test_year": test_year,
                "train_months": train_metrics["months"],
                "train_cagr": train_metrics["cagr"],
                "train_sharpe": train_metrics["sharpe"],
                "test_months": test_metrics["months"],
                "test_cagr": test_metrics["cagr"],
                "test_sharpe": test_metrics["sharpe"],
                "test_hit": test_metrics["hit"],
                "test_max_dd": test_metrics["max_dd"],
            }
        )
    return results


def deflated_sharpe(sharpe: float, skew: float, kurt: float, n: int, trials: int) -> float:
    if not np.isfinite(sharpe) or n <= 1:
        return np.nan
    numerator = sharpe * math.sqrt(n - 1)
    denom_term = 1 - skew * sharpe + ((kurt - 1) / 4.0) * (sharpe ** 2)
    if denom_term <= 0:
        return np.nan
    sr_adjusted = numerator / math.sqrt(denom_term)
    from scipy.stats import norm
    alpha = 0.05
    z_alpha = norm.ppf(1 - alpha / (2 * trials))
    return sr_adjusted - z_alpha


def stationary_bootstrap_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    idx = []
    p = 1.0 / block_length
    while len(idx) < n:
        start = rng.integers(0, n)
        block = []
        while len(block) == 0 or rng.random() > p:
            block.append(start)
            start = (start + 1) % n
            if len(block) >= n:
                break
        idx.extend(block)
    return np.array(idx[:n])


def spa_pvalue(diff_matrix: np.ndarray, block_length: int = 12, bootstraps: int = 1000, seed: int = 123) -> float:
    if diff_matrix.size == 0:
        return np.nan
    n, k = diff_matrix.shape
    theta_hat = diff_matrix.mean(axis=0)
    max_stat = theta_hat.max()
    centered = diff_matrix - theta_hat
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(bootstraps):
        idx = stationary_bootstrap_indices(n, block_length, rng)
        boot = centered[idx, :]
        stat = boot.mean(axis=0).max()
        if stat >= max_stat:
            exceed += 1
    return (exceed + 1) / (bootstraps + 1)


def run_strategy(
    strategy: str,
    out_dir: Path,
    db_path: Path,
    cdi_path: Path,
    params: Dict[str, float],
    borrow: float,
    dispersion_gate: Optional[str] = None,
    dispersion_gate_window: Optional[int] = None,
) -> Path:
    ensure_dir(out_dir)
    base_cmd = [
        sys.executable,
        f"scripts/{'momentum' if strategy == 'momentum' else 'value'}_br.py",
        "--db-path",
        str(db_path),
        "--cdi-path",
        str(cdi_path),
        "--cohorts",
        "3",
        "--beta-overlay",
        "--apply-vol-target",
        "--band-keep",
        f"{params['band_keep']:.5f}",
        "--band-add",
        f"{params['band_add']:.5f}",
        "--sector-tol",
        f"{params['sector_tol']:.5f}",
        "--exit-cap-frac",
        f"{params['exit_cap_frac']:.5f}",
        "--ls-turnover-budget",
        f"{params['ls_turnover_budget']:.5f}",
        "--overlay-band",
        f"{params['overlay_band']:.5f}",
        "--short-borrow-annual",
        f"{borrow:.2f}",
        "--out-dir",
        str(out_dir),
    ]
    if strategy == "momentum":
        base_cmd.extend(["--micro-add-frac", f"{params['micro_add_frac']:.5f}"])
    if dispersion_gate and dispersion_gate != "none":
        base_cmd.extend(["--dispersion-gate", dispersion_gate])
    if dispersion_gate_window:
        base_cmd.extend(["--dispersion-gate-window", str(int(dispersion_gate_window))])
    run_command(base_cmd, Path.cwd())
    return out_dir / f"{'momentum' if strategy == 'momentum' else 'value'}_br_timeseries.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced OOS / robustness analysis for combined LS sleeve.")
    parser.add_argument("--db-path", required=True, type=Path)
    parser.add_argument("--cdi-path", required=True, type=Path)
    parser.add_argument("--base-borrow", type=float, default=0.08)
    parser.add_argument("--workdir", type=Path, default=Path("results/enhanced_oos"))
    parser.add_argument("--no-rerun-base", action="store_true", help="Reuse existing base run if available.")
    parser.add_argument("--dispersion-gate", type=str, default="none",
                        choices=["none", "quantile50", "quantile60", "quantile70", "quantile80", "quantile90"],
                        help="Dispersion gate setting applied to both momentum and value strategies.")
    args = parser.parse_args()

    workdir = ensure_dir(args.workdir)
    base_dir = ensure_dir(workdir / "base")
    base_mom_dir = ensure_dir(base_dir / "momentum")
    base_val_dir = ensure_dir(base_dir / "value")

    base_paths: Dict[str, RunPaths] = {}

    if args.no_rerun_base and (base_mom_dir / "momentum_br_timeseries.csv").exists() and (base_val_dir / "value_br_timeseries.csv").exists():
        print("Reusing existing base runs.")
    else:
        print("Running base momentum/value backtests ...")
        run_strategy(
            "momentum",
            base_mom_dir,
            args.db_path,
            args.cdi_path,
            DEFAULT_PARAMS,
            args.base_borrow,
            args.dispersion_gate,
            None,
        )
        run_strategy(
            "value",
            base_val_dir,
            args.db_path,
            args.cdi_path,
            DEFAULT_PARAMS,
            args.base_borrow,
            args.dispersion_gate,
            None,
        )

    base_combined_path = base_dir / "combined_timeseries.csv"
    combine_timeseries(
        base_mom_dir / "momentum_br_timeseries.csv",
        base_val_dir / "value_br_timeseries.csv",
        base_combined_path,
    )
    base_df = load_combined(base_dir)
    rf_series = base_df["cdi"]
    base_metrics_full = compute_metrics(base_df["combo"], rf_series)
    base_metrics_oos = compute_metrics(base_df.loc[base_df.index >= "2019-01-31", "combo"], rf_series)
    wf_metrics = walk_forward_metrics(base_df, rf_series, start_year=2016, end_year=2025)

    scenarios: List[Dict[str, object]] = []
    scenario_returns: List[pd.Series] = []
    gate_series: Dict[str, pd.Series] = {}
    base_gate_key = args.dispersion_gate if args.dispersion_gate != "none" else "none"
    gate_series[base_gate_key] = base_df["combo"].copy()

    stress_plan = [
        ("band_keep", 0.9),
        ("band_keep", 0.8),
        ("band_keep", 0.5),
        ("band_keep", min(0.99, DEFAULT_PARAMS["band_keep"] * 1.1)),
        ("ls_turnover_budget", 0.8),
        ("ls_turnover_budget", 1.2),
        ("ls_turnover_budget", 0.5),
        ("ls_turnover_budget", 1.5),
    ]

    for param, mult in stress_plan:
        params = DEFAULT_PARAMS.copy()
        if param == "band_keep":
            params["band_keep"] = max(0.1, min(0.99, DEFAULT_PARAMS["band_keep"] * mult))
            params["band_add"] = max(params["band_keep"] + 0.01, min(0.99, DEFAULT_PARAMS["band_add"] * mult))
        elif param == "band_add":
            params["band_add"] = max(0.5, min(0.99, DEFAULT_PARAMS["band_add"] * mult))
        elif param == "sector_tol":
            params["sector_tol"] = min(0.30, DEFAULT_PARAMS["sector_tol"] * mult)
        elif param == "ls_turnover_budget":
            params["ls_turnover_budget"] = max(0.05, min(0.60, DEFAULT_PARAMS["ls_turnover_budget"] * mult))
        else:
            params[param] = DEFAULT_PARAMS[param] * mult

        tag = f"{param}_{mult:.2f}".replace(".", "p")
        scen_dir = ensure_dir(workdir / f"stress_{tag}")
        mom_dir = ensure_dir(scen_dir / "momentum")
        val_dir = ensure_dir(scen_dir / "value")
        print(f"Running stress scenario {tag} ...")
        mom_path = run_strategy(
            "momentum", mom_dir, args.db_path, args.cdi_path, params, args.base_borrow, args.dispersion_gate, None
        )
        val_path = run_strategy(
            "value", val_dir, args.db_path, args.cdi_path, params, args.base_borrow, args.dispersion_gate, None
        )
        combo_path = scen_dir / "combined_timeseries.csv"
        df = combine_timeseries(mom_path, val_path, combo_path)
        scen_metrics = compute_metrics(df["combo"], df["cdi"])
        scen_metrics_oos = compute_metrics(df.loc[df.index >= "2019-01-31", "combo"], df["cdi"])
        scenarios.append(
            {
                "name": tag,
                "param": param,
                "value": mult,
                "full": scen_metrics,
                "oos": scen_metrics_oos,
                "dispersion_gate": args.dispersion_gate,
            }
        )
        scenario_returns.append(df["combo"])

    dispersion_variants = ["none", "quantile50", "quantile60", "quantile70", "quantile80", "quantile90"]
    for gate in dispersion_variants:
        if gate == args.dispersion_gate:
            continue
        params = DEFAULT_PARAMS.copy()
        tag = f"gate_{gate}"
        scen_dir = ensure_dir(workdir / f"stress_{tag}")
        mom_dir = ensure_dir(scen_dir / "momentum")
        val_dir = ensure_dir(scen_dir / "value")
        print(f"Running stress scenario {tag} ...")
        mom_path = run_strategy(
            "momentum",
            mom_dir,
            args.db_path,
            args.cdi_path,
            params,
            args.base_borrow,
            gate if gate != "none" else None,
            None,
        )
        val_path = run_strategy(
            "value",
            val_dir,
            args.db_path,
            args.cdi_path,
            params,
            args.base_borrow,
            gate if gate != "none" else None,
            None,
        )
        combo_path = scen_dir / "combined_timeseries.csv"
        df = combine_timeseries(mom_path, val_path, combo_path)
        scen_metrics = compute_metrics(df["combo"], df["cdi"])
        scen_metrics_oos = compute_metrics(df.loc[df.index >= "2019-01-31", "combo"], df["cdi"])
        scenarios.append(
            {
                "name": tag,
                "param": "dispersion_gate",
                "value": gate,
                "full": scen_metrics,
                "oos": scen_metrics_oos,
                "dispersion_gate": gate,
            }
        )
        scenario_returns.append(df["combo"])
        gate_series[gate] = df["combo"].copy()

    rolling_windows = [60, 120]
    rolling_gates = ["quantile70", "quantile80"]
    for gate in rolling_gates:
        for window in rolling_windows:
            if gate == args.dispersion_gate and window == getattr(args, "dispersion_gate_window", None):
                continue
            params = DEFAULT_PARAMS.copy()
            tag = f"gate_{gate}_win{window}"
            scen_dir = ensure_dir(workdir / f"stress_{tag}")
            mom_dir = ensure_dir(scen_dir / "momentum")
            val_dir = ensure_dir(scen_dir / "value")
            print(f"Running stress scenario {tag} ...")
            mom_path = run_strategy(
                "momentum",
                mom_dir,
                args.db_path,
                args.cdi_path,
                params,
                args.base_borrow,
                gate,
                window,
            )
            val_path = run_strategy(
                "value",
                val_dir,
                args.db_path,
                args.cdi_path,
                params,
                args.base_borrow,
                gate,
                window,
            )
            combo_path = scen_dir / "combined_timeseries.csv"
            df = combine_timeseries(mom_path, val_path, combo_path)
            scen_metrics = compute_metrics(df["combo"], df["cdi"])
            scen_metrics_oos = compute_metrics(df.loc[df.index >= "2019-01-31", "combo"], df["cdi"])
            scenarios.append(
                {
                    "name": tag,
                    "param": "dispersion_gate_rolling",
                    "value": {"gate": gate, "window": window},
                    "full": scen_metrics,
                    "oos": scen_metrics_oos,
                    "dispersion_gate": gate,
                    "dispersion_gate_window": window,
                }
            )
            scenario_returns.append(df["combo"])

    candidate_keys = [k for k in ["none", "quantile50", "quantile60", "quantile70", "quantile80", "quantile90"] if k in gate_series]
    default_gate = base_gate_key if base_gate_key in gate_series else (candidate_keys[0] if candidate_keys else None)
    if default_gate and len(candidate_keys) >= 2:
        wf_series = pd.Series(index=base_df.index, dtype=float)
        selection_schedule: List[Dict[str, object]] = []
        years = sorted(base_df.index.year.unique())
        for year in years:
            year_mask = base_df.index.year == year
            if not year_mask.any():
                continue
            train_mask = base_df.index < pd.Timestamp(f"{year}-01-01")
            chosen_gate = default_gate
            best_sharpe = -np.inf
            if train_mask.sum() >= 24:
                for gate in candidate_keys:
                    gate_series_data = gate_series.get(gate)
                    if gate_series_data is None:
                        continue
                    train_series = gate_series_data.loc[train_mask]
                    if train_series.dropna().shape[0] < 12:
                        continue
                    metrics = compute_metrics(train_series, rf_series)
                    sharpe_val = metrics.get("sharpe")
                    if sharpe_val is not None and np.isfinite(sharpe_val) and sharpe_val > best_sharpe:
                        best_sharpe = sharpe_val
                        chosen_gate = gate
            idx_year = base_df.index[year_mask]
            source_series = gate_series.get(chosen_gate, base_df["combo"])
            wf_series.loc[idx_year] = source_series.reindex(idx_year).values
            selection_schedule.append(
                {
                    "year": int(year),
                    "gate": chosen_gate,
                    "train_months": int(train_mask.sum()),
                    "train_sharpe": float(best_sharpe if np.isfinite(best_sharpe) else np.nan),
                }
            )
        wf_series = wf_series.fillna(base_df["combo"])
        wf_full = compute_metrics(wf_series, rf_series)
        wf_oos = compute_metrics(wf_series.loc[wf_series.index >= "2019-01-31"], rf_series)
        scenarios.append(
            {
                "name": "gate_walk_forward",
                "param": "dispersion_gate",
                "value": "walk_forward_sharpe",
                "full": wf_full,
                "oos": wf_oos,
                "selection": selection_schedule,
            }
        )
        scenario_returns.append(wf_series)

    base_returns = base_df["combo"]
    trials = 1 + len(scenario_returns)
    ds_full = deflated_sharpe(
        base_metrics_full["sharpe"],
        base_metrics_full["skew"],
        base_metrics_full["kurt"],
        base_metrics_full["months"],
        trials,
    )

    aligned_diffs = []
    aligned_index = base_returns.dropna().index
    for ret in scenario_returns:
        sr = ret.reindex(aligned_index).dropna()
        base_aligned = base_returns.reindex(sr.index)
        aligned_diffs.append((sr - base_aligned).values)
    diff_matrix = np.column_stack(aligned_diffs) if aligned_diffs else np.empty((0, 0))
    spa_p = spa_pvalue(diff_matrix) if diff_matrix.size else np.nan

    dispersion_analysis: Optional[Dict[str, object]] = None
    if {"mom_dispersion", "val_dispersion"}.issubset(base_df.columns):
        disp_series = base_df[["mom_dispersion", "val_dispersion"]].mean(axis=1)
        forward_returns = base_df["combo"].shift(-1)
        mask = disp_series.notna() & forward_returns.notna()
        if mask.any():
            disp_vals = disp_series.loc[mask]
            fwd_vals = forward_returns.loc[mask]
            try:
                from scipy.stats import linregress
            except ImportError:
                linregress = None  # type: ignore
            linreg = {}
            if linregress is not None:
                lr = linregress(disp_vals.values, fwd_vals.values)
                linreg = {
                    "slope": float(lr.slope),
                    "intercept": float(lr.intercept),
                    "rvalue": float(lr.rvalue),
                    "pvalue": float(lr.pvalue),
                    "stderr": float(lr.stderr),
                }
            bin_stats = []
            try:
                bins = pd.qcut(disp_vals, 5, duplicates="drop")
                grouped = fwd_vals.groupby(bins)
                for interval, idx in grouped.groups.items():
                    bin_stats.append(
                        {
                            "bin": str(interval),
                            "count": int(len(idx)),
                            "next_month_mean": float(fwd_vals.loc[idx].mean()),
                        }
                    )
            except ValueError:
                bin_stats = []
            dispersion_analysis = {
                "observations": int(mask.sum()),
                "linregress": linreg,
                "bin_stats": bin_stats,
            }

    report = {
        "base": {
            "full": base_metrics_full,
            "oos_2019_plus": base_metrics_oos,
            "deflated_sharpe": ds_full,
        },
        "walk_forward": wf_metrics,
        "stress_scenarios": scenarios,
        "spa_pvalue": spa_p,
        "trial_count": trials,
    }
    if dispersion_analysis is not None:
        report["dispersion_cross_section"] = dispersion_analysis

    report_path = workdir / "enhanced_oos_summary.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=lambda x: x if isinstance(x, (int, float, str)) else float(x))
    print(f"\nEnhanced OOS summary written to {report_path}")


if __name__ == "__main__":
    main()
