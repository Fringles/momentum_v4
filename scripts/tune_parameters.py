#!/usr/bin/env python3
"""Parameter sweep and evaluation for the Brazil momentum strategy."""

import argparse
import json
import math
import os
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from momentum_br import (
    Config,
    config_from_args,
    make_parser as make_base_parser,
    ols_alpha_newey_west,
    run_analysis,
)


PARAM_RANGES = {
    "band_keep": (0.75, 0.85),
    "band_add": (0.88, 0.97),
    "sector_tol": (0.08, 0.12),
    "exit_cap_frac": (0.03, 0.07),
    "ls_turnover_budget": (0.25, 0.35),
    "micro_add_frac": (0.01, 0.03),
    "overlay_band": (0.08, 0.12),
}


def parse_args() -> Tuple[argparse.Namespace, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Parameter tuning helper for the momentum strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-end-date", default="2018-12-31", help="Inclusive YYYY-MM-DD for training backtest.")
    parser.add_argument(
        "--train-eval-start",
        default="2011-01-01",
        help="YYYY-MM-DD lower bound for training metrics (rows before are ignored when scoring).",
    )
    parser.add_argument(
        "--cv-start",
        default="2016-01-01",
        help="YYYY-MM-DD lower bound for cross-validation metrics (subset of training run).",
    )
    parser.add_argument(
        "--cv-end",
        default="2018-12-31",
        help="YYYY-MM-DD upper bound for cross-validation metrics.",
    )
    parser.add_argument("--oos-start-date", default="2019-01-01", help="Inclusive YYYY-MM-DD start for OOS run.")
    parser.add_argument(
        "--oos-end-date",
        default=None,
        help="Inclusive YYYY-MM-DD end for OOS run (default = latest available data).",
    )
    parser.add_argument("--samples", type=int, default=10, help="Number of parameter combinations to evaluate.")
    parser.add_argument("--seed", type=int, default=17, help="Seed for reproducible sampling.")
    parser.add_argument(
        "--output-root",
        default="results/tuning",
        help="Directory to store tuning runs (a timestamped subfolder is created inside).",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of top configs to evaluate out-of-sample.")
    parser.add_argument("--max-train-drawdown", type=float, default=0.15, help="Maximum acceptable |LS drawdown| in train.")
    parser.add_argument("--max-train-turnover", type=float, default=0.40, help="Maximum acceptable average LS turnover.")
    parser.add_argument("--min-alpha-t", type=float, default=2.0, help="Minimum t-stat for LS alpha after 50 bps costs.")
    parser.add_argument("--min-net-sharpe", type=float, default=0.0, help="Minimum Sharpe for LS net of 50 bps costs.")

    tuning_args, remaining = parser.parse_known_args()
    base_parser = make_base_parser()
    strategy_args = base_parser.parse_args(remaining)
    return tuning_args, strategy_args


def ensure_directory(path: str) -> str:
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def sample_parameters(rng: np.random.Generator) -> Dict[str, float]:
    params: Dict[str, float] = {}
    params["band_keep"] = float(rng.uniform(*PARAM_RANGES["band_keep"]))
    # ensure band_add stays above band_keep by at least 0.03
    for _ in range(10):
        candidate = float(rng.uniform(*PARAM_RANGES["band_add"]))
        if candidate > params["band_keep"] + 0.03:
            params["band_add"] = min(candidate, 0.99)
            break
    else:
        params["band_add"] = min(0.99, params["band_keep"] + 0.04)
    params["sector_tol"] = float(rng.uniform(*PARAM_RANGES["sector_tol"]))
    params["exit_cap_frac"] = float(rng.uniform(*PARAM_RANGES["exit_cap_frac"]))
    params["ls_turnover_budget"] = float(rng.uniform(*PARAM_RANGES["ls_turnover_budget"]))
    params["micro_add_frac"] = float(rng.uniform(*PARAM_RANGES["micro_add_frac"]))
    params["overlay_band"] = float(rng.uniform(*PARAM_RANGES["overlay_band"]))
    return params


def load_timeseries(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["month_end"])
    df = df.set_index("month_end").sort_index()
    return df


def compute_basic_metrics(returns: pd.Series) -> Dict[str, Any]:
    returns = returns.dropna()
    months = len(returns)
    if months == 0:
        return {
            "months": 0,
            "mean_monthly": math.nan,
            "t_stat": math.nan,
            "cagr": math.nan,
            "ann_vol": math.nan,
            "sharpe": math.nan,
            "max_drawdown": math.nan,
            "hit_rate": math.nan,
        }
    years = months / 12.0
    mean = float(returns.mean())
    std = float(returns.std(ddof=1)) if months > 1 else 0.0
    t_stat = float(mean / std * math.sqrt(months)) if months > 1 and std > 0 else math.nan
    ann_vol = float(std * math.sqrt(12.0)) if std > 0 else 0.0
    cum = float((1.0 + returns).prod())
    cagr = float(cum ** (1.0 / years) - 1.0) if years > 0 else math.nan
    sharpe = float(mean / std * math.sqrt(12.0)) if months > 1 and std > 0 else math.nan
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    max_dd = float(dd.min()) if not dd.empty else math.nan
    hit_rate = float((returns > 0).mean())
    return {
        "months": months,
        "mean_monthly": mean,
        "t_stat": t_stat,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
    }


def compute_cost_metrics(df: pd.DataFrame, cost_bps: float, lags: int) -> Dict[str, Any]:
    if "LS" not in df or "turnover_LS" not in df:
        return {}
    per_side_rate = (cost_bps / 10000.0) / 2.0
    net = df["LS"].astype(float) - per_side_rate * df["turnover_LS"].astype(float)
    net_metrics = compute_basic_metrics(net)
    bm = df.get("BOVA11", pd.Series(dtype=float))
    joined = pd.concat([net, bm], axis=1).dropna()
    if joined.empty:
        alpha_stats = {"alpha": math.nan, "alpha_t": math.nan, "beta": math.nan}
    else:
        alpha_stats = ols_alpha_newey_west(joined.iloc[:, 0], joined.iloc[:, 1], lags=lags)
    net_metrics.update(
        {
            "alpha": alpha_stats.get("alpha"),
            "alpha_t": alpha_stats.get("alpha_t"),
            "beta": alpha_stats.get("beta"),
        }
    )
    return net_metrics


def evaluate_period(df: pd.DataFrame, start: Optional[str], end: Optional[str], lags: int) -> Dict[str, Any]:
    sub = df.copy()
    if start:
        sub = sub[sub.index >= pd.to_datetime(start)]
    if end:
        sub = sub[sub.index <= pd.to_datetime(end)]
    raw_metrics = compute_basic_metrics(sub.get("LS", pd.Series(dtype=float)))
    cost_50 = compute_cost_metrics(sub, 50.0, lags)
    avg_turnover = float(sub.get("turnover_LS", pd.Series(dtype=float)).dropna().mean()) if "turnover_LS" in sub else math.nan
    return {
        "raw": raw_metrics,
        "cost_50": cost_50,
        "avg_turnover": avg_turnover,
    }


def to_native(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def build_config(base: Config, params: Dict[str, float], **kwargs: Any) -> Config:
    return replace(
        base,
        band_keep=params["band_keep"],
        band_add=params["band_add"],
        sector_tol=params["sector_tol"],
        exit_cap_frac=params["exit_cap_frac"],
        ls_turnover_budget=params["ls_turnover_budget"],
        micro_add_frac=params["micro_add_frac"],
        overlay_band=params["overlay_band"],
        **kwargs,
    )


def constraint_pass(summary: Dict[str, Any], limits: argparse.Namespace) -> Tuple[bool, Dict[str, bool]]:
    raw = summary["raw"]
    cost = summary["cost_50"]
    avg_turn = summary["avg_turnover"]
    checks = {
        "drawdown": abs(raw.get("max_drawdown", math.inf)) <= limits.max_train_drawdown,
        "turnover": avg_turn <= limits.max_train_turnover if not math.isnan(avg_turn) else False,
        "alpha_t": abs(cost.get("alpha_t", 0.0) or 0.0) >= limits.min_alpha_t,
        "net_sharpe": (cost.get("sharpe") or -math.inf) >= limits.min_net_sharpe,
    }
    return all(checks.values()), checks


def run():
    tuning_args, strategy_args = parse_args()
    rng = np.random.default_rng(tuning_args.seed)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    root_dir = ensure_directory(tuning_args.output_root)
    run_root = ensure_directory(os.path.join(root_dir, timestamp))

    base_cfg = config_from_args(strategy_args)

    results: List[Dict[str, Any]] = []

    for idx in range(1, tuning_args.samples + 1):
        params = sample_parameters(rng)
        run_id = f"run_{idx:02d}"
        run_dir = ensure_directory(os.path.join(run_root, run_id))
        train_dir = ensure_directory(os.path.join(run_dir, "train"))

        cfg_train = build_config(
            base_cfg,
            params,
            start_date=base_cfg.start_date,
            end_date=tuning_args.train_end_date,
            out_dir=train_dir,
        )

        print(f"\n[{run_id}] Training backtest with params: {params}")
        run_analysis(cfg_train)

        train_ts_path = os.path.join(train_dir, "momentum_br_timeseries.csv")
        if not os.path.exists(train_ts_path):
            print(f"[{run_id}] WARNING: missing timeseries output, skipping.")
            continue
        df_train = load_timeseries(train_ts_path)

        train_eval = evaluate_period(df_train, tuning_args.train_eval_start, tuning_args.train_end_date, cfg_train.nw_lags)
        cv_eval = evaluate_period(df_train, tuning_args.cv_start, tuning_args.cv_end, cfg_train.nw_lags)

        pass_flag, checks = constraint_pass(train_eval, tuning_args)

        entry: Dict[str, Any] = {
            "run_id": run_id,
            "params": params,
            "train": train_eval,
            "train_constraints": checks,
            "train_constraints_pass": pass_flag,
            "cv": cv_eval,
            "dirs": {"train": train_dir},
        }
        results.append(entry)

    # Select top configurations
    def alpha_key(item: Dict[str, Any]) -> float:
        cost = item["train"]["cost_50"]
        alpha = cost.get("alpha")
        return alpha if isinstance(alpha, (float, int)) else -math.inf

    valid = [r for r in results if r.get("train_constraints_pass")]
    valid_sorted = sorted(valid, key=alpha_key, reverse=True)
    top_runs = valid_sorted[: min(tuning_args.top_k, len(valid_sorted))]

    # Evaluate OOS for top runs
    for entry in top_runs:
        run_id = entry["run_id"]
        params = entry["params"]
        oos_dir = ensure_directory(os.path.join(run_root, run_id, "oos"))
        cfg_oos = build_config(
            base_cfg,
            params,
            start_date=tuning_args.oos_start_date,
            end_date=tuning_args.oos_end_date,
            out_dir=oos_dir,
        )
        print(f"\n[{run_id}] Out-of-sample backtest ...")
        run_analysis(cfg_oos)
        oos_ts_path = os.path.join(oos_dir, "momentum_br_timeseries.csv")
        if os.path.exists(oos_ts_path):
            df_oos = load_timeseries(oos_ts_path)
            entry["oos"] = evaluate_period(df_oos, tuning_args.oos_start_date, tuning_args.oos_end_date, cfg_oos.nw_lags)
        else:
            entry["oos"] = {"raw": {}, "cost_50": {}, "avg_turnover": math.nan}
        entry["dirs"]["oos"] = oos_dir

    summary_path = os.path.join(run_root, "tuning_results.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(to_native({"results": results, "top_runs": [r["run_id"] for r in top_runs]}), fh, indent=2)

    # Also write quick CSV overview
    rows = []
    for r in results:
        train_raw = r["train"]["raw"]
        train_cost = r["train"]["cost_50"]
        row = {
            "run_id": r["run_id"],
            "band_keep": r["params"]["band_keep"],
            "band_add": r["params"]["band_add"],
            "sector_tol": r["params"]["sector_tol"],
            "exit_cap_frac": r["params"]["exit_cap_frac"],
            "ls_turnover_budget": r["params"]["ls_turnover_budget"],
            "micro_add_frac": r["params"]["micro_add_frac"],
            "overlay_band": r["params"]["overlay_band"],
            "train_months": train_raw.get("months"),
            "train_mean": train_raw.get("mean_monthly"),
            "train_drawdown": train_raw.get("max_drawdown"),
            "train_turnover": r["train"]["avg_turnover"],
            "train_cost50_alpha": train_cost.get("alpha"),
            "train_cost50_alpha_t": train_cost.get("alpha_t"),
            "train_cost50_sharpe": train_cost.get("sharpe"),
            "constraints_pass": r.get("train_constraints_pass"),
        }
        if "oos" in r:
            oos_cost = r["oos"]["cost_50"]
            row.update(
                {
                    "oos_mean": r["oos"]["raw"].get("mean_monthly"),
                    "oos_drawdown": r["oos"]["raw"].get("max_drawdown"),
                    "oos_turnover": r["oos"]["avg_turnover"],
                    "oos_cost50_alpha": oos_cost.get("alpha"),
                    "oos_cost50_alpha_t": oos_cost.get("alpha_t"),
                    "oos_cost50_sharpe": oos_cost.get("sharpe"),
                }
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(run_root, "tuning_results.csv"), index=False)

    print(f"\nTuning complete. Summary written to {summary_path}")
    if top_runs:
        print("Top runs (by training 50 bps alpha):", ", ".join(r["run_id"] for r in top_runs))
    else:
        print("No configurations met the training constraints.")


if __name__ == "__main__":
    run()
