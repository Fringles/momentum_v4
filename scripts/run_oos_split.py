#!/usr/bin/env python3
"""Run momentum backtests on training and out-of-sample windows for comparison."""

import argparse
import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from momentum_br import config_from_args, make_parser as make_base_parser, run_analysis


def parse_args() -> Tuple[argparse.Namespace, argparse.Namespace]:
    """Parse high-level OOS wrapper arguments and underlying strategy args."""
    parser = argparse.ArgumentParser(
        description="Run training and out-of-sample momentum backtests with shared parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-end-date", required=True, help="Inclusive YYYY-MM-DD end date for training sample.")
    parser.add_argument("--oos-start-date", required=True, help="Inclusive YYYY-MM-DD start date for out-of-sample run.")
    parser.add_argument("--oos-end-date", default=None, help="Optional inclusive YYYY-MM-DD end date for out-of-sample run.")
    parser.add_argument(
        "--comparison-dir",
        default="results/oos_comparison",
        help="Root directory for train/ and oos/ backtest outputs plus comparison summary.",
    )
    parser.add_argument(
        "--skip-train-run",
        action="store_true",
        help="Reuse existing training outputs if present instead of running the backtest again.",
    )
    parser.add_argument(
        "--skip-oos-run",
        action="store_true",
        help="Reuse existing out-of-sample outputs if present instead of running the backtest again.",
    )
    oos_args, remaining = parser.parse_known_args()

    base_parser = make_base_parser()
    strategy_args = base_parser.parse_args(remaining)
    return oos_args, strategy_args


def ensure_directory(path: str) -> str:
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def compute_series_metrics(returns: pd.Series) -> Dict[str, Any]:
    returns = returns.dropna()
    months = len(returns)
    if months == 0:
        return {
            "months": 0,
            "mean_monthly": np.nan,
            "cagr": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "hit_rate": np.nan,
        }
    years = months / 12.0
    cum = float((1.0 + returns).prod())
    cagr = float(cum ** (1.0 / years) - 1.0) if years > 0 else np.nan
    vol = float(returns.std(ddof=1) * np.sqrt(12.0)) if months > 1 else np.nan
    mean_monthly = float(returns.mean())
    sharpe = float(mean_monthly / returns.std(ddof=1) * np.sqrt(12.0)) if months > 1 and returns.std(ddof=1) > 0 else np.nan
    dd = max_drawdown(returns)
    hit_rate = float((returns > 0).mean())
    t_stat = float(mean_monthly / returns.std(ddof=1) * np.sqrt(months)) if months > 1 and returns.std(ddof=1) > 0 else np.nan
    return {
        "months": months,
        "mean_monthly": mean_monthly,
        "t_stat": t_stat,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "hit_rate": hit_rate,
    }


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    return float(drawdown.min()) if not drawdown.empty else np.nan


def summarise_timeseries(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "period": {
            "start": df.index.min().strftime("%Y-%m-%d") if not df.empty else None,
            "end": df.index.max().strftime("%Y-%m-%d") if not df.empty else None,
            "months": int(len(df)),
        }
    }
    for col in ["LS", "LS_vt", "D10"]:
        if col in df.columns:
            summary[col] = compute_series_metrics(df[col])
    if "turnover_LS" in df.columns:
        summary["avg_turnover_LS"] = float(df["turnover_LS"].dropna().mean())
    if "turnover_D10" in df.columns:
        summary["avg_turnover_D10"] = float(df["turnover_D10"].dropna().mean())
    return summary


def load_timeseries(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Timeseries not found: {path}")
    df = pd.read_csv(path, parse_dates=["month_end"])
    df = df.set_index("month_end").sort_index()
    return df


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


def main() -> None:
    oos_args, strategy_args = parse_args()

    train_end = pd.to_datetime(oos_args.train_end_date)
    oos_start = pd.to_datetime(oos_args.oos_start_date)
    if oos_start <= train_end:
        raise SystemExit("Out-of-sample start date must be after the training end date.")

    comparison_root = ensure_directory(oos_args.comparison_dir)
    train_out_dir = ensure_directory(os.path.join(comparison_root, "train"))
    oos_out_dir = ensure_directory(os.path.join(comparison_root, "oos"))

    if not oos_args.skip_train_run:
        print(f"Running training backtest up to {train_end.date()} ...")
        train_cfg = config_from_args(strategy_args, end_date=oos_args.train_end_date, out_dir=train_out_dir)
        run_analysis(train_cfg)
    else:
        print("Skipping training backtest (reusing existing outputs).")

    if not oos_args.skip_oos_run:
        print(f"Running out-of-sample backtest from {oos_start.date()} ...")
        oos_cfg = config_from_args(
            strategy_args,
            start_date=oos_args.oos_start_date,
            end_date=oos_args.oos_end_date,
            out_dir=oos_out_dir,
        )
        run_analysis(oos_cfg)
    else:
        print("Skipping out-of-sample backtest (reusing existing outputs).")

    train_ts = load_timeseries(os.path.join(train_out_dir, "momentum_br_timeseries.csv"))
    oos_ts = load_timeseries(os.path.join(oos_out_dir, "momentum_br_timeseries.csv"))

    train_summary = summarise_timeseries(train_ts)
    oos_summary = summarise_timeseries(oos_ts)

    comparison: Dict[str, Any] = {"train": train_summary, "oos": oos_summary}

    if "LS" in train_summary and "LS" in oos_summary:
        comparison["delta_LS"] = {
            "cagr": oos_summary["LS"]["cagr"] - train_summary["LS"]["cagr"],
            "ann_vol": oos_summary["LS"]["ann_vol"] - train_summary["LS"]["ann_vol"],
            "mean_monthly": oos_summary["LS"]["mean_monthly"] - train_summary["LS"]["mean_monthly"],
        }

    summary_path = os.path.join(comparison_root, "oos_comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(to_native(comparison), fh, indent=2)

    print("\n=== Training Sample ===")
    print(json.dumps(to_native(train_summary), indent=2))
    print("\n=== Out-of-Sample Sample ===")
    print(json.dumps(to_native(oos_summary), indent=2))
    print(f"\nComparison summary written to {summary_path}")


if __name__ == "__main__":
    main()
