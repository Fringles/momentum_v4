#!/usr/bin/env python3
"""Parameter sweep for combined LS Momentum + LS Value sleeves."""

import argparse
import json
import math
import os
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statistics import NormalDist

from momentum_br import (
    Config as MomConfig,
    config_from_args as mom_config_from_args,
    make_parser as make_mom_parser,
    run_analysis as run_momentum,
)
from value_br import (
    Config as ValConfig,
    config_from_args as val_config_from_args,
    make_parser as make_val_parser,
    run_analysis as run_value,
)


PARAM_RANGES = {
    "band_keep": (0.78, 0.84),
    "band_add": (0.90, 0.96),
    "sector_tol": (0.08, 0.11),
    "exit_cap_frac": (0.05, 0.08),
    "ls_turnover_budget": (0.24, 0.33),
    "micro_add_frac": (0.015, 0.035),
    "overlay_band": (0.09, 0.13),
}


def parse_args() -> Tuple[argparse.Namespace, argparse.Namespace, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Tune combined LS Momentum + LS Value parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-end-date", default="2018-12-31")
    parser.add_argument("--train-eval-start", default="2011-01-01")
    parser.add_argument("--cv-start", default="2016-01-01")
    parser.add_argument("--cv-end", default="2018-12-31")
    parser.add_argument("--oos-start-date", default="2019-01-01")
    parser.add_argument("--oos-end-date", default=None)
    parser.add_argument("--oos-report-start", default="2019-01-01",
                        help="Start date for reporting headline OOS metrics (kill rule still uses full OOS window).")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-root", default="results/tuning_combined")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-train-drawdown", type=float, default=0.15)
    parser.add_argument("--max-train-turnover", type=float, default=0.40)
    parser.add_argument("--min-alpha-t", type=float, default=2.0)
    parser.add_argument("--min-net-sharpe", type=float, default=0.0)
    tuning_args, remaining = parser.parse_known_args()

    mom_parser = make_mom_parser()
    val_parser = make_val_parser()
    mom_args = mom_parser.parse_args(remaining)
    val_args = val_parser.parse_args(remaining)
    return tuning_args, mom_args, val_args


def ensure_directory(path: str) -> str:
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def sample_parameters(rng: np.random.Generator) -> Dict[str, float]:
    params: Dict[str, float] = {}
    params["band_keep"] = float(rng.uniform(*PARAM_RANGES["band_keep"]))
    params["band_add"] = float(rng.uniform(max(params["band_keep"] + 0.02, PARAM_RANGES["band_add"][0]),
                                           PARAM_RANGES["band_add"][1]))
    params["sector_tol"] = float(rng.uniform(*PARAM_RANGES["sector_tol"]))
    params["exit_cap_frac"] = float(rng.uniform(*PARAM_RANGES["exit_cap_frac"]))
    params["ls_turnover_budget"] = float(rng.uniform(*PARAM_RANGES["ls_turnover_budget"]))
    params["micro_add_frac"] = float(rng.uniform(*PARAM_RANGES["micro_add_frac"]))
    params["overlay_band"] = float(rng.uniform(*PARAM_RANGES["overlay_band"]))
    return params


def load_timeseries(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["month_end"])
    return df.set_index("month_end").sort_index()


def compute_basic_metrics(series: pd.Series) -> Dict[str, Any]:
    s = pd.Series(series).dropna()
    months = len(s)
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
            "skew": math.nan,
            "kurt": math.nan,
        }
    years = months / 12.0
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if months > 1 else math.nan
    t_stat = float(mean / std * math.sqrt(months)) if months > 1 and std and std > 0 else math.nan
    ann_vol = float(std * math.sqrt(12.0)) if std and std > 0 else math.nan
    cum = float((1.0 + s).prod())
    cagr = float(cum ** (1.0 / years) - 1.0) if years > 0 else math.nan
    sharpe = float(mean / std * math.sqrt(12.0)) if std and std > 0 else math.nan
    wealth = (1.0 + s.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    max_dd = float(dd.min()) if not dd.empty else math.nan
    hit_rate = float((s > 0).mean())
    return {
        "months": months,
        "mean_monthly": mean,
        "t_stat": t_stat,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
        "skew": float(s.skew()) if months > 0 else math.nan,
        "kurt": float(s.kurtosis()) if months > 0 else math.nan,
    }


def concat_timeseries(train: pd.DataFrame, oos: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([train, oos]).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def build_combined_series(mom_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    idx = mom_df.index.union(val_df.index)
    mom_col = "LS_net" if "LS_net" in mom_df.columns else "LS"
    val_col = "LS_net" if "LS_net" in val_df.columns else "LS"
    mom_ret = mom_df.get(mom_col, pd.Series(dtype=float)).reindex(idx)
    val_ret = val_df.get(val_col, pd.Series(dtype=float)).reindex(idx)
    cdi = mom_df.get("CDI", pd.Series(dtype=float)).reindex(idx)
    if "CDI" in val_df.columns:
        cdi = cdi.where(~cdi.isna(), val_df["CDI"].reindex(idx))
    combo = 0.5 * (mom_ret + val_ret)
    df = pd.DataFrame({
        "combo": combo,
        "mom": mom_ret,
        "val": val_ret,
        "cdi": cdi,
    }).dropna(subset=["combo"])
    return df


def compute_series_stats(series: pd.Series, rf: Optional[pd.Series] = None) -> Dict[str, Any]:
    s = series.dropna()
    months = len(s)
    if months == 0:
        return {
            "months": 0,
            "mean": math.nan,
            "cagr": math.nan,
            "vol": math.nan,
            "sharpe": math.nan,
            "sharpe_excess": math.nan,
            "hit": math.nan,
            "max_dd": math.nan,
            "skew": math.nan,
            "kurt": math.nan,
        }
    years = months / 12.0
    cum = float((1.0 + s).prod())
    cagr = cum ** (1.0 / years) - 1.0 if years > 0 else math.nan
    vol = float(s.std(ddof=0) * math.sqrt(12.0)) if months > 1 else math.nan
    mean = float(s.mean())
    sharpe = mean / s.std(ddof=0) * math.sqrt(12.0) if months > 1 and s.std(ddof=0) > 0 else math.nan
    if rf is not None:
        rf_aligned = rf.reindex(s.index).fillna(0.0)
        excess = (s - rf_aligned).dropna()
        sharpe_ex = excess.mean() / excess.std(ddof=0) * math.sqrt(12.0) if len(excess) > 1 and excess.std(ddof=0) > 0 else math.nan
    else:
        sharpe_ex = math.nan
    wealth = (1.0 + s.fillna(0.0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    stats = {
        "months": months,
        "mean": mean,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sharpe_excess": sharpe_ex,
        "hit": float((s > 0).mean()),
        "max_dd": float(drawdown.min()),
        "skew": float(s.skew()),
        "kurt": float(s.kurtosis()),
    }
    return stats


def compute_deflated_sharpe(metric: Dict[str, Any], trials: int) -> float:
    sharpe = metric.get("sharpe_excess")
    if sharpe is None or not np.isfinite(sharpe):
        sharpe = metric.get("sharpe")
    skew = metric.get("skew")
    kurt = metric.get("kurt")
    n = metric.get("months", 0)
    if not (np.isfinite(sharpe) and np.isfinite(skew) and np.isfinite(kurt)) or n <= 1:
        return math.nan
    numerator = sharpe * math.sqrt(max(n - 1, 1))
    denom_term = 1 - skew * sharpe + ((kurt - 1) / 4.0) * (sharpe ** 2)
    if denom_term <= 0:
        return math.nan
    sr_adj = numerator / math.sqrt(denom_term)
    alpha = 0.05
    dist = NormalDist()
    z_alpha = dist.inv_cdf(1 - alpha / max(2 * trials, 2))
    return sr_adj - z_alpha


def compute_walk_forward(series_df: pd.DataFrame, start_year: int, end_year: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        train_mask = series_df.index.year < year
        test_mask = series_df.index.year == year
        if not train_mask.any() or not test_mask.any():
            continue
        train = series_df.loc[train_mask]
        test = series_df.loc[test_mask]
        if train.empty or test.empty:
            continue
        train_stats = compute_series_stats(train["combo"], train.get("cdi"))
        test_stats = compute_series_stats(test["combo"], test.get("cdi"))
        results.append(
            {
                "test_year": year,
                "train_months": train_stats["months"],
                "train_cagr": train_stats["cagr"],
                "train_sharpe": train_stats["sharpe_excess"],
                "test_months": test_stats["months"],
                "test_cagr": test_stats["cagr"],
                "test_sharpe": test_stats["sharpe_excess"],
                "test_hit": test_stats["hit"],
                "test_max_dd": test_stats["max_dd"],
            }
        )
    return results


def compute_cost_metrics(df: pd.DataFrame, cost_bps: float, lags: int) -> Dict[str, Any]:
    base_col = "LS_net" if "LS_net" in df.columns else "LS"
    if base_col not in df or "turnover_LS" not in df:
        return {}
    per_side_rate = (cost_bps / 10000.0) / 2.0
    net = df[base_col].astype(float) - per_side_rate * df["turnover_LS"].astype(float)
    metrics = compute_basic_metrics(net)
    bm = df.get("BOVA11", pd.Series(dtype=float))
    joined = pd.concat([net, bm], axis=1).dropna()
    if joined.empty:
        alpha_stats = {"alpha": math.nan, "alpha_t": math.nan, "beta": math.nan}
    else:
        from momentum_br import ols_alpha_newey_west

        alpha_stats = ols_alpha_newey_west(joined.iloc[:, 0], joined.iloc[:, 1], lags=lags)
    metrics.update(
        {
            "alpha": alpha_stats.get("alpha"),
            "alpha_t": alpha_stats.get("alpha_t"),
            "beta": alpha_stats.get("beta"),
        }
    )
    return metrics


def evaluate_combined(
    mom_df: pd.DataFrame,
    val_df: pd.DataFrame,
    start: Optional[str],
    end: Optional[str],
    lags: int,
    cost_bps: float,
) -> Dict[str, Any]:
    m = mom_df.copy()
    v = val_df.copy()
    if start:
        start_ts = pd.to_datetime(start)
        m = m[m.index >= start_ts]
        v = v[v.index >= start_ts]
    if end:
        end_ts = pd.to_datetime(end)
        m = m[m.index <= end_ts]
        v = v[v.index <= end_ts]
    mom_return_col = "LS_net" if "LS_net" in m.columns else "LS"
    val_return_col = "LS_net" if "LS_net" in v.columns else "LS"
    join = pd.concat(
        [
            m[mom_return_col],
            m.get("turnover_LS"),
            m.get("BOVA11"),
            v[val_return_col],
            v.get("turnover_LS"),
            m.get("LS_carry"),
            m.get("LS_borrow_cost"),
            v.get("LS_carry"),
            v.get("LS_borrow_cost"),
        ],
        axis=1,
        keys=[
            "mom_return",
            "mom_turnover",
            "BOVA11",
            "val_return",
            "val_turnover",
            "mom_carry",
            "mom_borrow",
            "val_carry",
            "val_borrow",
        ],
    ).dropna(subset=["mom_return", "val_return"])
    if join.empty:
        return {
            "combined": {"raw": compute_basic_metrics(pd.Series(dtype=float)), "cost": {}, "avg_turnover": math.nan},
            "momentum": {"raw": {}, "cost": {}},
            "value": {"raw": {}, "cost": {}},
        }
    for col in ["mom_turnover", "val_turnover", "mom_carry", "mom_borrow", "val_carry", "val_borrow"]:
        if col in join.columns:
            join[col] = join[col].fillna(0.0)
    join["combined_return"] = 0.5 * (join["mom_return"] + join["val_return"])
    join["combined_turnover"] = 0.5 * (join["mom_turnover"] + join["val_turnover"])
    join["combined_carry"] = 0.5 * (join.get("mom_carry", 0.0) + join.get("val_carry", 0.0))
    join["combined_borrow"] = 0.5 * (join.get("mom_borrow", 0.0) + join.get("val_borrow", 0.0))
    join["combined_financing"] = join["combined_carry"] - join["combined_borrow"]
    per_side_rate = (cost_bps / 10000.0) / 2.0
    join["mom_net"] = join["mom_return"] - per_side_rate * join["mom_turnover"]
    join["val_net"] = join["val_return"] - per_side_rate * join["val_turnover"]
    join["combined_net"] = 0.5 * (join["mom_net"] + join["val_net"])

    combined_raw = compute_basic_metrics(join["combined_return"])
    combined_cost = compute_basic_metrics(join["combined_net"])
    bm = join["BOVA11"].dropna()
    if not bm.empty:
        from momentum_br import ols_alpha_newey_west

        alpha_stats = ols_alpha_newey_west(join["combined_net"], bm, lags=lags)
    else:
        alpha_stats = {"alpha": math.nan, "alpha_t": math.nan, "beta": math.nan}
    combined_cost.update(
        {
            "alpha": alpha_stats.get("alpha"),
            "alpha_t": alpha_stats.get("alpha_t"),
            "beta": alpha_stats.get("beta"),
        }
    )
    momentum_eval = {
        "raw": compute_basic_metrics(join["mom_return"]),
        "cost": compute_cost_metrics(
            pd.concat([join["mom_return"], join["mom_turnover"], bm], axis=1, keys=["LS", "turnover_LS", "BOVA11"])
            .dropna(),
            cost_bps,
            lags,
        ),
    }
    value_eval = {
        "raw": compute_basic_metrics(join["val_return"]),
        "cost": compute_cost_metrics(
            pd.concat([join["val_return"], join["val_turnover"], bm], axis=1, keys=["LS", "turnover_LS", "BOVA11"])
            .dropna(),
            cost_bps,
            lags,
        ),
    }
    return {
        "combined": {
            "raw": combined_raw,
            "cost": combined_cost,
            "avg_turnover": float(join["combined_turnover"].mean()),
        },
        "momentum": momentum_eval,
        "value": value_eval,
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


def build_configs(
    mom_base: MomConfig,
    val_base: ValConfig,
    params: Dict[str, float],
    *,
    start_date: Optional[str],
    end_date: Optional[str],
    out_dir_mom: str,
    out_dir_val: str,
) -> Tuple[MomConfig, ValConfig]:
    mom_cfg = replace(
        mom_base,
        start_date=start_date if start_date is not None else mom_base.start_date,
        end_date=end_date if end_date is not None else mom_base.end_date,
        out_dir=out_dir_mom,
        band_keep=params["band_keep"],
        band_add=params["band_add"],
        sector_tol=params["sector_tol"],
        exit_cap_frac=params["exit_cap_frac"],
        ls_turnover_budget=params["ls_turnover_budget"],
        micro_add_frac=params["micro_add_frac"],
        overlay_band=params["overlay_band"],
    )
    val_cfg = replace(
        val_base,
        start_date=start_date if start_date is not None else val_base.start_date,
        end_date=end_date if end_date is not None else val_base.end_date,
        out_dir=out_dir_val,
        band_keep=params["band_keep"],
        band_add=params["band_add"],
        sector_tol=params["sector_tol"],
        exit_cap_frac=params["exit_cap_frac"],
        ls_turnover_budget=params["ls_turnover_budget"],
        micro_add_frac=params["micro_add_frac"],
        overlay_band=params["overlay_band"],
    )
    return mom_cfg, val_cfg


def constraint_pass(summary: Dict[str, Any], limits: argparse.Namespace) -> Tuple[bool, Dict[str, bool]]:
    raw = summary["combined"]["raw"]
    cost = summary["combined"]["cost"]
    avg_turn = summary["combined"]["avg_turnover"]
    checks = {
        "drawdown": abs(raw.get("max_drawdown", math.inf)) <= limits.max_train_drawdown,
        "turnover": avg_turn <= limits.max_train_turnover if not math.isnan(avg_turn) else False,
        "alpha_t": abs(cost.get("alpha_t", 0.0) or 0.0) >= limits.min_alpha_t,
        "net_sharpe": (cost.get("sharpe") or -math.inf) >= limits.min_net_sharpe,
    }
    return all(checks.values()), checks


def main() -> None:
    tuning_args, mom_args, val_args = parse_args()
    rng = np.random.default_rng(tuning_args.seed)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    root_dir = ensure_directory(tuning_args.output_root)
    run_root = ensure_directory(os.path.join(root_dir, timestamp))

    mom_base = mom_config_from_args(mom_args)
    val_base = val_config_from_args(val_args)

    results: List[Dict[str, Any]] = []

    for idx in range(1, tuning_args.samples + 1):
        params = sample_parameters(rng)
        run_id = f"run_{idx:02d}"
        run_dir = ensure_directory(os.path.join(run_root, run_id))
        train_mom_dir = ensure_directory(os.path.join(run_dir, "train", "momentum"))
        train_val_dir = ensure_directory(os.path.join(run_dir, "train", "value"))

        mom_cfg, val_cfg = build_configs(
            mom_base,
            val_base,
            params,
            start_date=mom_base.start_date,
            end_date=tuning_args.train_end_date,
            out_dir_mom=train_mom_dir,
            out_dir_val=train_val_dir,
        )

        print(f"\n[{run_id}] Training momentum backtest ...")
        run_momentum(mom_cfg)
        print(f"[{run_id}] Training value backtest ...")
        run_value(val_cfg)

        mom_ts_path = os.path.join(train_mom_dir, "momentum_br_timeseries.csv")
        val_ts_path = os.path.join(train_val_dir, "value_br_timeseries.csv")
        if not os.path.exists(mom_ts_path) or not os.path.exists(val_ts_path):
            print(f"[{run_id}] WARNING: missing timeseries; skipping.")
            continue

        mom_df = load_timeseries(mom_ts_path)
        val_df = load_timeseries(val_ts_path)

        train_eval = evaluate_combined(
            mom_df,
            val_df,
            tuning_args.train_eval_start,
            tuning_args.train_end_date,
            mom_cfg.nw_lags,
            50.0,
        )
        cv_eval = evaluate_combined(
            mom_df,
            val_df,
            tuning_args.cv_start,
            tuning_args.cv_end,
            mom_cfg.nw_lags,
            50.0,
        )

        pass_flag, checks = constraint_pass(train_eval, tuning_args)

        entry: Dict[str, Any] = {
            "run_id": run_id,
            "params": params,
            "train": train_eval,
            "cv": cv_eval,
            "train_constraints": checks,
            "train_constraints_pass": pass_flag,
            "dirs": {"train_momentum": train_mom_dir, "train_value": train_val_dir},
        }
        results.append(entry)

    def combined_alpha_key(item: Dict[str, Any]) -> float:
        alpha = item["train"]["combined"]["cost"].get("alpha")
        return alpha if isinstance(alpha, (float, int)) else -math.inf

    valid = [r for r in results if r.get("train_constraints_pass")]
    valid_sorted = sorted(valid, key=combined_alpha_key, reverse=True)
    top_runs = valid_sorted[: min(tuning_args.top_k, len(valid_sorted))]

    for entry in top_runs:
        run_id = entry["run_id"]
        params = entry["params"]
        oos_mom_dir = ensure_directory(os.path.join(run_root, run_id, "oos", "momentum"))
        oos_val_dir = ensure_directory(os.path.join(run_root, run_id, "oos", "value"))

        mom_cfg, val_cfg = build_configs(
            mom_base,
            val_base,
            params,
            start_date=tuning_args.oos_start_date,
            end_date=tuning_args.oos_end_date,
            out_dir_mom=oos_mom_dir,
            out_dir_val=oos_val_dir,
        )

        print(f"\n[{run_id}] OOS momentum backtest ...")
        run_momentum(mom_cfg)
        print(f"[{run_id}] OOS value backtest ...")
        run_value(val_cfg)

        mom_ts_path = os.path.join(oos_mom_dir, "momentum_br_timeseries.csv")
        val_ts_path = os.path.join(oos_val_dir, "value_br_timeseries.csv")
        if os.path.exists(mom_ts_path) and os.path.exists(val_ts_path):
            mom_df = load_timeseries(mom_ts_path)
            val_df = load_timeseries(val_ts_path)
            entry["oos"] = {
                "full": evaluate_combined(
                    mom_df,
                    val_df,
                    tuning_args.oos_start_date,
                    tuning_args.oos_end_date,
                    mom_cfg.nw_lags,
                    50.0,
                ),
                "report": evaluate_combined(
                    mom_df,
                    val_df,
                    tuning_args.oos_report_start,
                    tuning_args.oos_end_date,
                    mom_cfg.nw_lags,
                    50.0,
                ),
            }
            train_mom_df = load_timeseries(os.path.join(entry["dirs"]["train_momentum"], "momentum_br_timeseries.csv"))
            train_val_df = load_timeseries(os.path.join(entry["dirs"]["train_value"], "value_br_timeseries.csv"))
            full_mom_df = concat_timeseries(train_mom_df, mom_df)
            full_val_df = concat_timeseries(train_val_df, val_df)
            full_series = build_combined_series(full_mom_df, full_val_df)
            oos_series = build_combined_series(mom_df, val_df)
            oos_2019_series = full_series.loc[full_series.index >= pd.to_datetime("2019-01-01")]
            analytics = {
                "full": compute_series_stats(full_series["combo"], full_series.get("cdi")),
                "oos_full": compute_series_stats(oos_series["combo"], oos_series.get("cdi")),
                "oos_2019_plus": compute_series_stats(oos_2019_series["combo"], oos_2019_series.get("cdi")) if not oos_2019_series.empty else compute_series_stats(pd.Series(dtype=float)),
            }
            analytics["deflated_sharpe"] = compute_deflated_sharpe(analytics["full"], tuning_args.samples)
            start_year = max(full_series.index.min().year + 1, 2016) if not full_series.empty else 2016
            analytics["walk_forward"] = compute_walk_forward(full_series, start_year, full_series.index.max().year if not full_series.empty else start_year)
            entry["analytics"] = analytics
            combined_ts_path = os.path.join(run_root, run_id, "combined_train_oos_timeseries.csv")
            full_series.reset_index().to_csv(combined_ts_path, index=False)
            entry["dirs"]["combined_series"] = combined_ts_path
        entry["dirs"]["oos_momentum"] = oos_mom_dir
        entry["dirs"]["oos_value"] = oos_val_dir

    summary = {"results": results, "top_runs": [r["run_id"] for r in top_runs]}
    summary_path = os.path.join(run_root, "tuning_results.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(to_native(summary), fh, indent=2)

    rows = []
    for r in results:
        train_combined = r["train"]["combined"]
        row = {
            "run_id": r["run_id"],
            **{k: r["params"][k] for k in r["params"]},
            "train_mean": train_combined["raw"].get("mean_monthly"),
            "train_drawdown": train_combined["raw"].get("max_drawdown"),
            "train_turnover": train_combined.get("avg_turnover"),
            "train_alpha": train_combined["cost"].get("alpha"),
            "train_alpha_t": train_combined["cost"].get("alpha_t"),
            "train_sharpe": train_combined["cost"].get("sharpe"),
            "constraints_pass": r.get("train_constraints_pass"),
        }
        analytics = r.get("analytics", {})
        full_stats = analytics.get("full", {})
        oos2019_stats = analytics.get("oos_2019_plus", {})
        row.update(
            {
                "full_sharpe_excess": full_stats.get("sharpe_excess"),
                "full_cagr": full_stats.get("cagr"),
                "full_deflated_sharpe": analytics.get("deflated_sharpe"),
                "oos2019_sharpe": oos2019_stats.get("sharpe_excess"),
                "oos2019_cagr": oos2019_stats.get("cagr"),
            }
        )
        if "oos" in r:
            oos_report = r["oos"]["report"]["combined"]
            row.update(
                {
                    "oos_mean": oos_report["raw"].get("mean_monthly"),
                    "oos_drawdown": oos_report["raw"].get("max_drawdown"),
                    "oos_turnover": oos_report.get("avg_turnover"),
                    "oos_alpha": oos_report["cost"].get("alpha"),
                    "oos_alpha_t": oos_report["cost"].get("alpha_t"),
                    "oos_sharpe": oos_report["cost"].get("sharpe"),
                }
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(run_root, "tuning_results.csv"), index=False)

    print(f"\nCombined tuning complete. Summary written to {summary_path}")
    if top_runs:
        print("Top runs (train 50 bps alpha):", ", ".join(r["run_id"] for r in top_runs))
    else:
        print("No configurations met the training constraints.")


if __name__ == "__main__":
    main()
