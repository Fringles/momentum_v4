#!/usr/bin/env python3
"""Combine momentum and value LS sleeves into an equal-weight composite with financing breakdown."""

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def load_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["month_end"])
    return df.set_index("month_end").sort_index()


def choose_series(df: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
    if primary in df.columns:
        return df[primary].astype(float)
    if fallback in df.columns:
        return df[fallback].astype(float)
    raise KeyError(f"Neither '{primary}' nor '{fallback}' found in columns: {list(df.columns)}")


def _choose_series(df: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
    if primary in df.columns:
        return df[primary].astype(float)
    if fallback in df.columns:
        return df[fallback].astype(float)
    return pd.Series(index=df.index, dtype=float)


def _compute_combo_overlay(
    ls_series: pd.Series,
    bm_series: pd.Series,
    *,
    lookback_months: int = 36,
    overlay_halflife_days: int = 60,
    overlay_band: float = 0.11293,
    overlay_shrink: float = 0.5,
) -> tuple[pd.Series, pd.Series]:
    if ls_series.empty:
        idx = bm_series.index if not bm_series.empty else pd.Index([])
        return pd.Series(0.0, index=idx), pd.Series(0.0, index=idx)

    idx = ls_series.index
    ls = ls_series.astype(float)
    bm = bm_series.reindex(idx).astype(float)
    if bm.replace({0.0: np.nan}).dropna().empty:
        zero = pd.Series(0.0, index=idx)
        return zero.copy(), zero.copy()

    hl_months = max(1, int(round(overlay_halflife_days / 21.0)))
    look = max(6, int(lookback_months))
    betas = []
    hedges = []
    h_prev = 0.0
    for i in range(len(idx)):
        start = max(0, i - look)
        df = pd.concat([ls.iloc[start:i], bm.iloc[start:i]], axis=1).dropna()
        if len(df) < 6:
            betas.append(0.0)
            hedges.append(h_prev)
            continue
        ages = np.arange(len(df))[::-1]
        decay = np.log(2.0) / float(hl_months)
        weights = np.exp(-decay * ages)
        weights = weights / weights.sum()
        x = df.iloc[:, 1]
        y = df.iloc[:, 0]
        x_c = x - float(np.dot(weights, x))
        y_c = y - float(np.dot(weights, y))
        varx = float(np.dot(weights, x_c ** 2))
        cov = float(np.dot(weights, x_c * y_c))
        beta = cov / varx if varx > 0 else 0.0
        beta_hat = (1.0 - overlay_shrink) * beta
        if abs(beta_hat) > overlay_band:
            h_t = -beta_hat
        else:
            h_t = h_prev
        betas.append(beta_hat)
        hedges.append(h_t)
        h_prev = h_t

    beta_series = pd.Series(betas, index=idx)
    hedge_series = pd.Series(hedges, index=idx)
    return beta_series, hedge_series


def main(
    momentum_path: Path,
    value_path: Path,
    output_path: Path,
    out_equity_curve: Optional[Path],
) -> None:
    mom = load_timeseries(momentum_path)
    val = load_timeseries(value_path)
    joined = mom.join(
        val,
        how="inner",
        lsuffix="_mom",
        rsuffix="_val",
    )
    if joined.empty:
        raise ValueError("Joined momentum/value timeseries is empty; confirm overlapping dates.")

    mom_raw = choose_series(mom, "LS", "LS_pre_vt")
    mom_net = choose_series(mom, "LS_net", "LS")
    val_raw = choose_series(val, "LS", "LS_pre_vt")
    val_net = choose_series(val, "LS_net", "LS")

    mom_pre_overlay = _choose_series(mom, "LS_pre_overlay", "LS")
    val_pre_overlay = _choose_series(val, "LS_pre_overlay", "LS")

    mom_turnover = mom.get("turnover_LS", pd.Series(index=mom.index, data=np.nan)).astype(float)
    val_turnover = val.get("turnover_LS", pd.Series(index=val.index, data=np.nan)).astype(float)

    mom_carry = mom.get("LS_carry", pd.Series(index=mom.index, data=0.0)).astype(float).fillna(0.0)
    mom_borrow = mom.get("LS_borrow_cost", pd.Series(index=mom.index, data=0.0)).astype(float).fillna(0.0)
    val_carry = val.get("LS_carry", pd.Series(index=val.index, data=0.0)).astype(float).fillna(0.0)
    val_borrow = val.get("LS_borrow_cost", pd.Series(index=val.index, data=0.0)).astype(float).fillna(0.0)

    mom_dispersion = mom.get("dispersion", pd.Series(index=mom.index, data=np.nan)).astype(float)
    val_dispersion = val.get("dispersion", pd.Series(index=val.index, data=np.nan)).astype(float)
    mom_gate = mom.get("dispersion_gate_active", pd.Series(index=mom.index, data=np.nan)).astype(float)
    val_gate = val.get("dispersion_gate_active", pd.Series(index=val.index, data=np.nan)).astype(float)

    mom_gross_long = mom.get("ls_gross_long", pd.Series(index=mom.index, data=np.nan)).astype(float)
    mom_gross_short = mom.get("ls_gross_short", pd.Series(index=mom.index, data=np.nan)).astype(float)
    val_gross_long = val.get("ls_gross_long", pd.Series(index=val.index, data=np.nan)).astype(float)
    val_gross_short = val.get("ls_gross_short", pd.Series(index=val.index, data=np.nan)).astype(float)

    cdi = mom.get("CDI", val.get("CDI", pd.Series(index=joined.index, data=np.nan))).astype(float)
    bova = mom.get("BOVA11", val.get("BOVA11", pd.Series(index=joined.index, data=np.nan))).astype(float)

    combo_pre_overlay = 0.5 * (
        mom_pre_overlay.loc[joined.index] + val_pre_overlay.loc[joined.index]
    )
    combo_beta_est, combo_hedge_ratio = _compute_combo_overlay(
        combo_pre_overlay,
        bova.loc[joined.index],
    )
    combo_raw = combo_pre_overlay + combo_hedge_ratio * bova.loc[joined.index]
    combo_carry = 0.5 * (mom_carry.loc[joined.index] + val_carry.loc[joined.index])
    combo_borrow = 0.5 * (mom_borrow.loc[joined.index] + val_borrow.loc[joined.index])
    combo_financing = combo_carry - combo_borrow
    combo_net = combo_raw + combo_financing
    combo_turnover = 0.5 * (
        mom_turnover.loc[joined.index].fillna(0.0) + val_turnover.loc[joined.index].fillna(0.0)
    )
    combo_gross_long = 0.5 * (
        mom_gross_long.loc[joined.index] + val_gross_long.loc[joined.index]
    )
    combo_gross_short = 0.5 * (
        mom_gross_short.loc[joined.index] + val_gross_short.loc[joined.index]
    )

    combo_gate = np.minimum(mom_gate.loc[joined.index], val_gate.loc[joined.index]) if not mom_gate.empty and not val_gate.empty else pd.Series(index=joined.index, data=np.nan)

    risk_cols = [
        "risk_ls_weighted_multiplier",
        "risk_ls_weighted_score",
        "risk_ls_flagged",
        "risk_ls_high",
        "risk_d10_weighted_multiplier",
        "risk_d10_weighted_score",
        "risk_d10_flagged",
        "risk_d10_high",
        "risk_universe_flagged",
        "risk_universe_high",
        "risk_universe_avg_multiplier",
    ]
    risk_data: Dict[str, np.ndarray] = {}
    for col in risk_cols:
        if col in mom.columns:
            risk_data[f"mom_{col}"] = mom[col].reindex(joined.index).astype(float).values
        if col in val.columns:
            risk_data[f"val_{col}"] = val[col].reindex(joined.index).astype(float).values

    def _concat_available(column: str) -> Optional[pd.DataFrame]:
        series_list = []
        if column in mom.columns:
            series_list.append(mom[column].reindex(joined.index))
        if column in val.columns:
            series_list.append(val[column].reindex(joined.index))
        if series_list:
            return pd.concat(series_list, axis=1)
        return None

    for column in ["risk_ls_weighted_multiplier", "risk_ls_weighted_score", "risk_d10_weighted_multiplier", "risk_d10_weighted_score", "risk_universe_avg_multiplier"]:
        df_series = _concat_available(column)
        if df_series is not None:
            risk_data[f"combo_{column}"] = df_series.mean(axis=1).values

    for column in ["risk_ls_flagged", "risk_ls_high", "risk_d10_flagged", "risk_d10_high", "risk_universe_flagged", "risk_universe_high"]:
        df_series = _concat_available(column)
        if df_series is not None:
            risk_data[f"combo_{column}"] = df_series.sum(axis=1, min_count=1).values

    out_dict = {
            "mom_raw": mom_raw.loc[joined.index].values,
            "mom_net": mom_net.loc[joined.index].values,
            "mom": mom_net.loc[joined.index].values,
            "mom_carry": mom_carry.loc[joined.index].values,
            "mom_borrow": mom_borrow.loc[joined.index].values,
            "mom_dispersion": mom_dispersion.loc[joined.index].values,
            "mom_gate_active": mom_gate.loc[joined.index].values,
            "mom_turnover": mom_turnover.loc[joined.index].values,
            "mom_gross_long": mom_gross_long.loc[joined.index].values,
            "mom_gross_short": mom_gross_short.loc[joined.index].values,
            "val_raw": val_raw.loc[joined.index].values,
            "val_net": val_net.loc[joined.index].values,
            "val": val_net.loc[joined.index].values,
            "val_carry": val_carry.loc[joined.index].values,
            "val_borrow": val_borrow.loc[joined.index].values,
            "val_dispersion": val_dispersion.loc[joined.index].values,
            "val_gate_active": val_gate.loc[joined.index].values,
            "val_turnover": val_turnover.loc[joined.index].values,
            "val_gross_long": val_gross_long.loc[joined.index].values,
            "val_gross_short": val_gross_short.loc[joined.index].values,
            "bova11": bova.loc[joined.index].values,
            "cdi": cdi.loc[joined.index].values,
            "combo_pre_overlay": combo_pre_overlay.values,
            "combo_beta_est": combo_beta_est.values,
            "combo_hedge_ratio": combo_hedge_ratio.values,
            "combo_raw": combo_raw.values,
            "combo_net": combo_net.values,
            "combo": combo_net.values,
            "combo_carry": combo_carry.values,
            "combo_borrow": combo_borrow.values,
            "combo_financing": combo_financing.values,
            "combo_turnover": combo_turnover.values,
            "combo_gross_long": combo_gross_long.values,
            "combo_gross_short": combo_gross_short.values,
            "combo_gate_active": combo_gate.values,
        }
    out_dict.update(risk_data)
    out = pd.DataFrame(out_dict, index=joined.index)
    out.index.name = "month_end"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path)

    if out_equity_curve is not None:
        eq = (1.0 + out["combo"]).cumprod()
        curve = pd.DataFrame({"month_end": out.index, "equity_curve": eq.values})
        out_equity_curve.parent.mkdir(parents=True, exist_ok=True)
        curve.to_csv(out_equity_curve, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine momentum and value LS sleeves with financing adjustments.")
    parser.add_argument("--momentum", required=True, type=Path, help="Path to momentum LS timeseries CSV")
    parser.add_argument("--value", required=True, type=Path, help="Path to value LS timeseries CSV")
    parser.add_argument("--out", required=True, type=Path, help="Destination CSV for combined series")
    parser.add_argument("--equity-curve", type=Path, default=None, help="Optional path for combined equity curve CSV")
    args = parser.parse_args()
    main(args.momentum, args.value, args.out, args.equity_curve)
