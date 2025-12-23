#!/usr/bin/env python3
"""Shared helpers for lagged fundamental metrics."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sqlite3
import os


@dataclass
class LaggedMetric:
    """Container for lagged fundamental time series."""

    values: pd.DataFrame
    source_dates: pd.DataFrame


def read_fundamental_metric(cfg, metric_code: str) -> pd.DataFrame:
    """Return tidy fundamentals dataframe [date, ticker, value] for the given metric code."""
    assert os.path.exists(cfg.db_path), f"Database not found: {cfg.db_path}"
    start_pad = None
    if getattr(cfg, "start_date", None):
        start_pad = (pd.Timestamp(cfg.start_date) - pd.DateOffset(months=24)).strftime("%Y-%m-%d")
    end_pad = None
    if getattr(cfg, "end_date", None):
        end_pad = (pd.Timestamp(cfg.end_date) + pd.DateOffset(months=6)).strftime("%Y-%m-%d")

    with sqlite3.connect(cfg.db_path) as con:
        q = """
        SELECT
            f.period_date AS date,
            s.ticker AS ticker,
            f.value AS value
        FROM fundamentals f
        JOIN fundamental_metrics m ON m.metric_id = f.metric_id
        JOIN securities s ON s.security_id = f.security_id
        WHERE m.metric_code = ?
          AND s.security_type = 'Stock'
          AND s.country = 'BR'
          AND f.value IS NOT NULL
        """
        params: List[object] = [metric_code]
        if start_pad:
            q += " AND f.period_date >= ?"
            params.append(start_pad)
        if end_pad:
            q += " AND f.period_date <= ?"
            params.append(end_pad)
        q += " ORDER BY f.period_date ASC"
        df = pd.read_sql_query(q, con, params=params, parse_dates=["date"])  # type: ignore

    if df.empty:
        return df

    df["ticker"] = df["ticker"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "value"])
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
    df = df.sort_values(["date", "ticker"])
    return df


def build_lagged_metric(
    cfg,
    metric_code: str,
    calendar_month_index: pd.DatetimeIndex,
    lag_months: int = 3,
) -> LaggedMetric:
    """Generate a month-end series for the given metric applying a reporting lag."""
    raw = read_fundamental_metric(cfg, metric_code)
    if raw.empty:
        empty = pd.DataFrame(index=calendar_month_index)
        return LaggedMetric(values=empty, source_dates=empty.copy())

    pivot = (
        raw.pivot(index="date", columns="ticker", values="value")
        .sort_index()
        .astype(float)
    )

    # Shift release dates forward by lag months to mimic reporting delay
    lag_months = max(0, int(lag_months or 0))
    lag_offset = pd.offsets.MonthEnd(lag_months)

    shifted = pivot.copy()
    if lag_months > 0:
        shifted.index = shifted.index + lag_offset
    else:
        shifted.index = shifted.index + pd.offsets.MonthEnd(0)

    values = shifted.reindex(calendar_month_index).ffill()

    src_arr = np.tile(pivot.index.to_numpy()[:, None], (1, pivot.shape[1]))
    source = pd.DataFrame(src_arr, index=pivot.index, columns=pivot.columns)
    if lag_months > 0:
        source.index = source.index + lag_offset
    else:
        source.index = source.index + pd.offsets.MonthEnd(0)
    source = source.reindex(calendar_month_index).ffill()

    return LaggedMetric(values=values, source_dates=source)


def build_lagged_metrics(
    cfg,
    metric_codes: Sequence[str],
    calendar_month_index: pd.DatetimeIndex,
    lag_months: int = 3,
) -> Dict[str, LaggedMetric]:
    """Convenience wrapper to compute lagged series for multiple metric codes."""
    out: Dict[str, LaggedMetric] = {}
    for code in metric_codes:
        out[code] = build_lagged_metric(cfg, code, calendar_month_index, lag_months=lag_months)
    return out


def compute_staleness_months(source_dates: pd.DataFrame) -> pd.DataFrame:
    """Return staleness in months for each month/ticker based on source dates."""
    if source_dates.empty:
        return pd.DataFrame(index=source_dates.index, columns=source_dates.columns)

    idx_series = pd.Series(source_dates.index, index=source_dates.index)

    def _months(row: pd.Series) -> pd.Series:
        ref = idx_series.loc[row.name]
        return row.apply(lambda d: ((ref.year - d.year) * 12 + (ref.month - d.month)) if pd.notna(d) else np.nan)

    return source_dates.apply(_months, axis=1)

