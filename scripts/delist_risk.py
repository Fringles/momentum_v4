#!/usr/bin/env python3
"""Delisting risk heuristics shared across momentum and value engines."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DelistRiskConfig:
    """Parameters controlling the delisting-risk watchlist."""

    short_lookback: int = 21
    min_trading_days: int = 15
    halt_streak: int = 3
    liquidity_slump_ratio: float = 0.5
    absolute_liquidity_floor_multiplier: float = 0.75
    zero_volume_days: int = 4
    score_multipliers: Mapping[int, float] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.score_multipliers is None:
            object.__setattr__(
                self,
                "score_multipliers",
                {
                    0: 1.0,
                    1: 0.75,
                    2: 0.5,
                    3: 0.25,
                },
            )

    def multiplier(self, score: int) -> float:
        score = max(0, score)
        if score in self.score_multipliers:
            return self.score_multipliers[score]
        # Cap at the worst bucket if score exceeds available keys
        max_key = max(self.score_multipliers.keys())
        return self.score_multipliers[max_key]


def _max_consecutive_missing(values: pd.Series) -> int:
    arr = values.notna().to_numpy()
    max_run = 0
    run = 0
    for traded in arr:
        if traded:
            run = 0
        else:
            run += 1
            if run > max_run:
                max_run = run
    return int(max_run)


def compute_delisting_risk(
    po: pd.DataFrame,
    pv: pd.DataFrame,
    tickers: Sequence[str],
    as_of: pd.Timestamp,
    long_med_vol: pd.Series,
    liquidity_threshold: float,
    config: Optional[DelistRiskConfig] = None,
) -> Dict[str, Dict[str, object]]:
    """Return per-ticker delisting risk diagnostics using only information up to as_of."""

    cfg = config or DelistRiskConfig()
    if not tickers:
        return {}
    cols = [t for t in tickers if t in po.columns]
    if not cols:
        return {}

    po_window = po.loc[:as_of, cols].tail(cfg.short_lookback)
    pv_window = pv.loc[:as_of, cols].tail(cfg.short_lookback)

    trade_counts = po_window.notna().sum(axis=0)
    max_halt = {t: _max_consecutive_missing(po_window[t]) for t in cols}
    zero_volume = (pv_window.fillna(0.0) <= 0.0).sum(axis=0)
    short_med_vol = pv_window.median(axis=0, skipna=True)
    long_med = long_med_vol.reindex(cols)

    liquidity_floor = float(cfg.absolute_liquidity_floor_multiplier * liquidity_threshold)

    out: Dict[str, Dict[str, object]] = {}
    for t in cols:
        flags: List[str] = []
        tc = int(trade_counts.get(t, 0))
        mh = int(max_halt.get(t, 0))
        zv = int(zero_volume.get(t, 0))
        short_med = float(short_med_vol.get(t, np.nan)) if t in short_med_vol else np.nan
        long_med_val = float(long_med.get(t, np.nan)) if t in long_med else np.nan

        if mh >= cfg.halt_streak:
            flags.append("halt_run")
        if tc < cfg.min_trading_days:
            flags.append("thin_trading")
        slump_threshold = cfg.liquidity_slump_ratio * (
            long_med_val if np.isfinite(long_med_val) and long_med_val > 0 else liquidity_threshold
        )
        if np.isfinite(short_med):
            if short_med < max(liquidity_floor, slump_threshold):
                flags.append("liquidity_slump")
        else:
            flags.append("no_recent_volume")
        if zv >= cfg.zero_volume_days:
            flags.append("zero_volume_streak")

        unique_flags = sorted(set(flags))
        score = len(unique_flags)
        out[t] = {
            "score": score,
            "flags": unique_flags,
            "multiplier": cfg.multiplier(score),
            "trading_days": tc,
            "max_halt_run": mh,
            "short_median_volume": short_med if np.isfinite(short_med) else np.nan,
            "long_median_volume": long_med_val if np.isfinite(long_med_val) else np.nan,
            "zero_volume_days": zv,
        }
    # Ensure every requested ticker gets an entry (even if column missing)
    for t in tickers:
        if t not in out:
            out[t] = {
                "score": 0,
                "flags": [],
                "multiplier": 1.0,
                "trading_days": 0,
                "max_halt_run": 0,
                "short_median_volume": np.nan,
                "long_median_volume": np.nan,
                "zero_volume_days": 0,
            }
    return out


def summarize_weight_exposure(
    weight_objects: Union[Dict[str, float], Sequence[Dict[str, float]]],
    risk_profiles: Mapping[str, Mapping[str, object]],
) -> Dict[str, float]:
    """Return weighted multiplier / counts for the supplied weights."""

    if isinstance(weight_objects, dict):
        weights_iter = [weight_objects]
    else:
        weights_iter = list(weight_objects)

    total_abs = 0.0
    weighted_mult = 0.0
    weighted_score = 0.0
    flagged_names: set = set()
    high_names: set = set()

    for weights in weights_iter:
        for ticker, weight in weights.items():
            w_abs = abs(float(weight))
            if w_abs <= 0.0:
                continue
            info = risk_profiles.get(ticker, {})
            mult = float(info.get("multiplier", 1.0))
            score = int(info.get("score", 0))
            weighted_mult += w_abs * mult
            weighted_score += w_abs * score
            total_abs += w_abs
            if score > 0:
                flagged_names.add(ticker)
            if score >= 2:
                high_names.add(ticker)

    summary = {
        "weighted_multiplier": float(weighted_mult / total_abs) if total_abs > 0 else np.nan,
        "weighted_score": float(weighted_score / total_abs) if total_abs > 0 else np.nan,
        "flagged": float(len(flagged_names)),
        "high": float(len(high_names)),
    }
    return summary


def summarize_universe(risk_profiles: Mapping[str, Mapping[str, object]]) -> Dict[str, float]:
    """Aggregate risk statistics across the full universe."""

    if not risk_profiles:
        return {"flagged": 0.0, "high": 0.0, "avg_multiplier": np.nan}
    multipliers = [float(info.get("multiplier", 1.0)) for info in risk_profiles.values()]
    flagged = sum(1 for info in risk_profiles.values() if int(info.get("score", 0)) > 0)
    high = sum(1 for info in risk_profiles.values() if int(info.get("score", 0)) >= 2)
    return {
        "flagged": float(flagged),
        "high": float(high),
        "avg_multiplier": float(np.mean(multipliers)) if multipliers else np.nan,
    }

