#!/usr/bin/env python3
"""Regime-aware sizing helper used by momentum/value sleeves.

Generates a smoothed scaling series based on dispersion, realized volatility
shocks, and local macro (CDI) shifts. The resulting multiplier can be used to
modulate hedge ratios or gross exposure without touching the underlying ranks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class RegimeScalerConfig:
    weight_dispersion: float = 0.4
    weight_vol: float = 0.4
    weight_macro: float = 0.2
    min_scale: float = 0.8
    max_scale: float = 1.1
    scale_sensitivity: float = 0.05
    smoothing_months: int = 3
    dispersion_window: int = 36
    zscore_window: int = 24
    vol_short_window: int = 3
    vol_long_window: int = 12
    cdi_short_window: int = 3
    cdi_long_window: int = 12
    apply_to: str = "hedge"  # hedge | gross | both


@dataclass
class RegimeScalerResult:
    scale: pd.Series
    score: pd.Series
    components: Dict[str, pd.Series]


def _rolling_zscore(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    window = max(3, int(window))
    min_periods = min_periods if min_periods is not None else min(12, window)
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std(ddof=0)
    z = (series - rolling_mean) / rolling_std.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan)


def load_regime_scaler_config(spec: Optional[str]) -> Optional[RegimeScalerConfig]:
    """Return a config object from a CLI spec (None | 'default' | json path)."""
    if not spec:
        return None
    if spec.lower() == "default":
        return RegimeScalerConfig()
    path = Path(spec).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Regime scaler config not found: {path}")
    data = json.loads(path.read_text())
    unknown = {k: v for k, v in data.items() if not hasattr(RegimeScalerConfig, k)}
    if unknown:
        raise ValueError(f"Unknown regime scaler keys: {sorted(unknown.keys())}")
    return RegimeScalerConfig(**{k: v for k, v in data.items() if hasattr(RegimeScalerConfig, k)})


def compute_regime_scaler(pf: pd.DataFrame, cfg: RegimeScalerConfig) -> RegimeScalerResult:
    """Compute a smoothed scaling series aligned with pf index."""
    if pf.empty:
        empty = pd.Series(index=pf.index, dtype=float)
        return RegimeScalerResult(scale=empty + 1.0, score=empty, components={})

    idx = pf.index
    dispersion = pf.get("dispersion", pd.Series(dtype=float)).astype(float).reindex(idx)
    ls_base = (
        pf.get("LS_pre_overlay")
        if "LS_pre_overlay" in pf.columns
        else pf.get("LS", pd.Series(dtype=float))
    ).astype(float).reindex(idx)
    cdi = pf.get("CDI", pd.Series(dtype=float)).astype(float).reindex(idx)

    # Dispersion: low dispersion (negative z) increases risk signal -> positive score.
    disp_z = _rolling_zscore(dispersion, cfg.dispersion_window)
    disp_component = -disp_z  # invert so that low dispersion => positive risk component

    # Volatility shock: short vs long realized vol ratio.
    vol_short = ls_base.rolling(window=max(2, cfg.vol_short_window), min_periods=max(2, cfg.vol_short_window)).std(ddof=0)
    vol_long = ls_base.rolling(window=max(4, cfg.vol_long_window), min_periods=max(4, cfg.vol_long_window)).std(ddof=0)
    vol_ratio = (vol_short / vol_long).replace([np.inf, -np.inf], np.nan)
    vol_z = _rolling_zscore(vol_ratio, cfg.zscore_window, min_periods=6).fillna(0.0)

    # Macro: steepening CDI (short > long) -> higher risk penalty.
    cdi_short = cdi.rolling(window=max(1, cfg.cdi_short_window), min_periods=max(1, cfg.cdi_short_window)).mean()
    cdi_long = cdi.rolling(window=max(1, cfg.cdi_long_window), min_periods=max(1, cfg.cdi_long_window)).mean()
    cdi_gap = (cdi_short - cdi_long).replace([np.inf, -np.inf], np.nan)
    macro_z = _rolling_zscore(cdi_gap, cfg.zscore_window, min_periods=6).fillna(0.0)

    disp_component = disp_component.fillna(0.0)
    score_raw = (
        cfg.weight_dispersion * disp_component
        + cfg.weight_vol * vol_z
        + cfg.weight_macro * macro_z
    )
    score_raw = score_raw.fillna(0.0)
    score = score_raw.rolling(window=max(1, cfg.smoothing_months), min_periods=1).mean()

    scale = 1.0 + cfg.scale_sensitivity * score
    scale = scale.clip(lower=cfg.min_scale, upper=cfg.max_scale)
    scale = scale.fillna(1.0)

    components = {
        "regime_component_dispersion": disp_component,
        "regime_component_vol": vol_z,
        "regime_component_macro": macro_z,
    }
    return RegimeScalerResult(scale=scale, score=score, components=components)

