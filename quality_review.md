# Quality Sleeve Assessment (March 2025)

## Summary Decision
- Quality is **not** being added to the live sleeve mix.
- We continue with the base equal-weight Momentum + Value (M+V) implementation.

## Evidence Review
- Re-ran enhanced OOS diagnostics for four mixes: M+V, M+V+Q equal-weight, 40/40/20 (M/V/Q), and Quality standalone. Artifacts live in `results/enhanced_oos_quality/`.
- `quality_combo_base_stats.csv`: M+V keeps the strongest full-sample profile (12.1% CAGR, 0.53 Sharpe, -6.8% max DD). Both Quality blends dilute Sharpe (≤0.40) and deepen drawdowns (≥-8.6%).
- `quality_combos_summary.json`: SPA p-values (~0.98 vs. base) indicate Quality mixes fail to beat the M+V benchmark; deflated Sharpe falls from 5.49 (base) to 2.81 (40/40/20) and 0.83 (equal-weight).
- `quality_combo_regime_stats.json`: Quality drags in the regimes we care about (value/momentum drawdowns, COVID). Small pockets of relief (e.g., 2022 rate-shock) do not offset the broader shortfall.
- Risk alerts (`quality_br_risk_summary.json`, suite-level aggregations) increase materially: Quality introduces 61 extra LS flags and raises universe-level warnings to 224 months, without improving worst month (‑7.55%).
- Turnover declines when Quality is blended (0.32 → 0.27 monthly for equal-weight), but the performance deterioration and alert burden outweigh the operational benefit.

## Operational Guidance
- Leave Quality out of production runs; keep the current M+V workflow and cron.
- Preserve the evaluation artifacts for posterity. Future Quality experiments should target better drawdown control and alert suppression before reconsidering allocation.
