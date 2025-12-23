# Combined LS Momentum + Value – Full Backtest (Run 05 defaults)

## Configuration
- `cohorts`: 3 for both sleeves
- `beta_overlay`: enabled
- `apply_vol_target`: enabled (10% target)
- Guardrails (momentum & value):
  - `band_keep`: 0.81393
  - `band_add`: 0.90307
  - `sector_tol`: 0.09218
  - `exit_cap_frac`: 0.07257
  - `ls_turnover_budget`: 0.32496
  - `overlay_band`: 0.11293
- Value-specific: `ebitda_lag_months = 3`
- Micro adds use default 0.02918 (momentum) / 0.02918 (value internal default).
- Commands:
  ```bash
  python3 scripts/momentum_br.py --cohorts 3 --beta-overlay --apply-vol-target \
    --band-keep 0.81393 --band-add 0.90307 --sector-tol 0.09218 \
    --exit-cap-frac 0.07257 --ls-turnover-budget 0.32496 \
    --overlay-band 0.11293 --out-dir results/combined_default/momentum

  python3 scripts/value_br.py --cohorts 3 --beta-overlay --apply-vol-target \
    --band-keep 0.81393 --band-add 0.90307 --sector-tol 0.09218 \
    --exit-cap-frac 0.07257 --ls-turnover-budget 0.32496 \
    --overlay-band 0.11293 --out-dir results/combined_default/value
  ```
- Combined sleeve: equal-weight of LS momentum and LS value (post cost, pre overlay/VT).
- Outputs: `results/combined_default/combined_timeseries.csv`, `combined_equity_curve.png`.

## Combined LS Performance (2011-01 – 2025-09, 177 months)
- Mean monthly return: **0.886%** (t ≈ 6.76)
- Annualised return: **11.17%**
- Annualised volatility: **6.25%**
- CAPM alpha vs IBOV: **87 bps/month** (t ≈ 6.74), β ≈ **+0.02**
- Hit rate: **67.8%** positive months
- Max drawdown: **−5.6%**
- Cost grid (combined, 50 bps RT implied via individual sleeves): net alpha ≈ 83–90 bps/month once sleeve trading costs are reflected; blended drawdown remains around −6%.
- Equity curve plot saved at `results/combined_default/combined_equity_curve.png` (see below).

### Momentum Sleeve (Run 05 parameters)
- Mean monthly: **0.992%** (t = 4.04)
- CAPM alpha: **102.5 bps/mo** (t = 4.21) with β ≈ −0.052
- Max drawdown: −13.7%; turnover 35.9%/mo (≈78% reconstitution)

### Value Sleeve (Run 05 parameters – micro adds 0.02918 default)
- Mean monthly: **0.935%** (t = 4.78)
- CAPM alpha: **96.8 bps/mo** (t = 5.08) with β ≈ −0.050
- Max drawdown: −14.0%; turnover 27.7%/mo (≈70% reconstitution)

## Equity Curve
![Combined LS vs Benchmarks](combined_equity_curve.png)
