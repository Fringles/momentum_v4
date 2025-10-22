# Run 09 Parameter Set – Full Backtest (2011-01 to 2025-09)

## Configuration
- `cohorts`: 3
- `beta_overlay`: enabled
- `apply_vol_target`: enabled (10% annual target)
- `band_keep`: 0.8191192082182532
- `band_add`: 0.9027556845757927
- `sector_tol`: 0.08495767247886057
- `exit_cap_frac`: 0.06773663256565816
- `ls_turnover_budget`: 0.2620299928400281
- `micro_add_frac`: 0.029181996071167858
- `overlay_band`: 0.11592588762393818
- Other CLI parameters: defaults in `scripts/momentum_br.py`
- Command:
  ```bash
  python3 scripts/momentum_br.py \
    --cohorts 3 --beta-overlay --apply-vol-target \
    --band-keep 0.8191192082182532 \
    --band-add 0.9027556845757927 \
    --sector-tol 0.08495767247886057 \
    --exit-cap-frac 0.06773663256565816 \
    --ls-turnover-budget 0.2620299928400281 \
    --micro-add-frac 0.029181996071167858 \
    --overlay-band 0.11592588762393818 \
    --out-dir results/run09_full
  ```

## Headline Results (LS sleeves)
- Mean monthly return: 1.007% (t = 4.12)
- CAPM alpha vs IBOV: 102.9 bps/month (t = 4.19)
- Annualised IR: 1.23 (tracking error 10.0% ann.)
- Max drawdown: -13.7%
- Hit rate: 67.2%
- Turnover: 35.6%/month (reconstitution 79%, reweight 21%)
- Cost efficiency: 2.9 bps alpha per 100% turnover

### Vol-targeted LS (same stats as raw)
- Mean monthly return: 1.007% (t = 4.12)
- CAPM alpha: 102.9 bps/month (t = 4.19)
- Max drawdown: -13.7%
- Hit rate: 67.2%

### Baseline (pre-vol-target)
- Mean monthly return: 0.987% (t = 3.87)
- CAPM alpha: 101.3 bps/month (t = 3.99)
- Max drawdown: -18.3%
- Hit rate: 67.2%

### D10 Sleeve
- Mean monthly return: 1.029% (t = 2.14)
- Excess vs CDI: 0.266%/mo (t = 0.54)
- Max drawdown: -46.8%
- Hit rate: 56.5%

## Cost Grid (LS, half-sided bps)
| Costs (bps) | Net Alpha (bps/mo) | t-stat | IR (ann.) | Max DD | Hit Rate |
|-------------|--------------------|--------|-----------|--------|----------|
| 20          | 99.3               | 4.04   | 1.19      | -12.29%| 66.1%    |
| 50          | 94.0               | 3.81   | 1.13      | -12.51%| 66.1%    |
| 100         | 85.1               | 3.44   | 1.02      | -12.87%| 64.4%    |

## Subperiod Diagnostics
- **2011-01 to 2018-12**: LS mean 1.184%/mo (t = 3.27)
- **2019-01 to 2025-12**: LS mean 0.797%/mo (t = 2.61)

## Outputs
- Timeseries: `results/run09_full/momentum_br_timeseries.csv`
- Summary JSON: `results/run09_full/momentum_br_summary.json`
- Cost grid: `results/run09_full/momentum_br_cost_grid.csv`
- Target/order/state exports for every month under `results/run09_full/`
