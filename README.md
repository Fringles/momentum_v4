# Brazil Long/Short Momentum + Value Stack

This repository houses the production research stack for our Brazilian long/short strategy.  
It combines two independently managed sleeves—12‑2 cross‑sectional Momentum and EV/EBITDA Value—into an equal-weight composite with disciplined turnover control, beta hedging, and conservative delisting treatment.

---

## 1. Strategy Overview

| Sleeve | Signal window | Universe controls | Rebalance cadence | Key guardrails |
| ------ | ------------- | ----------------- | ----------------- | -------------- |
| **Momentum** | 12‑month lookback, skip the most recent month (12‑2), sector-relative percentiles | Liquidity ≥ BRL 2mm median daily volume (≈63 trading days), `min_eligible`=50 | 3 cohorts, each fully rebalanced quarterly, banded maintenance in-between | Keep/add bands (81.4% / 90.3%), adaptive exit caps, sector tolerance ±9.2pp, turnover budget ≈36% |
| **Value** | EV/EBITDA with 3‑month reporting lag, sector neutral | Same liquidity + lag hygiene (remove stale EBITDA) | 3 cohorts with identical rotation | Keep/add bands (81.4% / 90.3%), adaptive exit caps, turnover budget ≈28% |

**Composite portfolio**: arithmetic average of the two sleeves. Each sleeve targets 0.50 gross long and 0.50 gross short. After sleeve-level construction we:

1. Apply a composite beta overlay vs. IBOV (EWMA lookback ≈36 months, 0.5 shrink, ±0.1129 band).
2. Optionally target 10% annualized volatility (36-month rolling window).
3. Apply financing (CDI carry on cash, 8% annualized borrow cost on shorts) to obtain `LS_net`.

**Performance reference (data through Sep‑2025)**  
- Full sample: CAGR 12.1%, Sharpe 0.53 (annual 1.83), max drawdown ‑6.8%.  
- OOS 2019+: Sharpe 0.44 (annual 1.52) with a ‑6.2% max drawdown.  
These metrics come from `results/enhanced_oos_quality/base/combined_timeseries.csv` and `results/regime_test_combo/evaluation_summary.json`.

---

## 2. Data Requirements

### Prices & Metadata (SQLite)
```
daily_prices(date, open_price, close_price, adjusted_close, volume, security_id)
securities(security_id, ticker, country, security_type, sector_id)
sectors(sector_id, sector_name)
```
Filters: `country = 'BR'`, `security_type = 'Stock'`.

### Fundamentals (Value Sleeve)
`enterprise_value` and `ebitda_ttm` tables keyed by `ticker` and `date`. We enforce the reporting lag (`ebitda_lag_months`) and drop stale fundamentals (>18 months old).

### CDI Series
Excel file with columns `Date` and `Close` containing the CDI index level. Used to compute financing carry and excess returns.

Ensure the database and CDI file include data through the most recent trading day before you run monthly targets.

---

## 3. Repository Layout

```
scripts/
  momentum_br.py         # momentum sleeve engine
  value_br.py            # value sleeve engine
  combine_ls_sleeves.py  # builds the equal-weight composite
  enhanced_oos.py        # robustness and stress harness
  validate_strategy.py   # light validation on monthly targets
  monitor_live_strategy.py
  delist_risk.py         # shared delisting risk module
results/                 # run artifacts (targets, timeseries, summaries)
state/                   # optional cohort snapshots for audit
regime_configs/          # saved regime scaler configs (experiments)
```

Historic experiment folders (e.g., `results/quality_watch`, `results/regime_test_*`) are kept for reference but not part of production.

---

## 4. Running the Sleeves

### Momentum (base configuration)
```bash
python3 scripts/momentum_br.py \
  --cohorts 3 \
  --beta-overlay \
  --apply-vol-target \
  --db-path "/path/to/financial_data.db" \
  --cdi-path "/path/to/cdi_series.xlsx"
```

### Value
```bash
python3 scripts/value_br.py \
  --cohorts 3 \
  --beta-overlay \
  --apply-vol-target \
  --db-path "/path/to/financial_data.db" \
  --cdi-path "/path/to/cdi_series.xlsx"
```

### Key CLI Flags (shared)
| Flag | Purpose | Default Notes |
| ---- | ------- | ------------- |
| `--cohorts` | Number of staggered cohorts (1 or 3). Production = 3. |
| `--beta-overlay` | Enables IBOV hedge overlay with EWMA beta. |
| `--apply-vol-target` | Enforces 10% annualized vol on the hedged series. |
| `--ls-turnover-budget` | Monthly turnover budget (0.32496 for momentum, 0.2788 for value). |
| `--adaptive-exit-cap` | Adaptive limit on exits per side per month (default ON). |
| `--short-borrow-annual` | Borrow cost for shorts (default 8%). |
| `--write-state-snapshots` | Persist cohort JSON snapshots in `state/`. |
| `--live-capital` | When provided, produces `orders_live_YYYY-MM.csv` sized for the account. |
| `--dispersion-gate <quantile>` | Optional “do not rebalance when cross-sectional dispersion is low”. Production uses `none`. |
| `--regime-scaler` | Optional regime-aware scaler. Leave unset for base behavior (see §8). |

Both engines treat missing exit prices as full write-offs and haircut high-delist-risk names using the multiplier map defined in `delist_risk.py`.

---

## 5. Combining the Sleeves

After running both sleeves:
```bash
python3 scripts/combine_ls_sleeves.py \
  --momentum results/momentum_br_timeseries.csv \
  --value    results/value_br_timeseries.csv \
  --out      results/combined_timeseries.csv
```

The combined file contains:
- `combo`, `combo_net`, `combo_carry`, `combo_borrow`, `combo_financing`
- Averaged turnover, gross exposures, risk flag counts
- Source sleeve risk columns (`mom_risk_*`, `val_risk_*`)

---

## 6. Monthly Operations Runbook

1. **Update data**  
   - Import the latest prices and CDI index through the final trading day of the month.
   - Verify fundamentals (EV / EBITDA) are refreshed.

2. **Generate targets & orders**  
   ```bash
   python3 scripts/momentum_br.py ... --live-capital <amount> --write-state-snapshots
   python3 scripts/value_br.py ...    --live-capital <amount> --write-state-snapshots
   ```

3. **Review outputs**  
   - Console summary: CAPM alpha, Sharpe, worst month, turnover, cost grid.
   - Check `results/targets_YYYY-MM.csv` (cohort-level) and `results/account_targets_YYYY-MM.csv` (netted).
   - Inspect `results/orders_live_YYYY-MM.csv` for execution sizes.
   - Run `scripts/combine_ls_sleeves.py` to refresh `results/combined_timeseries.csv` (and any scenario folders) so the composite carries the latest `combo_hedge_ratio`/`combo_beta_est` columns.
   - Examine `risk_flags_YYYY-MM.csv` for delisting alerts and weight haircuts.

4. **Validate**  
   ```bash
   python3 scripts/validate_strategy.py results/targets_YYYY-MM.csv
   ```

5. **Trade at T+1 open**  
   - Execute orders using the `orders_live` file and size the IBOV overlay from `combo_hedge_ratio` in `results/combined_timeseries.csv` (or the matched scenario directory).
   - Vol-target scale (`vt_scale`) is already baked into the exported weights when `--apply-vol-target` is set.

6. **Post-trade**  
   - Archive orders, targets, and the refreshed `combined_timeseries.csv` alongside the run artefacts.
   - Update `state/` snapshots (helps diff expected vs. realized).
   - Optional monitoring:  
     ```bash
     python3 scripts/monitor_live_strategy.py report state YYYY-MM
     ```

7. **Monthly diagnostics**  
   - Run `combine_ls_sleeves.py` followed by `enhanced_oos.py` or the evaluation helper (see §7) to track OOS metrics.

---

## 7. Evaluation & Diagnostics

### Enhanced OOS Harness
```bash
python3 scripts/enhanced_oos.py \
  --db-path "/path/to/financial_data.db" \
  --cdi-path "/path/to/cdi_series.xlsx" \
  --workdir results/enhanced_oos_quality \
  --no-rerun-base
```

Outputs:
- `enhanced_oos_summary.json`: base metrics, deflated Sharpe, SPA p-value, walk-forward table, turnover stress results.
- Stress scenarios for band keep/add, turnover budget, dispersion gate variants.

### Quick Composite Evaluation
To compare an alternative composite against the base:
```bash
python3 - <<'PY'
import json, pandas as pd
from enhanced_oos import compute_metrics, deflated_sharpe, spa_pvalue

base = pd.read_csv("results/enhanced_oos_quality/base/combined_timeseries.csv",
                   parse_dates=["month_end"]).set_index("month_end")
candidate = pd.read_csv("results/my_test/combined_timeseries.csv",
                        parse_dates=["month_end"]).set_index("month_end")
df = pd.concat([base["combo"], candidate["combo"]], axis=1, join="inner").dropna()
diff = (df.iloc[:,1] - df.iloc[:,0]).values.reshape(-1,1)
metrics = compute_metrics(candidate["combo"], candidate["cdi"])
spa = spa_pvalue(diff)
print(metrics, spa)
PY
```

### Risk Monitoring
`*_risk_summary.json` captures:
- Full, 12m, and 24m max drawdowns.
- Worst month.
- Latest delist risk multipliers and flag counts.
- Console alerts print when thresholds breach (worst month worse than ‑6%, 12m DD worse than ‑8%, LS risk multiplier <0.60, etc.).

---

## 8. Optional Regime Scaler (Experiment)

The engines accept `--regime-scaler <config>` pointing to a JSON file built from `RegimeScalerConfig` (see `scripts/regime_scaler.py`). This multiplier was prototyped to adjust gross exposure or hedge intensity based on:
- Sector dispersion z-score (low dispersion ⇒ scale down risk),
- Short vs. long LS realized vol,
- CDI slope (macro tightening proxy).

**Status**: the default settings yield minimal impact (scale hovers near 1). A more aggressive macro-weighted configuration was tested (`regime_configs/aggressive_macro.json`) but reduced Sharpe and deepened drawdowns; see `results/regime_test_combo_macro/discard_note.json`.  
**Production guidance**: run without the scaler unless you explicitly test and validate new parameters.

---

## 9. Experiments & Known Decisions

- **Quality Sleeve** (`results/quality_watch/`): A standalone Quality factor and blended weights (M+V+Q) were analysed (`results/enhanced_oos_quality/quality_combo_base_stats.csv`). Quality increased risk alerts and diluted Sharpe; we chose to keep the live mix at pure Momentum + Value (documented in `quality_review.md`).
- **Aggressive Regime Overlay** (`results/regime_test_*_macro/`): The gross-only, high-sensitivity overlay underperformed the base composite. Outcome recorded, feature disabled by default.
- **Delisting Treatment**: Since March 2025, missing exit prices are written as `0.0` (full loss). Expect worse backtest metrics compared with pre-fix runs; see `delisting_handling.md`.

---

## 10. Troubleshooting

| Issue | Likely cause | Fix |
| ----- | ------------ | ---- |
| `FileNotFoundError` for DB/CDI | Path typo or sync lag | Confirm mount path, ensure Google Drive/Cloud storage is mounted locally. |
| Empty universe warning | Liquidity threshold too high or database missing recent data | Refresh `daily_prices`, lower `--liquidity-threshold` temporarily to diagnose. |
| `validate_strategy.py` fails | Sector cap breach, cohort mismatch, or NaNs in targets | Inspect the offending rows in `targets_YYYY-MM.csv`, verify fundamentals coverage, rerun after fixing data. |
| Hedge ratio same month-to-month | Beta estimate within ±band; expected behavior. Adjust `--overlay-band` if you want more frequent hedge updates. |
| Large negative LS return with zero turnover | Likely delisting write-off. Check `risk_flags_YYYY-MM.csv` and underlying ticker news. |

---

## 11. What to Do Next

- **Monthly**: Follow the runbook, archive outputs, keep `results/enhanced_oos_quality/` updated.  
- **Quarterly**: Review enhanced OOS summary and risk alerts, revisit Quality/regime experiments only if new evidence surfaces.  
- **Ad hoc**: Use `combine_ls_sleeves.py` + `enhanced_oos.py` to test parameter tweaks (band widths, turnover budgets, overlay halflife) before contemplating production changes.

This README mirrors the current production playbook. Update it whenever strategy logic, run cadence, or validation tooling changes. If you run new experiments, drop a note in the *Experiments* section with conclusions so future readers understand which branches were tested and why they were accepted or discarded.
