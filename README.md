# Brazil Momentum Strategy (12-2, Sector-Neutral, 3-Cohort Staggered)

This repository contains a long/short momentum backtest for Brazilian equities with:
- Sector-neutral 12-2 momentum signal on monthly closes
- Three staggered cohorts (quarterly per cohort, monthly rotation)
- Banded, minimal-trading cohort maintenance with guardrails
- Composite-level beta overlay vs IBOV, then 10% vol targeting
- Cost grid, turnover decomposition, and CAPM alpha/IR yardstick

If you're new to this project, start with the Quick Start, then refer to Strategy Design and Configuration for details.

---

## Quick Start

Prerequisites
- Python 3.9+
- Packages: numpy, pandas, statsmodels, matplotlib (optional for plots)
- Data: SQLite DB with Brazilian equities, Excel CDI series

Run the main backtest (recommended defaults)

```bash
python3 scripts/momentum_br.py --cohorts 3 --beta-overlay --apply-vol-target \
  --db-path "/path/to/financial_data.db" \
  --cdi-path "/path/to/cdi_series.xlsx"
```

Generate plots (optional)

```bash
python3 scripts/plot_momentum_br.py \
  --results-path results/momentum_br_timeseries.csv \
  --out-dir results
```

A detailed write-up with current results lives at:
- `results/momentum_strategy_report.md`

---

## Data Requirements

SQLite schema (used columns):
- `daily_prices(date, open_price, close_price, adjusted_close, volume, security_id)`
- `securities(security_id, ticker, country, security_type, sector_id)`
- `sectors(sector_id, sector_name)`

Universe filter: `security_type = 'Stock'` and `country = 'BR'`.

CDI series: Excel with columns:
- `Date` (datetime)
- `Close` (index level) — used to compute period returns as ratio.

---

## Strategy Design

Signal
- Monthly adjusted closes; momentum = ln(P[t-2]) − ln(P[t-12]).
- Sector-neutral ranking via within-sector z-scores; global sort.

Portfolio formation
- Long sleeve (L): top decile; Short sleeve (S): bottom decile.
- Beta-neutralize sleeves using 12-month daily betas vs IBOV; scale sides to keep gross ≈ 1 (0.5 per side).

Cohorts and staggering
- Universe split into 3 cohorts; monthly rotation so each cohort rebalances once per quarter.
- Composite return equals the average of cohort returns; composite turnover is the average of cohort turnovers.

Banded rebalancing (inside each cohort)
- Keep bands: Long keep if percentile ≥ 80th; Short keep if ≤ 20th.
- Add bands: Long add if ≥ 90th and not in cohort; Short add if ≤ 10th and not in cohort.
- Non-rebalance months: process exits only; defer adds to cohort's next full rebalance.

Guardrails (to limit drift and trading)
- Side gross tolerance: enforce 0.50 ± 2% only when outside band.
- Single-name cap/floor per side: cap = min(2× equal, 2.5%), floor = 0.25× equal.
- Sector drift nudge: within side, keep sector shares within ±10pp of equal-by-sector; apply small pro-rata nudge when breached.
- Exit cap (non-rebalance): at most 5% of target names per side may exit per month; others are carried.
- Concentration emergency: one-off equalize when effective N < 70% of target or > 20% at floor.

Composite overlay and vol targeting (ordering matters)
- Overlay: composite LS beta vs IBOV from EWMA (~60 trading days), 50% shrink toward 0, ±0.10 band; applied first.
- Vol targeting: 10% annualized volatility using 36-month rolling vol; applied to hedged series.

Timing hygiene
- Portfolios are formed at calendar month-end using data through t, traded at next day's open (T+1), and held to next T+1 open.

---

## Performance Yardstick

Primary for LS
- CAPM alpha vs IBOV (monthly), HAC t-stat (Newey–West), beta
- Alpha information ratio (IR, annualized) = annualized alpha / annualized tracking error (std of CAPM residuals)

Reference metrics
- LS: mean monthly (raw), hit rate, max drawdown
- D10: mean excess vs CDI, HAC t-stat, annualized Sharpe (ex-CDI)

---

## Configuration (CLI Flags)

Main inputs
- `--db-path`: SQLite database path (required)
- `--cdi-path`: CDI Excel path (required)
- `--cohorts`: number of cohorts (default: 1; use 3 for staggering)

Banded/guardrails (defaults tuned)
- `--band-keep 0.80` `--band-add 0.90`
- `--sector-tol 0.10` (±10pp per side)
- `--gross-tol 0.02` (±2% around 0.50 per side)
- Exit caps:
  - Adaptive caps enabled by default: `--adaptive-exit-cap`
  - Fixed cap alternative: `--no-adaptive-exit-cap --exit-cap-frac 0.05`
  - Adaptive rule (per side, non-rebalance months):
    - cap = clip(5%, 15%, 5% + 40%×p + 5%×1[D in top tercile] + 5%×R)
    - p: fraction breaching keep band; D: dispersion between D10/D1 medians; R: reversal regime flag
- Turnover budget (default ON):
  - `--use-turnover-budget` (disable with `--no-use-turnover-budget`)
  - `--ls-turnover-budget 0.30` targets ≈30%/mo LS turnover; budget enforced via severity queue and emergency override.

Overlay and vol targeting
- `--beta-overlay` (enable composite overlay)
- `--overlay-halflife-days 60` `--overlay-shrink 0.5` `--overlay-band 0.10`
 - `--apply-vol-target` (apply 10% ann. vol targeting to hedged LS)
 - `--vol-target-ann 0.10` `--vol-window-months 36` `--vol-min-months 12`

Other
- Liquidity, start/end dates, NW lags, and minimum eligible count also configurable; see `--help`.

---

## Outputs

- Timeseries: `results/momentum_br_timeseries.csv`
  - `D10`, `LS`, `turnover_D10`, `turnover_LS`, `turnover_LS_reconst`, `turnover_LS_reweight`, `CDI`, `BOVA11`, `LS_pre_overlay`, `beta_est`, `hedge_ratio`, `vt_scale`, `LS_vt`, `LS_pre_vt`.
- Summary: `results/momentum_br_summary.json` (LS mean/t, CAPM alpha/t/beta, IR, TE; D10 metrics).
- Cost grid: `results/momentum_br_cost_grid.csv` (20/50/100 bps, plus vol-targeted LS at 10%).
- Plots (optional): generated by `scripts/plot_momentum_br.py`.
- Console summary prints include a turnover decomposition and alpha per 1× turnover cost efficiency.

---

## Live Trading Operations

### Monthly Execution Workflow

**Step 1: Data Preparation**
- Ensure database is updated with latest daily prices through previous trading day
- Verify CDI series is current
- Update database and CDI paths in command if needed

**Step 2: Generate Targets and Orders**
```bash
python3 scripts/momentum_br.py --cohorts 3 --beta-overlay --apply-vol-target \
  --live-capital 10000000 --write-state-snapshots \
  --db-path "/path/to/financial_data.db" \
  --cdi-path "/path/to/cdi_series.xlsx"
```

**Step 3: Pre-Trade Validation**
```bash
python3 scripts/validate_strategy.py results/targets_2024-01.csv
```

**Step 4: Review Outputs**
- Check console output for strategy performance metrics
- Review `results/targets_YYYY-MM.csv` for position changes (by cohort)
- Review `results/account_targets_YYYY-MM.csv` for netted account-level targets
- Review `results/orders_live_YYYY-MM.csv` for netted live orders (recommended to execute)
- Optional: `results/orders_YYYY-MM.csv` shows cohort-level orders for attribution
- Optional: `results/allocations_YYYY-MM.csv` maps parent orders back to cohorts
- Verify `state/cohort_{j}_YYYY-MM.json` files are created

**Step 5: Execute Trades**
- Trade at next trading day's opening prices (T+1)
- Use orders CSV as guide for manual execution
- Apply composite beta overlay if `hedge_ratio != 0` in timeseries
- Monitor execution vs targets

**Step 6: Post-Trade Documentation**
- Archive all outputs with execution confirmations
- Update live capital for next month if needed
- Monitor P&L vs expected returns

### Cold Start (New Accounts)

Use cold start when bootstrapping a new account or portfolio sleeve with no existing positions.

- What it does: sets `prev_weight = 0` and sizes `delta_weight = ramp_frac × target_weight` (default `ramp_frac = 1/cohorts`).
- Vol targeting: if `--apply-vol-target` is supplied, LS targets/deltas in cold-start exports are already vol‑scaled for live execution.
- Outputs written:
  - `results/account_targets_coldstart_YYYY-MM.csv`
  - `results/orders_live_coldstart_YYYY-MM.csv`
- Recommended: choose `--ramp-frac` (e.g., 0.33 for 3 cohorts) to distribute market impact across months.

Example:
```bash
python3 scripts/momentum_br.py \
  --cohorts 3 --beta-overlay --apply-vol-target \
  --cold-start --ramp-frac 0.33 \
  --live-capital 10000000 --write-state-snapshots \
  --db-path "/path/to/financial_data.db" \
  --cdi-path "/path/to/cdi_series.xlsx"
```

### Timing Calendar

**Formation Date**: Last trading day of each calendar month
- Use all data available through this date
- Generate targets and orders after market close

**Execution Date**: Next trading day (T+1)
- Execute all trades at opening prices
- Apply hedge ratio for composite beta overlay
- Hold positions until next T+1

**Rebalancing Schedule** (3-cohort mode):
- **January, April, July, October**: Cohort 0 full rebalance
- **February, May, August, November**: Cohort 1 full rebalance  
- **March, June, September, December**: Cohort 2 full rebalance
- Non-rebalance cohorts: exits only, subject to adaptive caps

### Emergency Procedures

**High Turnover Alert** (>40% monthly):
- Check `rationale` column in targets for unusual `add/exit` activity
- Verify market conditions haven't triggered emergency rebalancing
- Consider tightening exit caps with `--exit-cap-frac 0.03`

**Validation Failures**:
- Review error messages from validation script
- Check universe size and liquidity filters
- Verify database completeness for formation period

**System Failures**:
- State snapshots in `state/` directory enable recovery
- Can regenerate any month's targets using same parameters
- All parameters logged in console output for reproducibility

### Risk Monitoring

**Daily Checks**:
- Portfolio gross exposure should be ~100% (0.50 per side)
- Net beta exposure should be near zero (check `beta_est` in timeseries)
- Individual position sizes within caps (2.5% max, 0.125% min floor)

**Monthly Reviews**:
- CAPM alpha vs IBOV (target: >80 bps/month)
- Information ratio (target: >1.0 annualized)
- Turnover efficiency (target: >2.5 bps alpha per 1× turnover)
- Max drawdown tracking (alert if >20%)

### Configuration for Live Trading

**Recommended Settings**:
```bash
--cohorts 3                    # Enable staggered rebalancing
--beta-overlay                 # Composite beta hedging vs IBOV
--apply-vol-target            # Apply 10% ann. VT to hedged LS
--adaptive-exit-cap            # Dynamic exit management
--live-capital 10000000        # Your live capital (enables orders export)
--write-state-snapshots        # Enable operational tracking
--sector-tol 0.10             # ±10pp sector drift tolerance
--gross-tol 0.02              # ±2% gross exposure tolerance
--exit-cap-frac 0.05          # 5% max exits per side (non-rebalance months)
```

**Performance Tuning**:
- Increase `--exit-cap-frac` to 0.075 if turnover consistently <25%/month
- Decrease to 0.03 if turnover >40%/month
- Adjust `--band-keep` (0.75-0.85) and `--band-add` (0.85-0.95) for different turnover profiles

---

## Example Commands

Run with defaults, 3 cohorts + overlay + VT

```bash
python3 scripts/momentum_br.py --cohorts 3 --beta-overlay --apply-vol-target \
  --db-path "/path/to/financial_data.db" \
  --cdi-path "/path/to/cdi_series.xlsx"
```

Tweak rebalancing bands and guardrails

```bash
python3 scripts/momentum_br.py --cohorts 3 --beta-overlay \
  --band-keep 0.82 --band-add 0.92 \
  --sector-tol 0.12 --gross-tol 0.03 --exit-cap-frac 0.05 \
  --db-path "/path/to/financial_data.db" --cdi-path "/path/to/cdi_series.xlsx"
```

Cold-start bootstrap (new account)

```bash
python3 scripts/momentum_br.py \
  --cohorts 3 --beta-overlay --apply-vol-target \
  --cold-start --ramp-frac 0.33 \
  --live-capital 10000000 \
  --db-path "/path/to/financial_data.db" \
  --cdi-path "/path/to/cdi_series.xlsx"
```

---

## Turnover / Alpha Tradeoff (Empirical)

Exit-cap sweep on non-rebalance months (3 cohorts, bands 80/90, sector tol 10pp, gross tol ±2%, overlay + VT):
- 10% exits: LS turnover ≈ 39%/mo; alpha ≈ 110.5 bps/mo; IR ≈ 1.30; ~72% of turnover = reconstitution
- 7.5% exits: LS turnover ≈ 34%/mo; alpha ≈ 108.2 bps/mo; IR ≈ 1.30; ~79% reconstitution
- 5% exits (default): LS turnover ≈ 31%/mo; alpha ≈ 106.3 bps/mo; IR ≈ 1.30; ~83% reconstitution

Interpretation: Tightening the exit cap reduces turnover materially with ~10% alpha bleed and stable IR; reconstitution dominates trading.

Adaptive caps: spend more turnover when p is high, signal separation D is strong, or reversal risk R=1; conserve otherwise. Exits are prioritized by breach severity and speed with a two-strike rule for marginal breaches.

---

## Methodology Notes

- Beta estimation for sleeves (cohort rebalance) uses 12 months of daily returns; composite overlay beta uses EWMA on monthly LS vs IBOV.
- Newey–West lags default to 6 for HAC t-stats; adjust with `--nw-lags` if desired.
- Liquidity filter uses rolling 63-day median daily volume and a configurable threshold.
- All weights and turnover use simple L1 changes in target weights; turnover decomposition splits adds/exits vs reweighting.

---

## Troubleshooting

- Missing data or very short histories can suppress beta estimates or momentum labels; ensure sufficient coverage in the DB.
- If you see warnings about pandas resampling (`'M'`), they are cosmetic; behavior matches month-end semantics.
- If CDI or BOVA11 data is sparse at period endpoints, the code uses nearest previous values to compute period returns.

---

## License

Internal research use. Do not redistribute without permission.
