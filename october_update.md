# October Allocation Update - 10 Mar 2025

## Data Refresh Checklist
- Update `financial_data.db` with Brazilian equity daily prices plus `securities`/`sectors` metadata through the 10 Mar 2025 formation date; the allocator reads these tables for stocks and also fetches BOVA11 levels from the same database.
- Refresh `cdi_series.xlsx` so the CDI index includes the latest business day. The allocator uses it to compute period returns for the new allocation window.

## Cold Start With Live Capital
- Command template:
  ```bash
  python3 scripts/momentum_br.py \
    --cohorts 3 --beta-overlay --apply-vol-target \
    --cold-start --ramp-frac 1.0 \
    --live-capital 10000000 --write-state-snapshots \
    --db-path "/path/to/financial_data.db" \
    --cdi-path "/path/to/cdi_series.xlsx"
  ```
- Flags:
  - `--live-capital` enables account-level targets and live order exports sized to your capital (shares rounded by `--lot-size`, default 1).
  - `--cold-start` ignores any previous weights, sizing deltas off the ramp fraction.
  - `--ramp-frac` defaults to `1 / cohorts`; set 1.0 to load the full LS exposure on day one, or pick a lower fraction if you prefer to stage in.
  - `--write-state-snapshots` saves `state/cohort_{j}_YYYY-MM.json` to document the bootstrap state.

## Expected Artifacts
- Standard monthly outputs (timeseries, summary, targets, orders) plus cold-start files:
  - `results/account_targets_coldstart_YYYY-MM.csv`
  - `results/orders_live_coldstart_YYYY-MM.csv`
- Validate targets with `python3 scripts/validate_strategy.py results/targets_YYYY-MM.csv` and monitor via `python3 scripts/monitor_live_strategy.py report state YYYY-MM` as needed.

## Monthly Rebalance Guide
- Re-run `python3 scripts/momentum_br.py` each month so the newly exported `targets_YYYY-MM.csv` and `orders_live_YYYY-MM.csv` reflect the next `trade_date` listed in `results/momentum_br_timeseries.csv`.
- After regenerating targets/orders, refresh the daily price indices:
  ```bash
  python scripts/export_daily_prices.py --apply-overlay --results-dir results --db-path "G:\My Drive\mel-database\financial_data.db" --cdi-path "G:\My Drive\mel-database\cdi_series.xlsx"
  ```
- Execute the LS orders on the published trade date, then hold the book until the following `next_trade_date`; the staggered cohorts are already encoded in the files.
- Adjust the BOVA11 overlay on the same trade date using `hedge_ratio * live_capital` from the latest exports; if the ratio stays inside the 0.10 band, keep the prior hedge.
- Vol targeting is embedded in the weights and orders, so after fills just verify gross long and short notionals sit near 51% of capital per side and that net beta is neutral.
- After execution, log the fills and reconcile versus the exported `account_targets_YYYY-MM.csv` to confirm alignment.

## Next Steps
1. Confirm both data sources are up to date through 10 Mar 2025.
2. Run the cold-start command with the correct paths, capital, and desired ramp fraction.
3. Review cold-start CSVs to ensure weights and share counts meet expectations before execution.
4. Schedule a monthly check-in to regenerate orders, adjust the hedge, and reconcile fills against the latest exports.
