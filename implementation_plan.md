# Execution-First Plan (Exact Strategy, Simpler Ops)

Principle: Do not change the final strategy. Execute the exact spec (3 staggered cohorts, banded keep/add, adaptive exit caps, turnover budget, composite beta overlay, 10% VT) while simplifying the operational workflow and interfaces.

## Phase 0 — One-Command Monthly Run
- Single CLI invocation runs the full strategy as-is and writes artifacts.
- Inputs: SQLite DB (prices, sectors), CDI Excel; paths via flags.
- Command: `python3 scripts/momentum_br.py --cohorts 3 --beta-overlay --db-path ... --cdi-path ...`
- Outputs (kept):
  - `results/momentum_br_timeseries.csv`, `results/momentum_br_summary.json`, `results/momentum_br_cost_grid.csv`.
  - Plots optional via `scripts/plot_momentum_br.py`.

## Phase 1 — Trade-Ready Exports (Manual Entry)
- Targets export (per month): cohort-aware LS and D10 targets consistent with the strategy’s banded logic.
  - File: `results/targets_{YYYY-MM}.csv` with: `cohort_id, ticker, side(L/S), target_weight, prev_weight, delta_weight, rationale(keep/add/exit/reweight), sector, duration_months, strikes, notes`.
- Orders helper (optional, manual-friendly): compute shares from target deltas given capital.
  - Inputs: `--live-capital`, prior holdings CSV (optional), lot-size map (optional).
  - File: `results/orders_{YYYY-MM}.csv` with: `ticker, side, action(OPEN/CLOSE/BUY/SELL), w_delta, est_px_ref, shares, est_notional`.
- No OMS integration; CSVs are designed for copy/paste into broker tickets.

## Phase 2 — Minimal State & Determinism
- Persist cohort state snapshots post-run (for audit and repeatability):
  - File: `state/cohort_{j}_{YYYY-MM}.json` with weights, durations, strikes, target counts.
- Deterministic recomputation remains primary; state is a convenience for quick diffs and manual reviews.

## Phase 3 — Lightweight Checks (Keep It Simple)
- Pre-export checks (fail with clear message, no heavy infra):
  - Universe size ≥ `min_eligible`; liquidity threshold met (63-day median volume).
  - Side gross within 0.50 ± tol after reweight; name caps/floors respected.
  - Cohort rotation and exit caps respected; turnover budget respected in non-rebalance months.
- Post-export sanity: totals per side, counts per cohort, sector shares per side within tolerance (only at full rebalance, as per spec).

## Phase 4 — Documentation & Runbook
- README “Operations” section: exact monthly cadence (form at calendar month-end; trade at next day’s open; 3-cohort rotation).
- Step-by-step runbook (10 lines max):
  1) Update DB/CDI paths and `--start/--end` if needed.
  2) Run the main script with `--cohorts 3 --beta-overlay`.
  3) Inspect console summary and plots.
  4) Review `targets_{YYYY-MM}.csv` and `orders_{YYYY-MM}.csv`.
  5) Enter trades manually at T+1 open; hedge per overlay if desired.
  6) Archive outputs and state snapshots.

## Out of Scope (by design)
- Any change to strategy logic or parameters (keep/add bands, adaptive caps, turnover budgets, overlay, VT all kept intact).
- OMS connectivity, borrow management, advanced risk dashboards.

## Acceptance Criteria
- Exact strategy outputs preserved (stats, turnover, overlays, VT) across runs.
- Trade-ready CSVs export consistently with the internal target weights used to compute returns.
- Minimal operator burden: one command + manual review + manual order entry.

## Implementation Tasks
- Add targets export aligned with internal cohort logic.
- Add optional orders helper (shares from weights) with simple assumptions.
- Persist minimal cohort state snapshots.
- Add pre/post checks and a short “Operations” section in README.
