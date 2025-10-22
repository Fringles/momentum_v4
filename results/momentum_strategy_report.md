
# Brazil Momentum Long/Short Strategy - Institutional Report

## Executive Summary
- Sector-neutral 12-2 month momentum across Brazilian equities, implemented as a three-cohort long/short portfolio with systematic guardrails, beta overlay, and 10% volatility targeting.
- Backtest (Jan 2011 - Aug 2025, 176 months) delivers ≈12.5% annualised return at ~10% realised volatility, Sharpe ≈0.37 vs CDI, and 103 bps/month CAPM alpha versus IBOV (t ≈ 4.21) with residual beta around -0.05.
- Strategy holds roughly 55-60 names per side (min 30, max 88) with ~35.9% average monthly turnover, split ~27.9% reconstitution and 8.0% reweights, aided by banded maintenance and a turnover budget.
- Risk controls include sector neutrality, single-name caps, adaptive exit bands, a composite beta overlay (average hedge ratio ≈10%) and volatility scaling (average multiplier 1.04, active 94% of months).
- Drawdown discipline: worst peak-to-trough loss of -13.7%, 66.7% positive months, and -0.12 correlation to IBOV; long-only top-decile sleeve compounds at ~13% but with 25% volatility and beta near 1.
- Net-of-cost stress test: assuming 20 / 50 / 100 bps round-trip costs per month, annualised returns remain ≈13.0% / 12.3% / 11.2% with information ratios 1.21 / 1.15 / 1.04, highlighting robustness to execution friction.

> **Update (Run 05 combined defaults, Oct-2025 rerun)**: LS momentum delivers ≈12.5%/yr at ~10% vol with 102.5 bps/mo CAPM alpha (t≈4.21, β≈-0.05) and -13.7% max drawdown; value sleeve delivers 96.8 bps/mo alpha (t≈5.08) at 27.7% turnover. The equal-weight combined LS sleeve prints ≈12.5%/yr at 6.8% vol with 102 bps/mo alpha (t≈7.18), 71% hit rate, and -5.3% max drawdown. See `results/combined_default_summary.md` for full metrics and equity curve.

## Strategy Blueprint
### Universe & Data Hygiene
- Source: SQLite daily database (daily_prices, securities, sectors) filtered to Brazilian common stocks; CDI reference curve from Excel.
- Liquidity screen: 63-day median traded value >= BRL 2mm, minimum 50 eligible names per rebalance.
- Price inputs: adjusted open/close, with portfolios formed on month-end data and executed next trading day (T+1) at the open.
- Corporate actions: handled via adjusted prices; sector mappings use modal sector assignment per ticker.

### Alpha Signal & Ranking
- Momentum metric: log-return between t-12 and t-2 month-end closes (12-2 specification) computed on monthly resampled adjusted closes.
- Within-sector z-scoring to neutralise sector composition before global ranking.
- Decile sorting: top decile defines momentum winners, bottom decile defines laggards; D10 sleeve tracked for long-only diagnostics.

### Portfolio Construction
- Long/short sleeves sized to 50% gross per side (~100% gross, 0% net before overlay), selecting equal-weighted names within the target deciles.
- Three staggered cohorts rebalance quarterly on a rolling basis (one cohort each month) to reduce turnover drift while maintaining monthly composite updates.
- Banded maintenance: keep longs while percentile ≥ 81.4 and shorts ≤ 18.6; adds only when percentile ≥ 90.3 or ≤ 9.7, respectively.
- Exposure guardrails:
  - Single-name max weight = min(2x equal weight, 2.5%), floor = 25% of equal weight.
  - Sector drift constrained to +/-9.2 percentage points per side relative to equal-weight sector share.
  - Side gross tolerance +/-2 percentage points around 50% triggered only when breached.
  - Adaptive exit caps on non-rebalance months: baseline 7.3% of book with adjustments for signal dispersion and reversal regime.
  - Turnover budget of 32% monthly enforced via severity queue before emergency overrides; micro-adds capped near 1.8% of names.

### Rebalance & Trade Workflow
- Calendar: determine exchange month-ends from trading calendar, trade on following business day (T+1), hold to next cohort event; live mode detects partial months and defers.
- Order generation: exports include cohort targets, account-level nets, live orders, allocation plans, and state JSON snapshots enabling auditability.
- Cold-start support: optional ramp fraction (default 1/cohort) produces parallel target and order files with zero previous weights.

## Risk Management Architecture
### Composite Beta Overlay
- Rolling 36-month lookback with 60 trading day EWMA halflife and 50% shrinkage estimates composite beta.
- Hedge ratio adjusts when |beta_hat| > 0.113 band; average realised hedge ratio +10%, range -14.8% to +35.9%.
- Overlay reduces raw beta from -0.16 pre-overlay to -0.06 post-overlay and improves CAPM alpha from 104 bps/month (t = 4.56) to 112 bps/month (t = 4.82).

### Volatility Targeting
- 36-month rolling realised volatility (minimum 12 months) scales hedged LS returns toward 10% annualised.
- Average scale factor 1.04 (min 0.70, max 1.36); active 94% of observations, indicating persistent under-target raw volatility.
- Vol targeting trims drawdown from -16.6% pre-scale to -12.5% while lifting annual return from 12.2% to 13.3%.

### Additional Controls
- Sector neutrality enforced each cohort; names receiving emergency reweights only when concentration breaches 70% effective N or >20% weights at floor.
- Turnover decomposed and monitored monthly, with alerts if gross >40% or reconstitution share >75% of turnover.
- Validation scripts (validate_strategy.py) verify weight sum, gross exposures, liquidity compliance, and state continuity before orders are released.

## Backtest Methodology
- Sample: Jan 2011 - Aug 2025 (176 rebalances) after liquidity gates; earlier data retained for signal warm-up but excluded from performance.
- Returns: monthly holding-period returns measured from T+1 open to next T+1 open; CDI series used for excess-return metrics; BOVA11 (IBOV ETF) as equity benchmark.
- Statistical inference: Newey-West (6 lags) for mean/t-stat; CAPM alpha with HAC errors; information ratio uses annualised tracking error of CAPM residuals.
- Transaction costs: base results are gross; separate grid applies 20/50/100 bps round-trip charges proportional to monthly turnover per side.
- No slippage, borrow cost, or financing charges included beyond cost grid; live-ready files assume share rounding per provided --lot-size.

## Performance Analytics
### Long/Short Composite (cohorts combined)
| Series | Ann. Return | Ann. Vol | Sharpe vs CDI | CAPM Alpha (bps/mo) | Alpha t-stat | Beta vs IBOV | Info Ratio | Hit Rate | Max Drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LS (overlay + VT) | 13.3% | 10.0% | 0.39 | 112 | 4.82 | -0.06 | 1.36 | 67.6% | -12.5% |
| LS pre VT (overlay only) | 12.2% | 10.1% | 0.29 | 104 | 4.59 | -0.06 | 1.25 | 67.6% | -16.6% |
| LS pre overlay | 11.2% | 10.4% | 0.20 | 104 | 4.56 | -0.16 | 1.27 | 67.0% | -17.4% |

- Sub-periods: 2011-2018 delivered 14.4% annual return at 10.5% vol (IR 1.4), while 2019-2025 posted 12.0% at 9.5% vol; both regimes kept beta near zero and drawdown below 12%.
- Correlation to IBOV: -0.13 post overlay, offering diversification for domestic equity books.
- Positive-month frequency: 67.6% overall, 25th/75th percentile monthly returns -1.5% / +3.6%.

### Long-Only and Benchmark Context
| Series | Ann. Return | Ann. Vol | Sharpe vs CDI | Beta vs IBOV |
| --- | --- | --- | --- | --- |
| D10 Winners (gross) | 14.5% | 25.3% | 0.31 | 0.99 |
| IBOV (BOVA11) | 5.0% | 23.0% | -0.08 | 1.00 |

- Long-only sleeve highlights raw momentum edge but carries benchmark-like beta and double the LS volatility.
- LS composite captures 92% of long-only alpha with 40% of the volatility and near-zero market exposure.

### Turnover & Capacity
- Average monthly LS turnover 35.9%, with ~78% attributable to reconstitutions and 22% to reweights; exit caps limit non-event churn to <=7.3% of book unless dispersion spikes.
- Average holdings: 56 longs / 54 shorts; cohorting ensures minimum 30 names per side, peaking at 88 when breadth is high.
- Liquidity buffer: median cohort name trades at least 15x expected dollar turnover (per px_ref_open and liquidity threshold), supporting institutional lot sizes.

### Transaction Cost Sensitivity (vol-targeted LS)
| Cost (bps round-trip) | Mean Monthly Return | Ann. Return | Info Ratio | Max Drawdown |
| --- | --- | --- | --- | --- |
| 20 | 1.05% | 13.4% | 1.31 | -10.5% |
| 50 | 1.00% | 12.7% | 1.25 | -10.8% |
| 100 | 0.91% | 11.5% | 1.14 | -11.4% |

- Even under 100 bps costs, alpha compresses by only roughly 190 bps/year with limited impact on drawdown, thanks to staggered cohorts and turnover budgeting.

## Operational Readiness
- results/ directory houses full target, order, and allocation exports for every month in sample alongside validation-ready summary files (momentum_br_timeseries.csv, momentum_br_summary.json, momentum_br_cost_grid.csv).
- Monitoring utilities (monitor_live_strategy.py) produce daily P&L, exposures, and alerting; state snapshots allow audit trail of cohort membership.
- Vol-target and overlay parameters accessible via CLI flags for scenario analysis (--vol-target-ann, --overlay-halflife-days, etc.).

## Considerations & Next Steps
- Data extensions: incorporating borrow cost estimates and realised execution slippage would refine net projections for live deployment.
- Stress testing: additional scenario work (for example, 2015-2016 recession, 2020 COVID shock) with daily path analysis could further evidence intramonth risk control.
- Capital deployment: evaluate capacity under larger gross exposures by stress-testing turnover and liquidity with 2-3x capital assumptions.
- Operational: integrate automated validation into trading stack and rehearse cold-start process to ensure readiness for new capital inflows.

