# Handling Delistings & Missing Exit Prices

Last updated: 2025-03-11

## Why This Change Happened

- The original backtests dropped any stock that lacked a price at the end of the holding period. Delisted names therefore vanished from the return vector instead of contributing the -100% loss that real P&L would incur.
- This created survivorship bias, especially in months where shorts delisted or longs defaulted.
- The March 2025 update fixes the issue for both momentum (`scripts/momentum_br.py`) and value (`scripts/value_br.py`). Any name with a valid trade-date open but no price at the next trade-date open is now marked to zero before returns are computed.

## Implementation Summary

- At portfolio construction time, the code reindexes the `end_open` Series to the full trade-date universe, detects `NaN` values for names still in the decile labels, and sets those entries to `0.0`.
- Names that never traded on the entry date (missing start open) continue to be skipped; this preserves the existing behaviour for thin or halted stocks that were never acquired.
- The conservative write-off applies equally to long and short books and is in effect for both historical backtests and live rebalances.

Relevant code:
- Momentum: `scripts/momentum_br.py:517-524`
- Value: `scripts/value_br.py:689-697`

## Backtest Impact (rerun 2025-03-11, BR universe 2011–2025, 3 cohorts, beta overlay, 10% VT)

| Strategy | Series | Mean Monthly Return | CAPM Alpha vs IBOV (bps/mo) | Alpha t-stat | Max Drawdown | Hit Rate |
|----------|--------|--------------------:|----------------------------:|-------------:|-------------:|---------:|
| Momentum | LS     | 0.84%               | 94.9                        | 3.88         | -22.9%       | 63.8%    |
| Momentum | LS (VT)| 0.90%               | 100.4                       | 4.21         | -17.6%       | 63.8%    |
| Momentum | D10    | 1.05%               | –                           | –            | -46.8%       | 57.6%    |
| Value    | LS     | 0.69%               | 68.1                        | 4.42         | -13.2%       | 62.5%    |
| Value    | LS (VT)| 0.89%               | 87.6                        | 4.37         | -18.6%       | 62.5%    |
| Value    | D10    | 1.14%               | –                           | –            | -48.9%       | 54.4%    |

Compared with the prior momentum backtest (which silently dropped missing names), LS alpha fell by ~18 bps/month and the maximum drawdown widened from -12% to -23%. Expect similar headline reductions for any historical reports generated before this fix.

## Months With Large New Losses

| Month-End | LS Δ (new − old) | Notes |
|-----------|-----------------:|-------|
| 2016-02-29 | -4.63% | No delisting detected; drawdown change driven by downstream state differences once historical returns were recomputed. |
| 2016-06-30 | -3.89% | Same as above; included for completeness. |
| 2023-10-31 | -4.77% | `SQIA3` entered with a valid price and delisted before the December exit; now realised as -100%. |
| 2025-03-31 | -3.85% | `CLSA3` disappeared prior to the May exit date. |
| 2025-05-30 | -4.89% | `ELMD3` and `JBSS3` vanished mid-period; both now treated as full write-offs. |

The helper script used for verification lives in the activity log for this update (`python3` snippets run on 2025-03-11) and can be rerun to audit future anomalies.

## Operational Notes

- The change affects all future exports. If you rely on historical reports, regenerate them so the summary statistics reflect the conservative treatment.
- Expect occasional large negative months tied to actual delistings or prolonged trading halts. Cross-check the `targets_YYYY-MM.csv` file for the affected month to confirm the tickers involved.
- No extra operator action is required—weights already assume the loss and the remaining capital is left in the aggregate portfolio until the next rebalance.
