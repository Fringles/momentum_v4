  📅 Monthly Trading Workflow

  When to Run Each Step

  1️⃣ Last Trading Day of Month (after market close)
  Run the strategy to generate targets:
  python3 scripts/momentum_br.py --cohorts 3 --beta-overlay \
    --live-capital 10000000 --write-state-snapshots \
    --db-path "your/database/path" --cdi-path "your/cdi/path"

  2️⃣ Same Day - Review & Validate
  python3 scripts/validate_strategy.py results/targets_2024-01.csv
  - Review the targets CSV and orders CSV
  - Check the console output for performance metrics
  - Verify no validation errors

  3️⃣ Next Trading Day (T+1) - Execute at Market Open
  - Use results/orders_YYYY-MM.csv to execute trades
  - Place all orders at market open prices
  - Apply beta hedge if needed (check hedge_ratio in timeseries)

  4️⃣ After Execution - Same Day
  - Document executed prices and fills
  - Archive the month's files
  - Run monitoring to confirm positions:
  python3 scripts/monitor_live_strategy.py report state 2024-01

  Then What? Daily & Monthly Maintenance

  📊 Daily (Optional but Recommended)
  - Monitor P&L and risk metrics:
  python3 scripts/monitor_live_strategy.py report state YYYY-MM
  - Check for any risk alerts
  - No action needed unless alerts trigger

  📈 Mid-Month - Nothing!
  - The strategy holds positions until next month-end
  - Cohorts automatically handle exits based on their schedule
  - No manual intervention required

  🔄 Next Month-End - Repeat
  - Update database with latest prices
  - Run the strategy command again
  - New targets will account for:
    - Existing positions (from state files)
    - Cohort rotation schedule
    - Market changes

  Important Notes

  Cohort Rotation Schedule:
  - Cohort 0: Full rebalance in Jan, Apr, Jul, Oct
  - Cohort 1: Full rebalance in Feb, May, Aug, Nov
  - Cohort 2: Full rebalance in Mar, Jun, Sep, Dec
  - Other cohorts only process exits (limited by adaptive caps)

  What Happens Automatically:
  - Position tracking via state files
  - Turnover control (targets ~30% monthly)
  - Beta hedging calculations
  - Sector rebalancing within tolerances

  Your Monthly Time Commitment:
  1. Month-end: ~30 mins to run strategy, validate, prepare orders
  2. T+1 morning: Execute trades at open
  3. Daily: 5 mins to check monitoring (optional)

  The strategy is designed to be mostly hands-off between monthly rebalances!