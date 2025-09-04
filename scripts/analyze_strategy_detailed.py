#!/usr/bin/env python3
"""Detailed strategy analysis for 2015-2024 period."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load the timeseries
df = pd.read_csv('results/momentum_br_timeseries.csv', index_col=0, parse_dates=True)

# Filter for 2015-2024 period
df_period = df[(df.index >= '2015-01-01') & (df.index <= '2024-12-31')]

print('='*60)
print('DETAILED STRATEGY METRICS (2015-2024)')
print('='*60)

# 1. CAPM Alpha vs IBOV with HAC t-stat
ls_returns = df_period['LS'].dropna()
ibov_returns = df_period['BOVA11'].dropna()

# Align the series
aligned_data = pd.concat([ls_returns, ibov_returns], axis=1, keys=['LS', 'IBOV']).dropna()

# Run CAPM regression with HAC standard errors
X = sm.add_constant(aligned_data['IBOV'].values)
y = aligned_data['LS'].values
model = sm.OLS(y, X)
results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 6})

alpha_monthly = results.params[0]
alpha_bps = alpha_monthly * 10000  # Convert to basis points
alpha_t = results.tvalues[0]
beta = results.params[1]

print(f"\n1. CAPM ALPHA vs IBOV")
print(f"   Alpha: {alpha_bps:.1f} bps/month")
print(f"   HAC t-stat: {alpha_t:.2f}")
print(f"   Beta: {beta:.3f}")

# 2. Information Ratio (annualized)
residuals = results.resid
tracking_error = residuals.std() * np.sqrt(12)
alpha_annual = alpha_monthly * 12
ir_annual = alpha_annual / tracking_error if tracking_error > 0 else 0

print(f"\n2. INFORMATION RATIO")
print(f"   IR (annualized): {ir_annual:.2f}")
print(f"   Tracking Error (ann.): {tracking_error:.1%}")

# 3. Residual Beta (rolling analysis)
print(f"\n3. RESIDUAL BETA ANALYSIS")

# Calculate rolling 12-month betas
window = 12
rolling_betas = []
for i in range(window, len(aligned_data)):
    y_window = aligned_data['LS'].iloc[i-window:i].values
    x_window = sm.add_constant(aligned_data['IBOV'].iloc[i-window:i].values)
    try:
        model_window = sm.OLS(y_window, x_window)
        results_window = model_window.fit()
        rolling_betas.append(results_window.params[1])
    except:
        pass

if rolling_betas:
    beta_median = np.median(rolling_betas)
    beta_5th = np.percentile(rolling_betas, 2.5)
    beta_95th = np.percentile(rolling_betas, 97.5)
    print(f"   Median Beta: {beta_median:.3f}")
    print(f"   95% Band: [{beta_5th:.3f}, {beta_95th:.3f}]")

# 4. Maximum Drawdown
cumulative = (1 + ls_returns).cumprod()
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
max_dd = drawdown.min()

print(f"\n4. MAXIMUM DRAWDOWN")
print(f"   Max DD: {max_dd:.2%}")
print(f"   Max DD Date: {drawdown.idxmin().strftime('%Y-%m-%d')}")

# 5. Turnover Analysis
turnover_ls = df_period['turnover_LS'].dropna()
turnover_reconst = df_period['turnover_LS_reconst'].dropna()
turnover_reweight = df_period['turnover_LS_reweight'].dropna()

avg_turnover_total = turnover_ls.mean()
avg_turnover_reconst = turnover_reconst.mean()
avg_turnover_reweight = turnover_reweight.mean()

# Alpha per 1x turnover
alpha_per_turnover = alpha_bps / (avg_turnover_total * 100) if avg_turnover_total > 0 else 0

print(f"\n5. TURNOVER ANALYSIS")
print(f"   Average Monthly Turnover:")
print(f"     Total: {avg_turnover_total:.1%}")
print(f"     Reconstitution: {avg_turnover_reconst:.1%} ({avg_turnover_reconst/avg_turnover_total:.0%} of total)")
print(f"     Reweighting: {avg_turnover_reweight:.1%} ({avg_turnover_reweight/avg_turnover_total:.0%} of total)")
print(f"   Alpha per 1× turnover: {alpha_per_turnover:.1f} bps")

# 6. Worst 5 Drawdowns
print(f"\n6. WORST 5 DRAWDOWNS")

# Find drawdown periods
dd_periods = []
in_dd = False
start_idx = None

for i in range(len(drawdown)):
    if drawdown.iloc[i] < 0 and not in_dd:
        in_dd = True
        start_idx = i
    elif drawdown.iloc[i] == 0 and in_dd:
        in_dd = False
        if start_idx is not None:
            end_idx = i
            min_dd_idx = drawdown.iloc[start_idx:end_idx].idxmin()
            min_dd = drawdown.loc[min_dd_idx]
            dd_periods.append({
                'start': drawdown.index[start_idx],
                'trough': min_dd_idx,
                'end': drawdown.index[end_idx] if end_idx < len(drawdown) else drawdown.index[-1],
                'depth': min_dd
            })

# Handle ongoing drawdown at end
if in_dd and start_idx is not None:
    min_dd_idx = drawdown.iloc[start_idx:].idxmin()
    min_dd = drawdown.loc[min_dd_idx]
    dd_periods.append({
        'start': drawdown.index[start_idx],
        'trough': min_dd_idx,
        'end': drawdown.index[-1],
        'depth': min_dd
    })

# Sort by depth and take worst 5
dd_periods_sorted = sorted(dd_periods, key=lambda x: x['depth'])[:5]

for i, dd in enumerate(dd_periods_sorted, 1):
    duration = (dd['end'] - dd['start']).days
    print(f"   #{i}: {dd['depth']:.2%} depth")
    print(f"       Start: {dd['start'].strftime('%Y-%m-%d')}")
    print(f"       Trough: {dd['trough'].strftime('%Y-%m-%d')}")
    print(f"       End: {dd['end'].strftime('%Y-%m-%d')}")
    print(f"       Duration: {duration} days")

# 7. Additional Risk Metrics
print(f"\n7. ADDITIONAL RISK METRICS")

# Sharpe Ratio
excess_vs_cdi = df_period['LS'] - df_period['CDI']
excess_vs_cdi = excess_vs_cdi.dropna()
sharpe = excess_vs_cdi.mean() / excess_vs_cdi.std() * np.sqrt(12) if excess_vs_cdi.std() > 0 else 0
print(f"   Sharpe Ratio (vs CDI): {sharpe:.2f}")

# Sortino Ratio (downside deviation)
downside_returns = ls_returns[ls_returns < 0]
downside_dev = downside_returns.std() * np.sqrt(12)
sortino = (ls_returns.mean() * 12) / downside_dev if downside_dev > 0 else 0
print(f"   Sortino Ratio: {sortino:.2f}")

# Calmar Ratio (CAGR / Max DD)
cagr = ((cumulative.iloc[-1] ** (12/len(ls_returns))) - 1)
calmar = cagr / abs(max_dd) if max_dd != 0 else 0
print(f"   Calmar Ratio: {calmar:.2f}")

# Hit Rate
hit_rate = (ls_returns > 0).sum() / len(ls_returns) * 100
print(f"   Hit Rate: {hit_rate:.1f}%")

# 8. Market Regime Analysis
print(f"\n8. PERFORMANCE BY MARKET REGIME")

# Bull vs Bear months (IBOV > 0 vs IBOV < 0)
bull_months = aligned_data[aligned_data['IBOV'] > 0]
bear_months = aligned_data[aligned_data['IBOV'] < 0]

if len(bull_months) > 0:
    bull_avg = bull_months['LS'].mean() * 100
    print(f"   Bull Markets (IBOV > 0): {bull_avg:.2f}% avg return ({len(bull_months)} months)")

if len(bear_months) > 0:
    bear_avg = bear_months['LS'].mean() * 100
    print(f"   Bear Markets (IBOV < 0): {bear_avg:.2f}% avg return ({len(bear_months)} months)")

# High vs Low volatility regimes
ibov_vol = aligned_data['IBOV'].rolling(12).std()
median_vol = ibov_vol.median()
high_vol = aligned_data[ibov_vol > median_vol]
low_vol = aligned_data[ibov_vol <= median_vol]

if len(high_vol) > 0:
    high_vol_avg = high_vol['LS'].mean() * 100
    print(f"   High Vol Regime: {high_vol_avg:.2f}% avg return ({len(high_vol)} months)")

if len(low_vol) > 0:
    low_vol_avg = low_vol['LS'].mean() * 100
    print(f"   Low Vol Regime: {low_vol_avg:.2f}% avg return ({len(low_vol)} months)")