#!/usr/bin/env python3
"""Calculate performance metrics for a specific period."""

import pandas as pd
import numpy as np

# Load the timeseries
df = pd.read_csv('results/momentum_br_timeseries.csv', index_col=0, parse_dates=True)

# Filter for 2015-2024 period
df_period = df[(df.index >= '2015-01-01') & (df.index <= '2024-12-31')]

# Calculate cumulative returns
def calculate_metrics(returns, name, start_date, end_date):
    # Cumulative wealth
    cumwealth = (1 + returns).cumprod()
    
    # CAGR
    n_periods = len(returns)
    years = n_periods / 12
    final_value = cumwealth.iloc[-1] if len(cumwealth) > 0 else 1.0
    cagr = (final_value ** (1/years) - 1) * 100 if years > 0 else 0
    
    # Annual volatility
    annual_vol = returns.std() * np.sqrt(12) * 100
    
    # Max drawdown
    running_max = cumwealth.cummax()
    drawdown = (cumwealth - running_max) / running_max
    max_dd = drawdown.min() * 100 if len(drawdown) > 0 else 0
    
    return {
        'Strategy': name,
        'Period': f'{start_date} to {end_date}',
        'CAGR': cagr,
        'Annual Vol': annual_vol,
        'Max DD': max_dd,
        'Final Value': final_value
    }

# Get period dates
start_date = df_period.index[0].strftime('%Y-%m')
end_date = df_period.index[-1].strftime('%Y-%m')

# Calculate for each series
ls_metrics = calculate_metrics(df_period['LS'].dropna(), 'LS (Full Spec)', start_date, end_date)
bova_metrics = calculate_metrics(df_period['BOVA11'].dropna(), 'BOVA11', start_date, end_date)
cdi_metrics = calculate_metrics(df_period['CDI'].dropna(), 'CDI', start_date, end_date)

# Print results
print('='*60)
print('PERFORMANCE METRICS (2015-2024)')
print('='*60)
for metrics in [ls_metrics, bova_metrics, cdi_metrics]:
    print(f"\n{metrics['Strategy']}:")
    print(f"  Period: {metrics['Period']}")
    print(f"  CAGR: {metrics['CAGR']:.2f}%")
    print(f"  Annual Volatility: {metrics['Annual Vol']:.2f}%")
    print(f"  Max Drawdown: {metrics['Max DD']:.2f}%")
    print(f"  $1 grew to: ${metrics['Final Value']:.3f}")

# Additional LS statistics
print('\n' + '='*60)
print('ADDITIONAL LS STRATEGY METRICS (2015-2024)')
print('='*60)
ls_returns = df_period['LS'].dropna()
positive_months = (ls_returns > 0).sum()
total_months = len(ls_returns)
hit_rate = positive_months / total_months * 100

print(f'Hit Rate: {hit_rate:.1f}% ({positive_months}/{total_months} months)')
print(f'Average Monthly Return: {ls_returns.mean()*100:.2f}%')
print(f'Best Month: {ls_returns.max()*100:.2f}%')
print(f'Worst Month: {ls_returns.min()*100:.2f}%')

# Sharpe ratio vs CDI
excess_returns = df_period['LS'] - df_period['CDI']
excess_returns = excess_returns.dropna()
sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(12) if excess_returns.std() > 0 else 0
print(f'Sharpe Ratio (vs CDI): {sharpe:.2f}')

# Performance comparison
print('\n' + '='*60)
print('RELATIVE PERFORMANCE (2015-2024)')
print('='*60)
print(f"LS vs BOVA11 CAGR spread: {ls_metrics['CAGR'] - bova_metrics['CAGR']:.2f}%")
print(f"LS vs CDI CAGR spread: {ls_metrics['CAGR'] - cdi_metrics['CAGR']:.2f}%")

# Risk-adjusted returns
ls_sharpe_ratio = ls_metrics['CAGR'] / ls_metrics['Annual Vol'] if ls_metrics['Annual Vol'] > 0 else 0
bova_sharpe_ratio = bova_metrics['CAGR'] / bova_metrics['Annual Vol'] if bova_metrics['Annual Vol'] > 0 else 0
print(f"\nRisk-Adjusted Return (CAGR/Vol):")
print(f"  LS: {ls_sharpe_ratio:.2f}")
print(f"  BOVA11: {bova_sharpe_ratio:.2f}")

# Year by year breakdown
print('\n' + '='*60)
print('YEAR-BY-YEAR RETURNS (2015-2024)')
print('='*60)
df_period['Year'] = df_period.index.year
yearly_returns = df_period.groupby('Year').apply(lambda x: (1 + x['LS']).prod() - 1) * 100
yearly_bova = df_period.groupby('Year').apply(lambda x: (1 + x['BOVA11']).prod() - 1) * 100
yearly_cdi = df_period.groupby('Year').apply(lambda x: (1 + x['CDI']).prod() - 1) * 100

print(f"{'Year':<6} {'LS':>10} {'BOVA11':>10} {'CDI':>10} {'LS-BOVA':>10}")
print("-" * 50)
for year in yearly_returns.index:
    ls_ret = yearly_returns[year]
    bova_ret = yearly_bova[year] if year in yearly_bova.index else 0
    cdi_ret = yearly_cdi[year] if year in yearly_cdi.index else 0
    spread = ls_ret - bova_ret
    print(f"{year:<6} {ls_ret:>9.1f}% {bova_ret:>9.1f}% {cdi_ret:>9.1f}% {spread:>9.1f}%")