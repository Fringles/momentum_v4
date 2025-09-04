#!/usr/bin/env python3
"""Calculate performance metrics for the momentum strategy."""

import pandas as pd
import numpy as np

# Load the timeseries
df = pd.read_csv('results/momentum_br_timeseries.csv', index_col=0, parse_dates=True)

# Calculate cumulative returns
def calculate_metrics(returns, name):
    # Cumulative wealth
    cumwealth = (1 + returns).cumprod()
    
    # CAGR
    n_periods = len(returns)
    years = n_periods / 12
    final_value = cumwealth.iloc[-1]
    cagr = (final_value ** (1/years) - 1) * 100
    
    # Annual volatility
    annual_vol = returns.std() * np.sqrt(12) * 100
    
    # Max drawdown
    running_max = cumwealth.cummax()
    drawdown = (cumwealth - running_max) / running_max
    max_dd = drawdown.min() * 100
    
    # Period
    start = df.index[0].strftime('%Y-%m')
    end = df.index[-1].strftime('%Y-%m')
    
    return {
        'Strategy': name,
        'Period': f'{start} to {end}',
        'CAGR': cagr,
        'Annual Vol': annual_vol,
        'Max DD': max_dd,
        'Final Value': final_value
    }

# Calculate for each series
ls_metrics = calculate_metrics(df['LS'].dropna(), 'LS (Full Spec)')
bova_metrics = calculate_metrics(df['BOVA11'].dropna(), 'BOVA11')
cdi_metrics = calculate_metrics(df['CDI'].dropna(), 'CDI')

# Print results
print('='*60)
print('PERFORMANCE METRICS (Full Period)')
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
print('ADDITIONAL LS STRATEGY METRICS')
print('='*60)
ls_returns = df['LS'].dropna()
positive_months = (ls_returns > 0).sum()
total_months = len(ls_returns)
hit_rate = positive_months / total_months * 100

print(f'Hit Rate: {hit_rate:.1f}% ({positive_months}/{total_months} months)')
print(f'Average Monthly Return: {ls_returns.mean()*100:.2f}%')
print(f'Best Month: {ls_returns.max()*100:.2f}%')
print(f'Worst Month: {ls_returns.min()*100:.2f}%')

# Sharpe ratio vs CDI
excess_returns = df['LS'] - df['CDI']
excess_returns = excess_returns.dropna()
sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(12)
print(f'Sharpe Ratio (vs CDI): {sharpe:.2f}')

# Performance comparison
print('\n' + '='*60)
print('RELATIVE PERFORMANCE')
print('='*60)
print(f"LS vs BOVA11 CAGR spread: {ls_metrics['CAGR'] - bova_metrics['CAGR']:.2f}%")
print(f"LS vs CDI CAGR spread: {ls_metrics['CAGR'] - cdi_metrics['CAGR']:.2f}%")

# Risk-adjusted returns
ls_sharpe_ratio = ls_metrics['CAGR'] / ls_metrics['Annual Vol']
bova_sharpe_ratio = bova_metrics['CAGR'] / bova_metrics['Annual Vol']
print(f"\nRisk-Adjusted Return (CAGR/Vol):")
print(f"  LS: {ls_sharpe_ratio:.2f}")
print(f"  BOVA11: {bova_sharpe_ratio:.2f}")