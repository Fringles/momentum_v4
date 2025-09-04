#!/usr/bin/env python3
"""
Live monitoring utilities for Brazil Momentum Strategy.
Provides real-time tracking, alerts, and performance attribution.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime, timedelta
import sqlite3


def load_current_positions(state_dir: str, period: str) -> Dict[str, Dict]:
    """Load current positions from state files."""
    positions = {}
    
    for cohort_id in range(3):  # Assuming 3 cohorts
        state_file = os.path.join(state_dir, f"cohort_{cohort_id}_{period}.json")
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                positions[f'cohort_{cohort_id}'] = json.load(f)
    
    return positions


def calculate_live_pnl(positions: Dict, current_prices: pd.Series, prev_prices: pd.Series) -> Dict:
    """Calculate P&L for current positions."""
    pnl_data = {
        'total_pnl': 0.0,
        'long_pnl': 0.0,
        'short_pnl': 0.0,
        'position_pnl': {},
        'cohort_pnl': {}
    }
    
    for cohort_key, cohort_data in positions.items():
        cohort_pnl = 0.0
        
        # LS positions
        ls_weights = cohort_data.get('weights_ls', {})
        for ticker, weight in ls_weights.items():
            if ticker in current_prices and ticker in prev_prices:
                price_return = (current_prices[ticker] / prev_prices[ticker]) - 1
                position_pnl = weight * price_return
                cohort_pnl += position_pnl
                
                pnl_data['position_pnl'][ticker] = pnl_data['position_pnl'].get(ticker, 0) + position_pnl
                
                if weight > 0:
                    pnl_data['long_pnl'] += position_pnl
                else:
                    pnl_data['short_pnl'] += position_pnl
        
        pnl_data['cohort_pnl'][cohort_key] = cohort_pnl
        pnl_data['total_pnl'] += cohort_pnl
    
    return pnl_data


def check_risk_limits(positions: Dict, current_prices: pd.Series, config: Dict) -> List[Dict]:
    """Check various risk limits and generate alerts."""
    alerts = []
    
    # Aggregate all positions across cohorts
    total_long = {}
    total_short = {}
    
    for cohort_data in positions.values():
        ls_weights = cohort_data.get('weights_ls', {})
        for ticker, weight in ls_weights.items():
            if weight > 0:
                total_long[ticker] = total_long.get(ticker, 0) + weight
            elif weight < 0:
                total_short[ticker] = total_short.get(ticker, 0) + weight
    
    # Calculate current portfolio metrics
    gross_long = sum(abs(w) for w in total_long.values())
    gross_short = sum(abs(w) for w in total_short.values())
    net_exposure = gross_long + gross_short  # short weights are negative
    
    # Gross exposure checks
    target_gross = config.get('target_gross_per_side', 0.50)
    gross_tolerance = config.get('gross_tolerance', 0.02)
    
    if gross_long > target_gross + gross_tolerance:
        alerts.append({
            'type': 'GROSS_EXPOSURE',
            'severity': 'WARNING',
            'message': f'Long side gross exposure {gross_long:.3f} above limit {target_gross + gross_tolerance:.3f}',
            'value': gross_long,
            'limit': target_gross + gross_tolerance
        })
    
    if abs(gross_short) > target_gross + gross_tolerance:
        alerts.append({
            'type': 'GROSS_EXPOSURE', 
            'severity': 'WARNING',
            'message': f'Short side gross exposure {abs(gross_short):.3f} above limit {target_gross + gross_tolerance:.3f}',
            'value': abs(gross_short),
            'limit': target_gross + gross_tolerance
        })
    
    # Net exposure check
    net_limit = config.get('max_net_exposure', 0.05)
    if abs(net_exposure) > net_limit:
        alerts.append({
            'type': 'NET_EXPOSURE',
            'severity': 'WARNING',
            'message': f'Net exposure {net_exposure:.3f} above limit {net_limit:.3f}',
            'value': net_exposure,
            'limit': net_limit
        })
    
    # Single name concentration checks
    max_single_weight = config.get('max_single_name_weight', 0.025)
    all_positions = {**total_long, **total_short}
    
    for ticker, weight in all_positions.items():
        if abs(weight) > max_single_weight:
            alerts.append({
                'type': 'CONCENTRATION',
                'severity': 'ERROR',
                'message': f'Position {ticker} weight {abs(weight):.3f} above limit {max_single_weight:.3f}',
                'ticker': ticker,
                'value': abs(weight),
                'limit': max_single_weight
            })
    
    # Check for missing prices
    missing_prices = []
    for ticker in all_positions.keys():
        if ticker not in current_prices or pd.isna(current_prices.get(ticker)):
            missing_prices.append(ticker)
    
    if missing_prices:
        alerts.append({
            'type': 'DATA_QUALITY',
            'severity': 'WARNING',
            'message': f'Missing current prices for {len(missing_prices)} positions: {missing_prices[:5]}' + ('...' if len(missing_prices) > 5 else ''),
            'tickers': missing_prices
        })
    
    return alerts


def generate_daily_report(state_dir: str, period: str, prices_db: str, config: Optional[Dict] = None) -> str:
    """Generate daily monitoring report."""
    
    if config is None:
        config = {
            'target_gross_per_side': 0.50,
            'gross_tolerance': 0.02,
            'max_net_exposure': 0.05,
            'max_single_name_weight': 0.025,
        }
    
    report_lines = []
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report_lines.append("=" * 60)
    report_lines.append("MOMENTUM STRATEGY DAILY MONITORING REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Generated: {timestamp}")
    report_lines.append(f"Period: {period}")
    report_lines.append("")
    
    try:
        # Load current positions
        positions = load_current_positions(state_dir, period)
        
        if not positions:
            report_lines.append("⚠️  No position data found")
            return "\n".join(report_lines)
        
        # Get current prices (simplified - would connect to real price feed)
        # For demo, we'll simulate getting latest prices from database
        current_prices = pd.Series()  # Would be populated from real data source
        prev_prices = pd.Series()     # Would be populated from real data source
        
        # Position summary
        report_lines.append("POSITION SUMMARY")
        report_lines.append("-" * 16)
        
        total_positions = 0
        total_long_gross = 0
        total_short_gross = 0
        
        for cohort_key, cohort_data in positions.items():
            ls_weights = cohort_data.get('weights_ls', {})
            cohort_positions = len([w for w in ls_weights.values() if w != 0])
            cohort_long = sum(w for w in ls_weights.values() if w > 0)
            cohort_short = sum(abs(w) for w in ls_weights.values() if w < 0)
            
            total_positions += cohort_positions
            total_long_gross += cohort_long
            total_short_gross += cohort_short
            
            report_lines.append(f"{cohort_key}: {cohort_positions} positions, Long {cohort_long:.3f}, Short {cohort_short:.3f}")
        
        report_lines.append(f"TOTAL: {total_positions} positions, Long {total_long_gross:.3f}, Short {total_short_gross:.3f}")
        net_exposure = total_long_gross - total_short_gross
        report_lines.append(f"Net exposure: {net_exposure:.3f}")
        report_lines.append("")
        
        # Risk checks
        alerts = check_risk_limits(positions, current_prices, config)
        
        if alerts:
            report_lines.append("🚨 ALERTS")
            report_lines.append("-" * 10)
            for alert in alerts:
                severity_icon = "❌" if alert['severity'] == 'ERROR' else "⚠️"
                report_lines.append(f"{severity_icon} {alert['type']}: {alert['message']}")
            report_lines.append("")
        else:
            report_lines.append("✅ No risk alerts")
            report_lines.append("")
        
        # P&L summary (if price data available)
        if not current_prices.empty and not prev_prices.empty:
            pnl_data = calculate_live_pnl(positions, current_prices, prev_prices)
            
            report_lines.append("P&L SUMMARY")
            report_lines.append("-" * 11)
            report_lines.append(f"Total P&L: {pnl_data['total_pnl']:.4f}")
            report_lines.append(f"Long P&L:  {pnl_data['long_pnl']:.4f}")
            report_lines.append(f"Short P&L: {pnl_data['short_pnl']:.4f}")
            report_lines.append("")
            
            # Top contributors
            position_pnl = pnl_data['position_pnl']
            if position_pnl:
                top_contributors = sorted(position_pnl.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                report_lines.append("Top P&L contributors:")
                for ticker, pnl in top_contributors:
                    report_lines.append(f"  {ticker}: {pnl:+.4f}")
                report_lines.append("")
        
        # Cohort status
        report_lines.append("COHORT STATUS")
        report_lines.append("-" * 13)
        current_month = datetime.now().month
        rebalance_months = {0: [1, 4, 7, 10], 1: [2, 5, 8, 11], 2: [3, 6, 9, 12]}
        
        for i, (cohort_key, cohort_data) in enumerate(positions.items()):
            is_rebalance_month = current_month in rebalance_months.get(i, [])
            status = "REBALANCE" if is_rebalance_month else "MAINTENANCE"
            
            # Count positions by duration
            dur_long = cohort_data.get('dur_long', {})
            dur_short = cohort_data.get('dur_short', {})
            all_durations = list(dur_long.values()) + list(dur_short.values())
            avg_duration = np.mean(all_durations) if all_durations else 0
            
            report_lines.append(f"Cohort {i}: {status}, Avg holding: {avg_duration:.1f} months")
        
        report_lines.append("")
        
        # Operational notes
        report_lines.append("OPERATIONAL NOTES")
        report_lines.append("-" * 17)
        
        next_rebalance = None
        for cohort_id, months in rebalance_months.items():
            for month in sorted(months):
                if month > current_month:
                    next_rebalance = f"Cohort {cohort_id} in {datetime(2024, month, 1).strftime('%B')}"
                    break
            if next_rebalance:
                break
        
        if not next_rebalance:
            next_rebalance = f"Cohort 0 in January (next year)"
        
        report_lines.append(f"Next rebalance: {next_rebalance}")
        report_lines.append("Monitor for exit triggers on non-rebalance cohorts")
        report_lines.append("Check beta overlay hedge ratio in latest timeseries")
        
    except Exception as e:
        report_lines.append(f"Error generating report: {e}")
    
    report_lines.append("")
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)


def calculate_attribution(timeseries_file: str, period_start: str, period_end: str) -> pd.DataFrame:
    """Calculate performance attribution over a period."""
    
    if not os.path.exists(timeseries_file):
        return pd.DataFrame()
    
    df = pd.read_csv(timeseries_file, index_col=0, parse_dates=True)
    
    # Filter for period
    mask = (df.index >= period_start) & (df.index <= period_end)
    period_data = df[mask].copy()
    
    if period_data.empty:
        return pd.DataFrame()
    
    # Calculate attribution components
    attribution = []
    
    for date, row in period_data.iterrows():
        ls_ret = row.get('LS', 0)
        d10_ret = row.get('D10', 0)
        cdi_ret = row.get('CDI', 0)
        ibov_ret = row.get('BOVA11', 0)
        
        # Decompose LS return (simplified)
        excess_vs_cdi = ls_ret - cdi_ret
        excess_vs_ibov = ls_ret - ibov_ret
        
        attribution.append({
            'date': date,
            'LS_return': ls_ret,
            'D10_return': d10_ret,
            'excess_vs_CDI': excess_vs_cdi,
            'excess_vs_IBOV': excess_vs_ibov,
            'turnover_LS': row.get('turnover_LS', 0),
        })
    
    return pd.DataFrame(attribution)


def export_monitoring_data(state_dir: str, period: str, output_file: str):
    """Export position and monitoring data to CSV for external analysis."""
    
    positions = load_current_positions(state_dir, period)
    
    rows = []
    for cohort_key, cohort_data in positions.items():
        cohort_id = cohort_key.split('_')[1]
        
        # LS positions
        ls_weights = cohort_data.get('weights_ls', {})
        for ticker, weight in ls_weights.items():
            if weight != 0:
                side = 'Long' if weight > 0 else 'Short'
                duration = cohort_data.get('dur_long', {}).get(ticker, 0) if weight > 0 else cohort_data.get('dur_short', {}).get(ticker, 0)
                strikes = cohort_data.get('strikes_long', {}).get(ticker, 0) if weight > 0 else cohort_data.get('strikes_short', {}).get(ticker, 0)
                
                rows.append({
                    'cohort_id': cohort_id,
                    'book': 'LS',
                    'ticker': ticker,
                    'side': side,
                    'weight': abs(weight),
                    'duration_months': duration,
                    'strikes': strikes,
                    'period': period,
                    'export_timestamp': datetime.now().isoformat(),
                })
    
    if rows:
        pd.DataFrame(rows).to_csv(output_file, index=False)
        print(f"Monitoring data exported to {output_file}")
    else:
        print("No position data to export")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python monitor_live_strategy.py <command> <args>")
        print("Commands:")
        print("  report <state_dir> <period> [prices_db] - Generate daily report")
        print("  export <state_dir> <period> <output_file> - Export monitoring data")
        print("  attribution <timeseries_file> <start> <end> - Calculate attribution")
        print("\nExample:")
        print("  python monitor_live_strategy.py report state 2024-01")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'report':
        state_dir = sys.argv[2]
        period = sys.argv[3]
        prices_db = sys.argv[4] if len(sys.argv) > 4 else None
        
        report = generate_daily_report(state_dir, period, prices_db)
        print(report)
    
    elif command == 'export':
        state_dir = sys.argv[2]
        period = sys.argv[3]
        output_file = sys.argv[4]
        
        export_monitoring_data(state_dir, period, output_file)
    
    elif command == 'attribution':
        timeseries_file = sys.argv[2]
        start_date = sys.argv[3]
        end_date = sys.argv[4]
        
        attr = calculate_attribution(timeseries_file, start_date, end_date)
        if not attr.empty:
            print(attr.to_string(index=False))
        else:
            print("No data available for attribution analysis")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)