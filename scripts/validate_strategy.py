#!/usr/bin/env python3
"""
Validation utilities for Brazil Momentum Strategy live trading.
Provides pre/post-trade checks to ensure strategy integrity.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime


def validate_pre_trade(targets_file: str, config: Optional[Dict] = None) -> Tuple[bool, List[str]]:
    """
    Pre-trade validation checks on targets file.
    
    Args:
        targets_file: Path to targets_YYYY-MM.csv
        config: Optional configuration dictionary with validation parameters
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    if not os.path.exists(targets_file):
        return False, [f"Targets file not found: {targets_file}"]
    
    try:
        df = pd.read_csv(targets_file)
        
        # Set default validation parameters
        default_config = {
            'min_universe_size': 50,
            'max_single_name_weight': 0.025,  # 2.5%
            'min_single_name_weight': 0.00125,  # 0.125% (0.25 * equal weight)
            'target_gross_per_side': 0.50,
            'gross_tolerance': 0.02,  # ±2%
            'max_sector_deviation': 0.10,  # ±10pp
            'expected_cohorts': 3,
            'max_turnover_per_cohort': 0.50,  # 50% per cohort
        }
        
        if config:
            default_config.update(config)
        cfg = default_config
        
        # Basic structure checks
        required_cols = ['cohort_id', 'book', 'ticker', 'side', 'target_weight', 
                        'prev_weight', 'delta_weight', 'rationale', 'sector']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return False, errors
        
        # Universe size check
        universe_size = len(df['ticker'].unique())
        if universe_size < cfg['min_universe_size']:
            errors.append(f"Universe size too small: {universe_size} < {cfg['min_universe_size']}")
        
        # Check cohort consistency
        cohorts = sorted(df['cohort_id'].unique())
        if len(cohorts) != cfg['expected_cohorts']:
            errors.append(f"Expected {cfg['expected_cohorts']} cohorts, found: {cohorts}")
        
        # Per-book validation
        for book in df['book'].unique():
            book_df = df[df['book'] == book].copy()
            
            # Weight constraints per cohort
            for cohort_id in book_df['cohort_id'].unique():
                cohort_df = book_df[book_df['cohort_id'] == cohort_id].copy()
                
                if book == 'LS':
                    # LS book: check long/short sides separately
                    for side in ['L', 'S']:
                        side_df = cohort_df[cohort_df['side'] == side].copy()
                        if side_df.empty:
                            continue
                        
                        # Gross exposure check
                        gross = abs(side_df['target_weight']).sum()
                        target_gross = cfg['target_gross_per_side']
                        tol = cfg['gross_tolerance']
                        
                        if not (target_gross - tol <= gross <= target_gross + tol):
                            errors.append(f"Cohort {cohort_id} {side} side gross {gross:.3f} outside tolerance "
                                         f"[{target_gross-tol:.3f}, {target_gross+tol:.3f}]")
                        
                        # Single name weight checks
                        abs_weights = abs(side_df['target_weight'])
                        n_names = len(side_df[side_df['target_weight'] != 0])
                        if n_names > 0:
                            equal_weight = target_gross / n_names
                            max_allowed = min(2 * equal_weight, cfg['max_single_name_weight'])
                            min_allowed = cfg['min_single_name_weight']
                            
                            over_limit = abs_weights > max_allowed
                            under_limit = (abs_weights > 0) & (abs_weights < min_allowed)
                            
                            if over_limit.any():
                                violators = side_df[over_limit]['ticker'].tolist()
                                errors.append(f"Cohort {cohort_id} {side} names over max weight {max_allowed:.3f}: {violators}")
                            
                            if under_limit.any():
                                violators = side_df[under_limit]['ticker'].tolist()
                                errors.append(f"Cohort {cohort_id} {side} names under min weight {min_allowed:.4f}: {violators}")
                
                elif book == 'D10':
                    # D10 book: long-only portfolio
                    gross = cohort_df['target_weight'].sum()
                    target_gross = cfg['target_gross_per_side']  # Same target as one LS side
                    tol = cfg['gross_tolerance']
                    
                    if not (target_gross - tol <= gross <= target_gross + tol):
                        errors.append(f"Cohort {cohort_id} D10 gross {gross:.3f} outside tolerance "
                                     f"[{target_gross-tol:.3f}, {target_gross+tol:.3f}]")
        
        # Turnover checks
        for cohort_id in df['cohort_id'].unique():
            cohort_df = df[df['cohort_id'] == cohort_id].copy()
            turnover = abs(cohort_df['delta_weight']).sum()
            
            if turnover > cfg['max_turnover_per_cohort']:
                errors.append(f"Cohort {cohort_id} turnover {turnover:.2%} exceeds limit {cfg['max_turnover_per_cohort']:.2%}")
        
        # Check for NaN values in critical fields
        critical_fields = ['target_weight', 'delta_weight']
        for field in critical_fields:
            nan_count = df[field].isna().sum()
            if nan_count > 0:
                errors.append(f"Found {nan_count} NaN values in {field}")
        
        # Check reference prices for orders generation
        if 'px_ref_open' in df.columns:
            missing_px = df['px_ref_open'].isna().sum()
            total_rows = len(df)
            if missing_px > total_rows * 0.05:  # More than 5% missing
                errors.append(f"Missing reference prices for {missing_px}/{total_rows} positions ({missing_px/total_rows:.1%})")
    
    except Exception as e:
        errors.append(f"Error reading/validating targets file: {e}")
    
    return len(errors) == 0, errors


def validate_post_trade(results_file: str, targets_file: str) -> Tuple[bool, List[str]]:
    """
    Post-trade validation comparing executed results with targets.
    
    Args:
        results_file: Path to execution results CSV
        targets_file: Path to original targets CSV
    
    Returns:
        (is_valid, warnings)
    """
    warnings = []
    
    if not os.path.exists(results_file):
        warnings.append(f"Results file not found: {results_file}")
        return False, warnings
    
    if not os.path.exists(targets_file):
        warnings.append(f"Targets file not found: {targets_file}")
        return False, warnings
    
    try:
        results_df = pd.read_csv(results_file)
        targets_df = pd.read_csv(targets_file)
        
        # Basic comparison - this would be customized based on execution system
        target_positions = len(targets_df[targets_df['target_weight'] != 0])
        if 'executed' in results_df.columns:
            executed_positions = len(results_df[results_df['executed'] == True])
            execution_rate = executed_positions / target_positions if target_positions > 0 else 0
            
            if execution_rate < 0.95:  # Less than 95% execution
                warnings.append(f"Low execution rate: {execution_rate:.1%} ({executed_positions}/{target_positions})")
        
        # Add more post-trade checks as needed based on execution system
        
    except Exception as e:
        warnings.append(f"Error in post-trade validation: {e}")
    
    return len(warnings) == 0, warnings


def validate_state_consistency(state_dir: str, period: str) -> Tuple[bool, List[str]]:
    """
    Validate cohort state consistency across files.
    
    Args:
        state_dir: Path to state directory
        period: Period string (YYYY-MM)
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    if not os.path.exists(state_dir):
        errors.append(f"State directory not found: {state_dir}")
        return False, errors
    
    try:
        cohort_files = [f for f in os.listdir(state_dir) if f.startswith(f"cohort_") and f.endswith(f"_{period}.json")]
        
        if not cohort_files:
            errors.append(f"No cohort state files found for period {period}")
            return False, errors
        
        cohort_states = {}
        for file in cohort_files:
            cohort_id = int(file.split('_')[1])
            with open(os.path.join(state_dir, file), 'r') as f:
                cohort_states[cohort_id] = json.load(f)
        
        # Basic consistency checks
        expected_cohorts = 3  # Default for strategy
        if len(cohort_states) != expected_cohorts:
            errors.append(f"Expected {expected_cohorts} cohorts, found {len(cohort_states)}")
        
        # Check all cohorts have same period
        periods = set(state['month_end'] for state in cohort_states.values())
        if len(periods) > 1:
            errors.append(f"Inconsistent periods across cohorts: {periods}")
        
        # Check for required fields in each cohort
        required_fields = ['weights_ls', 'weights_d10', 'dur_long', 'dur_short', 'strikes_long', 'strikes_short']
        for cohort_id, state in cohort_states.items():
            missing_fields = [field for field in required_fields if field not in state]
            if missing_fields:
                errors.append(f"Cohort {cohort_id} missing fields: {missing_fields}")
    
    except Exception as e:
        errors.append(f"Error validating state consistency: {e}")
    
    return len(errors) == 0, errors


def generate_validation_report(targets_file: str, config: Optional[Dict] = None) -> str:
    """Generate a comprehensive validation report."""
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("MOMENTUM STRATEGY VALIDATION REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Targets file: {targets_file}")
    report_lines.append("")
    
    # Pre-trade validation
    is_valid, errors = validate_pre_trade(targets_file, config)
    
    report_lines.append("PRE-TRADE VALIDATION")
    report_lines.append("-" * 20)
    if is_valid:
        report_lines.append("✓ All pre-trade checks PASSED")
    else:
        report_lines.append("✗ Pre-trade validation FAILED")
        for error in errors:
            report_lines.append(f"  - {error}")
    
    report_lines.append("")
    
    # Basic file statistics
    if os.path.exists(targets_file):
        try:
            df = pd.read_csv(targets_file)
            report_lines.append("FILE STATISTICS")
            report_lines.append("-" * 15)
            report_lines.append(f"Total positions: {len(df)}")
            report_lines.append(f"Unique tickers: {df['ticker'].nunique()}")
            report_lines.append(f"Cohorts: {sorted(df['cohort_id'].unique())}")
            report_lines.append(f"Books: {sorted(df['book'].unique())}")
            
            # Turnover by cohort
            turnover_by_cohort = df.groupby('cohort_id')['delta_weight'].apply(lambda x: abs(x).sum())
            report_lines.append("Turnover by cohort:")
            for cohort_id, turnover in turnover_by_cohort.items():
                report_lines.append(f"  Cohort {cohort_id}: {turnover:.2%}")
            
            # Action breakdown
            action_counts = df['rationale'].value_counts()
            report_lines.append("Actions:")
            for action, count in action_counts.items():
                report_lines.append(f"  {action}: {count}")
            
        except Exception as e:
            report_lines.append(f"Error generating statistics: {e}")
    
    report_lines.append("")
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validate_strategy.py <targets_file> [config_json]")
        print("Example: python validate_strategy.py results/targets_2024-01.csv")
        sys.exit(1)
    
    targets_file = sys.argv[1]
    config = None
    
    if len(sys.argv) > 2:
        with open(sys.argv[2], 'r') as f:
            config = json.load(f)
    
    # Generate and print validation report
    report = generate_validation_report(targets_file, config)
    print(report)
    
    # Exit with error code if validation failed
    is_valid, _ = validate_pre_trade(targets_file, config)
    sys.exit(0 if is_valid else 1)