#!/usr/bin/env python3
"""
End-to-end testing for live trading workflow.
Tests all components of the momentum strategy live trading system.
"""

import os
import sys
import tempfile
import shutil
import pandas as pd
import json


def test_basic_strategy():
    """Test basic strategy execution."""
    print("\n" + "="*50)
    print("TEST 1: Basic Strategy Execution")
    print("="*50)
    
    # Check if we can import the strategy module
    try:
        sys.path.append('scripts')
        import momentum_br
        print("[PASS] Strategy module imports successfully")
        
        # Test CLI argument parsing
        parser = momentum_br.make_parser()
        args = parser.parse_args(['--db-path', 'dummy.db', '--cdi-path', 'dummy.xlsx'])
        print("[PASS] CLI argument parsing works")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Basic strategy test failed: {e}")
        return False


def test_validation():
    """Test the validation module."""
    print("\n" + "="*50)
    print("TEST 2: Validation Module")
    print("="*50)
    
    # Create sample targets file
    sample_targets = pd.DataFrame({
        'month_end': ['2024-01-31'] * 6,
        'trade_date': ['2024-02-01'] * 6,
        'cohort_id': [0, 0, 1, 1, 2, 2],
        'book': ['LS'] * 6,
        'ticker': ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'ABEV3', 'WEGE3'],
        'side': ['L', 'S', 'L', 'S', 'L', 'S'],
        'prev_weight': [0.0, 0.0, 0.02, -0.02, 0.01, -0.01],
        'target_weight': [0.025, -0.025, 0.025, -0.025, 0.02, -0.02],
        'delta_weight': [0.025, -0.025, 0.005, -0.005, 0.01, -0.01],
        'rationale': ['add', 'add', 'reweight', 'reweight', 'keep', 'keep'],
        'sector': ['Energy', 'Materials', 'Financials', 'Financials', 'Consumer', 'Industrials'],
        'duration_months': [0, 0, 3, 3, 5, 5],
        'strikes': [0, 0, 1, 1, 0, 0],
        'px_ref_open': [25.50, 45.20, 28.10, 22.75, 15.80, 35.40],
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_targets.to_csv(f.name, index=False)
        temp_file = f.name
    
    try:
        import validate_strategy
        
        is_valid, errors = validate_strategy.validate_pre_trade(temp_file)
        print(f"[PASS] Pre-trade validation completed: Valid={is_valid}, Errors={len(errors)}")
        
        report = validate_strategy.generate_validation_report(temp_file)
        print("[PASS] Validation report generated successfully")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Validation test failed: {e}")
        return False
    
    finally:
        os.unlink(temp_file)


def test_monitoring():
    """Test the monitoring module."""
    print("\n" + "="*50)
    print("TEST 3: Monitoring Module")
    print("="*50)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create sample state files
        for cohort_id in range(3):
            state_data = {
                "month_end": "2024-01-31",
                "trade_date": "2024-02-01",
                "cohort_id": cohort_id,
                "weights_ls": {
                    "PETR4": 0.025,
                    "VALE3": -0.025,
                    "ITUB4": 0.020,
                    "BBDC4": -0.020,
                },
                "weights_d10": {
                    "PETR4": 0.050,
                    "VALE3": 0.050,
                },
                "dur_long": {"PETR4": 2, "ITUB4": 5},
                "dur_short": {"VALE3": 3, "BBDC4": 1},
                "strikes_long": {"PETR4": 0, "ITUB4": 1},
                "strikes_short": {"VALE3": 0, "BBDC4": 0},
                "ls_target_counts": {"long": 25, "short": 25}
            }
            
            state_file = os.path.join(temp_dir, f"cohort_{cohort_id}_2024-01.json")
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
        
        import monitor_live_strategy
        
        positions = monitor_live_strategy.load_current_positions(temp_dir, "2024-01")
        print(f"[PASS] Position loading: Found {len(positions)} cohorts")
        
        current_prices = pd.Series({
            'PETR4': 25.50,
            'VALE3': 45.20,
            'ITUB4': 28.10,
            'BBDC4': 22.75,
        })
        
        config = {
            'target_gross_per_side': 0.50,
            'gross_tolerance': 0.02,
            'max_net_exposure': 0.05,
            'max_single_name_weight': 0.025,
        }
        
        alerts = monitor_live_strategy.check_risk_limits(positions, current_prices, config)
        print(f"[PASS] Risk limit checking: Generated {len(alerts)} alerts")
        
        report = monitor_live_strategy.generate_daily_report(temp_dir, "2024-01", None, config)
        print("[PASS] Daily report generated successfully")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Monitoring test failed: {e}")
        return False
    
    finally:
        shutil.rmtree(temp_dir)


def test_cli_flags():
    """Test new CLI flags."""
    print("\n" + "="*50)
    print("TEST 4: CLI Flags for Live Trading")
    print("="*50)
    
    try:
        import momentum_br
        
        parser = momentum_br.make_parser()
        
        test_args = [
            '--db-path', 'dummy.db',
            '--cdi-path', 'dummy.xlsx',
            '--live-capital', '1000000',
            '--lot-size', '100',
            '--write-state-snapshots',
        ]
        
        args = parser.parse_args(test_args)
        
        assert args.live_capital == 1000000, "live_capital not parsed correctly"
        assert args.lot_size == 100, "lot_size not parsed correctly"
        assert args.write_state_snapshots == True, "write_state_snapshots not parsed correctly"
        
        print("[PASS] All new CLI flags parsed correctly")
        
        config = momentum_br.Config(
            db_path=args.db_path,
            cdi_path=args.cdi_path,
            live_capital=args.live_capital,
            lot_size=args.lot_size,
            write_state_snapshots=args.write_state_snapshots,
        )
        
        print("[PASS] Config object created successfully with new parameters")
        return True
        
    except Exception as e:
        print(f"[FAIL] CLI flags test failed: {e}")
        return False


def test_file_structure():
    """Test file structure."""
    print("\n" + "="*50)
    print("TEST 5: Directory Structure and Files")
    print("="*50)
    
    required_files = [
        'scripts/momentum_br.py',
        'scripts/plot_momentum_br.py',
        'scripts/validate_strategy.py',
        'scripts/monitor_live_strategy.py',
        'README.md',
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"[PASS] Found: {file_path}")
    
    if missing_files:
        print(f"[FAIL] Missing files: {missing_files}")
        return False
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('state', exist_ok=True)
    print("[PASS] Required directories created/verified")
    
    return True


def main():
    """Run all tests."""
    print("MOMENTUM STRATEGY LIVE TRADING - END-TO-END TESTING")
    print("="*60)
    
    tests = [
        ("Basic Strategy Execution", test_basic_strategy),
        ("Validation Module", test_validation),
        ("Monitoring Module", test_monitoring),
        ("CLI Flags", test_cli_flags),
        ("Directory Structure", test_file_structure),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"[FAIL] {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nALL TESTS PASSED - Live trading system is ready!")
        return 0
    else:
        print(f"\n{total - passed} tests failed - Review issues before live trading")
        return 1


if __name__ == "__main__":
    sys.exit(main())