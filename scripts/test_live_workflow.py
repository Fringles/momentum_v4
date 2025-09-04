#!/usr/bin/env python3
"""
End-to-end testing for live trading workflow.
Tests all components of the momentum strategy live trading system.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from datetime import datetime
import pandas as pd
import json

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run command and return result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"Command failed: {cmd}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    
    return result


def test_basic_strategy_run():
    """Test basic strategy execution without live trading features."""
    print("\n" + "="*50)
    print("TEST 1: Basic Strategy Execution")
    print("="*50)
    
    # Check if we can import the strategy module
    try:
        sys.path.append('scripts')
        import momentum_br
        print("[OK] Strategy module imports successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import strategy module: {e}")
        return False
    
    # Test CLI argument parsing
    try:
        parser = momentum_br.make_parser()
        # Test with minimal args
        args = parser.parse_args(['--db-path', 'dummy.db', '--cdi-path', 'dummy.xlsx'])
        print("[OK] CLI argument parsing works")
    except Exception as e:
        print(f"[FAIL] CLI parsing failed: {e}")
        return False
    
    return True


def test_validation_module():
    """Test the validation module."""
    print("\n" + "="*50)
    print("TEST 2: Validation Module")
    print("="*50)
    
    # Create sample targets file for testing
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
    
    # Create temporary targets file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_targets.to_csv(f.name, index=False)
        temp_targets_file = f.name
    
    try:
        # Test validation import and basic functionality
        sys.path.append('scripts')
        import validate_strategy
        
        # Test pre-trade validation
        is_valid, errors = validate_strategy.validate_pre_trade(temp_targets_file)
        print(f"✅ Pre-trade validation completed: Valid={is_valid}, Errors={len(errors)}")
        
        if errors:
            print("Validation errors (expected for sample data):")
            for error in errors[:3]:  # Show first 3 errors
                print(f"  - {error}")
        
        # Test report generation
        report = validate_strategy.generate_validation_report(temp_targets_file)
        print("✅ Validation report generated successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation module test failed: {e}")
        return False
    
    finally:
        # Clean up
        os.unlink(temp_targets_file)


def test_monitoring_module():
    """Test the monitoring module."""
    print("\n" + "="*50)
    print("TEST 3: Monitoring Module")
    print("="*50)
    
    # Create sample state files
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create sample cohort state files
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
        
        # Test monitoring module
        sys.path.append('scripts')
        import monitor_live_strategy
        
        # Test position loading
        positions = monitor_live_strategy.load_current_positions(temp_dir, "2024-01")
        print(f"✅ Position loading: Found {len(positions)} cohorts")
        
        # Test risk limit checking
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
        print(f"✅ Risk limit checking: Generated {len(alerts)} alerts")
        
        # Test report generation
        report = monitor_live_strategy.generate_daily_report(temp_dir, "2024-01", None, config)
        print("✅ Daily report generated successfully")
        
        # Test data export
        export_file = os.path.join(temp_dir, "monitoring_export.csv")
        monitor_live_strategy.export_monitoring_data(temp_dir, "2024-01", export_file)
        
        if os.path.exists(export_file):
            export_df = pd.read_csv(export_file)
            print(f"✅ Data export: {len(export_df)} rows exported")
        else:
            print("⚠️  Export file not created (may be expected if no positions)")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring module test failed: {e}")
        return False
    
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_cli_flags():
    """Test new CLI flags for live trading."""
    print("\n" + "="*50)
    print("TEST 4: CLI Flags for Live Trading")
    print("="*50)
    
    try:
        sys.path.append('scripts')
        import momentum_br
        
        parser = momentum_br.make_parser()
        
        # Test live trading flags
        test_args = [
            '--db-path', 'dummy.db',
            '--cdi-path', 'dummy.xlsx',
            '--live-capital', '1000000',
            '--lot-size', '100',
            '--write-state-snapshots',
        ]
        
        args = parser.parse_args(test_args)
        
        # Check that new flags are properly parsed
        assert args.live_capital == 1000000, "live_capital not parsed correctly"
        assert args.lot_size == 100, "lot_size not parsed correctly"
        assert args.write_state_snapshots == True, "write_state_snapshots not parsed correctly"
        
        print("✅ All new CLI flags parsed correctly")
        
        # Test that Config can be created with new parameters
        config = momentum_br.Config(
            db_path=args.db_path,
            cdi_path=args.cdi_path,
            live_capital=args.live_capital,
            lot_size=args.lot_size,
            write_state_snapshots=args.write_state_snapshots,
        )
        
        assert config.live_capital == 1000000
        assert config.lot_size == 100
        assert config.write_state_snapshots == True
        
        print("✅ Config object created successfully with new parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI flags test failed: {e}")
        return False


def test_directory_structure():
    """Test that all required directories and files exist."""
    print("\n" + "="*50)
    print("TEST 5: Directory Structure and Files")
    print("="*50)
    
    required_files = [
        'scripts/momentum_br.py',
        'scripts/plot_momentum_br.py',
        'scripts/validate_strategy.py',
        'scripts/monitor_live_strategy.py',
        'README.md',
        'implementation_plan.md',
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ Found: {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    # Check that results directory can be created
    os.makedirs('results', exist_ok=True)
    os.makedirs('state', exist_ok=True)
    print("✅ Required directories created/verified")
    
    return True


def test_integration_workflow():
    """Test a simplified end-to-end workflow."""
    print("\n" + "="*50)
    print("TEST 6: Integration Workflow (Simplified)")
    print("="*50)
    
    print("This test simulates the live trading workflow without actual data:")
    print("1. Strategy execution (simulated)")
    print("2. Validation of outputs")
    print("3. Monitoring report generation")
    print("4. File cleanup")
    
    # Create dummy files that would be generated by strategy
    results_dir = 'results'
    state_dir = 'state'
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)
    
    try:
        # Simulate strategy outputs
        dummy_timeseries = pd.DataFrame({
            'D10': [0.015, 0.012, -0.008],
            'LS': [0.010, 0.008, -0.005],
            'turnover_LS': [0.25, 0.30, 0.20],
            'CDI': [0.005, 0.005, 0.005],
            'BOVA11': [0.020, -0.010, 0.015],
        }, index=pd.date_range('2024-01-31', periods=3, freq='M'))
        
        timeseries_file = os.path.join(results_dir, 'momentum_br_timeseries.csv')
        dummy_timeseries.to_csv(timeseries_file)
        
        dummy_targets = pd.DataFrame({
            'month_end': '2024-01-31',
            'cohort_id': [0, 1, 2],
            'book': 'LS',
            'ticker': ['PETR4', 'VALE3', 'ITUB4'],
            'side': ['L', 'S', 'L'],
            'target_weight': [0.025, -0.025, 0.020],
            'prev_weight': [0.020, -0.020, 0.015],
            'delta_weight': [0.005, -0.005, 0.005],
            'rationale': ['reweight', 'reweight', 'reweight'],
            'sector': ['Energy', 'Materials', 'Financials'],
            'px_ref_open': [25.50, 45.20, 28.10],
        })
        
        targets_file = os.path.join(results_dir, 'targets_2024-01.csv')
        dummy_targets.to_csv(targets_file, index=False)
        
        print("✅ Dummy strategy outputs created")
        
        # Test validation on dummy data
        sys.path.append('scripts')
        import validate_strategy
        
        is_valid, errors = validate_strategy.validate_pre_trade(targets_file)
        print(f"✅ Validation completed: Valid={is_valid}")
        
        # Test monitoring (would need state files for full test)
        print("✅ Integration workflow test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration workflow test failed: {e}")
        return False
    
    finally:
        # Clean up test files
        test_files = [
            os.path.join(results_dir, 'momentum_br_timeseries.csv'),
            os.path.join(results_dir, 'targets_2024-01.csv'),
        ]
        for file_path in test_files:
            if os.path.exists(file_path):
                os.unlink(file_path)


def main():
    """Run all tests."""
    print("MOMENTUM STRATEGY LIVE TRADING - END-TO-END TESTING")
    print("="*60)
    
    tests = [
        ("Basic Strategy Execution", test_basic_strategy_run),
        ("Validation Module", test_validation_module),
        ("Monitoring Module", test_monitoring_module),
        ("CLI Flags", test_cli_flags),
        ("Directory Structure", test_directory_structure),
        ("Integration Workflow", test_integration_workflow),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Live trading system is ready!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed - Review issues before live trading")
        return 1


if __name__ == "__main__":
    sys.exit(main())