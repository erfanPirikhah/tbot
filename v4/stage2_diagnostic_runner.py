#!/usr/bin/env python3
"""
Diagnostic backtest runner for Stage 2 - Deep Diagnostic Mapping
"""

import sys
import os
from datetime import datetime, timedelta
import json
import pandas as pd

# Add project path
project_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_path)

from providers.data_provider import ImprovedSimulatedProvider
from diagnostic.diagnostic_strategy_v2 import DiagnosticEnhancedRsiStrategyV5
from config.parameters import CONSERVATIVE_PARAMS


def run_diagnostic_backtest():
    """Run diagnostic backtest to map signal flow and identify blocking points"""
    
    print("STAGE 2 - STEP 1: Deep Diagnostic Mapping")
    print("="*70)
    
    # Generate simulated data
    print("Generating simulated data...")
    provider = ImprovedSimulatedProvider(seed=42)
    data = provider.fetch_data("BTCUSDT", "H1", 100)  # Use 100 candles for good analysis
    print(f"Generated {len(data)} candles")
    
    # Test with normal parameters
    print("\nTesting with Normal Mode...")
    normal_params = CONSERVATIVE_PARAMS.copy()
    normal_params.update({
        'test_mode_enabled': False,
        'bypass_contradiction_detection': False,
    })
    
    strategy_normal = DiagnosticEnhancedRsiStrategyV5(**normal_params)
    
    # Run simulation
    portfolio_values = []
    for i in range(50, len(data)):
        current_data = data.iloc[:i+1].copy()
        signal = strategy_normal.generate_signal(current_data, i)
        
        portfolio_values.append({
            'timestamp': current_data.index[-1],
            'portfolio_value': strategy_normal._portfolio_value,
            'price': current_data['close'].iloc[-1],
            'signal': signal.get('action', 'HOLD')
        })
    
    # Get diagnostic report for normal mode
    normal_report = strategy_normal.get_diagnostic_report()
    print(f"\nNORMAL MODE DIAGNOSTIC REPORT:")
    print(f"   Total Signals Checked: {normal_report['summary']['total_signals_checked']}")
    print(f"   Successful Entries: {normal_report['summary']['successful_entries']}")
    print(f"   Total Exits: {normal_report['summary']['total_exits']}")
    print("\n   Blocking Analysis:")
    for module, count in normal_report['blocking_analysis'].items():
        if count > 0:
            pct = normal_report['blocking_percentages'].get(f"{module}_pct", 0)
            print(f"     {module}: {count} blocks ({pct:.1f}%)")
    
    # Test with TestMode
    print("\nTesting with Test Mode...")
    test_params = CONSERVATIVE_PARAMS.copy()
    test_params.update({
        'test_mode_enabled': True,
        'bypass_contradiction_detection': True,
        'relax_risk_filters': True,
        'relax_entry_conditions': True,
        'rsi_oversold': 40,  # More permissive
        'rsi_overbought': 60,  # More permissive
        'rsi_entry_buffer': 10,  # Very loose
        'enable_trend_filter': False,  # Disable for test
        'enable_mtf': False,  # Disable for test
        'min_candles_between': 1,  # Minimum spacing
    })
    
    strategy_test = DiagnosticEnhancedRsiStrategyV5(**test_params)
    
    # Run simulation
    for i in range(50, len(data)):
        current_data = data.iloc[:i+1].copy()
        signal = strategy_test.generate_signal(current_data, i)
        
        # We just want to see if signals are generated
        pass
    
    # Get diagnostic report for test mode
    test_report = strategy_test.get_diagnostic_report()
    print(f"\nTEST MODE DIAGNOSTIC REPORT:")
    print(f"   Total Signals Checked: {test_report['summary']['total_signals_checked']}")
    print(f"   Successful Entries: {test_report['summary']['successful_entries']}")
    print(f"   Total Exits: {test_report['summary']['total_exits']}")
    print("\n   Blocking Analysis:")
    for module, count in test_report['blocking_analysis'].items():
        if count > 0:
            pct = test_report['blocking_percentages'].get(f"{module}_pct", 0)
            print(f"     {module}: {count} blocks ({pct:.1f}%)")
    
    # Rank modules by blocking percentage for normal mode
    print(f"\nMODULE BLOCKING RANKING (Normal Mode):")
    blocking_data = []
    for module, count in normal_report['blocking_analysis'].items():
        pct = normal_report['blocking_percentages'].get(f"{module}_pct", 0)
        if pct > 0:  # Only show modules that are actually blocking
            blocking_data.append((module, count, pct))
    
    # Sort by percentage blocked
    blocking_data.sort(key=lambda x: x[2], reverse=True)
    
    for i, (module, count, pct) in enumerate(blocking_data, 1):
        print(f"   {i}. {module}: {pct:.1f}% blocking ({count} of {normal_report['summary']['total_signals_checked']} signals)")
    
    # Create diagnostic report
    diagnostic_report = {
        "stage": "Stage 2 - Step 1: Deep Diagnostic Mapping",
        "execution_time": datetime.now().isoformat(),
        "data_summary": {
            "symbol": "BTCUSDT",
            "timeframe": "H1", 
            "candle_count": len(data),
            "date_range": {
                "start": str(data.index[0]),
                "end": str(data.index[-1])
            }
        },
        "normal_mode_analysis": {
            "parameters": {k: v for k, v in normal_params.items() if k in ['test_mode_enabled', 'rsi_oversold', 'rsi_overbought', 'rsi_entry_buffer', 'enable_trend_filter', 'enable_mtf', 'min_candles_between']},
            "results": normal_report
        },
        "test_mode_analysis": {
            "parameters": {k: v for k, v in test_params.items() if k in ['test_mode_enabled', 'rsi_oversold', 'rsi_overbought', 'rsi_entry_buffer', 'enable_trend_filter', 'enable_mtf', 'min_candles_between']},
            "results": test_report
        },
        "module_blocking_ranking": [
            {"rank": i+1, "module": module, "percentage_blocked": pct, "count_blocked": count}
            for i, (module, count, pct) in enumerate(blocking_data)
        ],
        "recommendations": []
    }
    
    # Add recommendations based on findings
    if blocking_data:
        top_blocker = blocking_data[0]
        if 'rsi' in top_blocker[0]:
            diagnostic_report["recommendations"].append("RSI module is the primary blocker - investigate oversold/overbought thresholds")
        elif 'momentum' in top_blocker[0]:
            diagnostic_report["recommendations"].append("Momentum module is blocking too many signals - adjust momentum thresholds")
        elif 'trend' in top_blocker[0]:
            diagnostic_report["recommendations"].append("Trend filter is too restrictive - review trend strength requirements")
        elif 'mtf' in top_blocker[0]:
            diagnostic_report["recommendations"].append("MTF filter is blocking signals - check alignment logic")
        elif 'contradiction' in top_blocker[0]:
            diagnostic_report["recommendations"].append("Contradiction system is overly aggressive - review contradiction thresholds")
    else:
        diagnostic_report["recommendations"].append("No major blocking modules identified - may be position spacing or other issues")
    
    return diagnostic_report, normal_report, test_report


def save_diagnostic_results(diagnostic_report):
    """Save diagnostic results to files"""
    
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", "stage2", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save the main diagnostic report
    with open(os.path.join(results_dir, "strategy_diagnostic_report.md"), 'w') as f:
        f.write("# STAGE 2 - STEP 1: Deep Diagnostic Mapping Report\n\n")
        f.write(f"**Generated:** {diagnostic_report['execution_time']}\n\n")
        f.write("## Data Summary\n")
        data_summary = diagnostic_report['data_summary']
        f.write(f"- Symbol: {data_summary['symbol']}\n")
        f.write(f"- Timeframe: {data_summary['timeframe']}\n")
        f.write(f"- Candles: {data_summary['candle_count']}\n")
        f.write(f"- Date Range: {data_summary['date_range']['start']} to {data_summary['date_range']['end']}\n\n")
        
        f.write("## Normal Mode Analysis\n")
        normal_results = diagnostic_report['normal_mode_analysis']['results']
        f.write(f"- Total Signals Checked: {normal_results['summary']['total_signals_checked']}\n")
        f.write(f"- Successful Entries: {normal_results['summary']['successful_entries']}\n\n")
        
        f.write("### Blocking Analysis\n")
        for module, count in normal_results['blocking_analysis'].items():
            if count > 0:
                pct = normal_results['blocking_percentages'].get(f"{module}_pct", 0)
                f.write(f"- {module}: {count} blocks ({pct:.1f}%)\n")
        f.write("\n")
        
        f.write("## Test Mode Analysis\n")
        test_results = diagnostic_report['test_mode_analysis']['results']
        f.write(f"- Total Signals Checked: {test_results['summary']['total_signals_checked']}\n")
        f.write(f"- Successful Entries: {test_results['summary']['successful_entries']}\n\n")
        
        f.write("### Blocking Analysis\n")
        for module, count in test_results['blocking_analysis'].items():
            if count > 0:
                pct = test_results['blocking_percentages'].get(f"{module}_pct", 0)
                f.write(f"- {module}: {count} blocks ({pct:.1f}%)\n")
        f.write("\n")
        
        f.write("## Module Blocking Ranking (Normal Mode)\n")
        for item in diagnostic_report['module_blocking_ranking']:
            f.write(f"{item['rank']}. **{item['module']}**: {item['percentage_blocked']:.1f}% blocking\n")
        f.write("\n")
        
        f.write("## Key Recommendations\n")
        for rec in diagnostic_report['recommendations']:
            f.write(f"- {rec}\n")
    
    # Save diagnostic log as JSON
    with open(os.path.join(results_dir, "diagnostic_log.json"), 'w') as f:
        # Create a simplified version without complex objects
        json_report = {
            "stage": diagnostic_report["stage"],
            "execution_time": diagnostic_report["execution_time"],
            "data_summary": diagnostic_report["data_summary"],
            "module_blocking_ranking": diagnostic_report["module_blocking_ranking"],
            "recommendations": diagnostic_report["recommendations"]
        }
        json.dump(json_report, f, indent=2, default=str)
    
    print(f"\nDiagnostic results saved to: {results_dir}")
    return results_dir


if __name__ == "__main__":
    print("Starting STAGE 2 - STEP 1: Deep Diagnostic Mapping")
    
    # Run the diagnostic backtest
    diagnostic_report, normal_report, test_report = run_diagnostic_backtest()
    
    # Save results
    results_dir = save_diagnostic_results(diagnostic_report)
    
    # Print console summary
    print(f"\nCONSOLE SUMMARY:")
    print(f"Total signals checked (Normal): {normal_report['summary']['total_signals_checked']}")
    print(f"Successful entries (Normal): {normal_report['summary']['successful_entries']}")
    print(f"Total signals checked (Test): {test_report['summary']['total_signals_checked']}")
    print(f"Successful entries (Test): {test_report['summary']['successful_entries']}")
    
    print(f"\nMODULE BLOCKING RANKING:")
    for item in diagnostic_report['module_blocking_ranking']:
        print(f"  {item['rank']}. {item['module']}: {item['percentage_blocked']:.1f}%")
    
    print(f"\nSTEP 1 COMPLETE: Deep Diagnostic Mapping finished")
    print(f"Results directory: {results_dir}")