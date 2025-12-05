#!/usr/bin/env python3
"""
Debug backtest to identify why no trades are occurring
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add project path
project_path = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_path)

from strategies.enhanced_rsi_strategy_v5 import EnhancedRsiStrategyV5, PositionType
from backtest.enhanced_rsi_backtest_v5 import EnhancedRSIBacktestV5
from config.parameters import OPTIMIZED_PARAMS_V5, TEST_MODE_CONFIG

# Set up logging to see detailed output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def debug_entry_conditions():
    """Debug the entry conditions to understand why no trades are occurring"""
    print("Debugging entry conditions...")
    
    # Create a backtester to get example data
    backtester = EnhancedRSIBacktestV5()
    
    try:
        # Get real data like the backtest does
        data = backtester.fetch_real_data_from_mt5(symbol="EURUSD", timeframe="H1", days_back=30)
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        print(f"Price range: {data['close'].min():.5f} to {data['close'].max():.5f}")
        
        # Check if RSI is properly calculated
        if 'RSI' in data.columns:
            print(f"RSI range: {data['RSI'].min():.2f} to {data['RSI'].max():.2f}")
            print(f"RSI < 30 (oversold): {len(data[data['RSI'] < 30])} instances")
            print(f"RSI > 70 (overbought): {len(data[data['RSI'] > 70])} instances")
        else:
            print("ERROR: RSI column not calculated!")
            
        # Test with TestMode parameters that are more permissive
        test_params = OPTIMIZED_PARAMS_V5.copy()
        test_params.update({
            'test_mode_enabled': True,
            'bypass_contradiction_detection': True,
            'relax_risk_filters': True,
            'relax_entry_conditions': True,
            'enable_all_signals': True,
            'max_trades_per_100': 100,
            'min_candles_between': 2,
            'rsi_entry_buffer': 3,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'enable_short_trades': True
        })
        
        strategy_testmode = EnhancedRsiStrategyV5(**test_params)
        print("\nTesting with TestMode parameters:")
        
        # Test entry conditions on various data points
        entries_testmode = 0
        
        for i in range(50, min(100, len(data))):  # Test first 50 data points after index 50
            current_data = data.iloc[:i+1].copy()
            
            # Check LONG conditions
            long_ok, long_conditions = strategy_testmode.check_entry_conditions(current_data, PositionType.LONG)
            
            if long_ok:
                current_rsi = current_data['RSI'].iloc[-1] if 'RSI' in current_data.columns else 50.0
                print(f"LONG signal at index {i}, RSI: {current_rsi:.2f}")
                print(f"  Conditions: {long_conditions}")
                entries_testmode += 1
                
            # Check SHORT conditions (only if short trades enabled)
            if strategy_testmode.enable_short_trades:
                short_ok, short_conditions = strategy_testmode.check_entry_conditions(current_data, PositionType.SHORT)
                
                if short_ok:
                    current_rsi = current_data['RSI'].iloc[-1] if 'RSI' in current_data.columns else 50.0
                    print(f"SHORT signal at index {i}, RSI: {current_rsi:.2f}")
                    print(f"  Conditions: {short_conditions}")
                    entries_testmode += 1
            
            # Stop after finding some signals for analysis
            if entries_testmode >= 5:
                break
        
        print(f"\nTestMode entries possible: {entries_testmode}")
        
        # Test with normal parameters (more restrictive)
        strategy_normal = EnhancedRsiStrategyV5(**OPTIMIZED_PARAMS_V5)
        print("\nTesting with normal parameters (more restrictive):")
        
        entries_normal = 0
        
        for i in range(50, min(100, len(data))):
            current_data = data.iloc[:i+1].copy()
            
            long_ok, long_conditions = strategy_normal.check_entry_conditions(current_data, PositionType.LONG)
            
            if long_ok:
                current_rsi = current_data['RSI'].iloc[-1] if 'RSI' in current_data.columns else 50.0
                print(f"LONG signal at index {i}, RSI: {current_rsi:.2f}")
                print(f"  Conditions: {long_conditions}")
                entries_normal += 1
                
            if strategy_normal.enable_short_trades:
                short_ok, short_conditions = strategy_normal.check_entry_conditions(current_data, PositionType.SHORT)
                
                if short_ok:
                    current_rsi = current_data['RSI'].iloc[-1] if 'RSI' in current_data.columns else 50.0
                    print(f"SHORT signal at index {i}, RSI: {current_rsi:.2f}")
                    print(f"  Conditions: {short_conditions}")
                    entries_normal += 1
            
            if entries_normal >= 5:
                break
        
        print(f"Normal entries possible: {entries_normal}")
        
        # Now test the actual backtest process
        print("\nTesting actual backtest with TestMode:")
        
        # Reinitialize with TestMode
        backtester_test = EnhancedRSIBacktestV5(initial_capital=10000.0)
        
        test_params_backtest = OPTIMIZED_PARAMS_V5.copy()
        test_params_backtest.update({
            'test_mode_enabled': True,
            'bypass_contradiction_detection': True,
            'relax_risk_filters': True,
            'relax_entry_conditions': True,
            'enable_all_signals': True,
            'enable_trend_filter': False,  # Temporarily disable to isolate
            'enable_mtf': False,  # Temporarily disable to isolate
            'max_trades_per_100': 50,
        })
        
        # Run quick backtest
        results = backtester_test.run_backtest(
            symbol="EURUSD",
            timeframe="H1",
            days_back=10,  # Quick test
            strategy_params=test_params_backtest
        )
        
        metrics = results.get('performance_metrics', {})
        print(f"\nBacktest Results with TestMode:")
        print(f"  Total Trades: {metrics.get('total_trades', 0)}")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.2f}%")
        print(f"  Total P&L: ${metrics.get('total_pnl', 0):,.2f}")
        
        # Now test with only one filter disabled at a time
        print("\n" + "="*60)
        print("Testing with filters disabled individually:")
        
        # Test without trend filter
        params_no_trend = OPTIMIZED_PARAMS_V5.copy()
        params_no_trend['enable_trend_filter'] = False
        params_no_trend['test_mode_enabled'] = True
        params_no_trend['bypass_contradiction_detection'] = True
        
        print("\nTesting without trend filter:")
        backtester_notrend = EnhancedRSIBacktestV5(initial_capital=10000.0)
        results_notrend = backtester_notrend.run_backtest(
            symbol="EURUSD",
            timeframe="H1", 
            days_back=10,
            strategy_params=params_no_trend
        )
        metrics_notrend = results_notrend.get('performance_metrics', {})
        print(f"  Total Trades: {metrics_notrend.get('total_trades', 0)}")
        
        # Test without MTF
        params_no_mtf = OPTIMIZED_PARAMS_V5.copy()
        params_no_mtf['enable_mtf'] = False
        params_no_mtf['test_mode_enabled'] = True
        params_no_mtf['bypass_contradiction_detection'] = True
        
        print("\nTesting without MTF filter:")
        backtester_no_mtf = EnhancedRSIBacktestV5(initial_capital=10000.0)
        results_no_mtf = backtester_no_mtf.run_backtest(
            symbol="EURUSD",
            timeframe="H1",
            days_back=10,
            strategy_params=params_no_mtf
        )
        metrics_no_mtf = results_no_mtf.get('performance_metrics', {})
        print(f"  Total Trades: {metrics_no_mtf.get('total_trades', 0)}")
        
        # Test without contradiction detection
        params_no_contra = OPTIMIZED_PARAMS_V5.copy()
        params_no_contra['bypass_contradiction_detection'] = True
        params_no_contra['test_mode_enabled'] = True
        
        print("\nTesting without contradiction detection:")
        backtester_no_contra = EnhancedRSIBacktestV5(initial_capital=10000.0)
        results_no_contra = backtester_no_contra.run_backtest(
            symbol="EURUSD",
            timeframe="H1",
            days_back=10, 
            strategy_params=params_no_contra
        )
        metrics_no_contra = results_no_contra.get('performance_metrics', {})
        print(f"  Total Trades: {metrics_no_contra.get('total_trades', 0)}")
        
        print("\n" + "="*60)
        print("DEBUG COMPLETE")
        print("If trades occurred in any of these tests, that filter was blocking trades")
        
    except Exception as e:
        print(f"Error in debug: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_entry_conditions()