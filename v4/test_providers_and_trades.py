#!/usr/bin/env python3
"""
Test script to verify data providers and trade generation
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

from providers.provider_registry import DataProviderRegistry
from strategies.enhanced_rsi_strategy_v5 import EnhancedRsiStrategyV5, PositionType
from config.parameters import OPTIMIZED_PARAMS_V5

# Set up logging to see detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_providers_and_trades():
    """Test if providers work and trades can be generated"""
    print("Testing providers and trade generation...")
    
    # Test provider registry
    registry = DataProviderRegistry(test_mode=False)
    
    print("Available providers:")
    connections = registry.test_all_connections()
    for provider, status in connections.items():
        print(f"  {provider}: {'✅' if status else '❌'}")
    
    print("\nFetching data using provider registry...")
    result = registry.get_data("EURUSD", "H1", 200)
    
    if result['success']:
        print(f"✅ Data fetched successfully from: {result['source']}")
        print(f"   Data shape: {result['data'].shape}")
        print(f"   Date range: {result['data'].index[0]} to {result['data'].index[-1]}")
        print(f"   Price range: {result['data']['close'].min():.5f} to {result['data']['close'].max():.5f}")
        
        data = result['data']
        
        # Calculate RSI to make sure it's available for strategy
        from backtest.enhanced_rsi_backtest_v5 import EnhancedRSIBacktestV5
        backtester = EnhancedRSIBacktestV5()
        data = backtester._calculate_rsi(data)
        
        print(f"   RSI range: {data['RSI'].min():.2f} to {data['RSI'].max():.2f}")
        print(f"   RSI < 30 (oversold): {len(data[data['RSI'] < 30])} instances")
        print(f"   RSI > 70 (overbought): {len(data[data['RSI'] > 70])} instances")
        
        # Test with strategy parameters that should allow trades
        print("\nTesting with permissive parameters:")
        test_params = OPTIMIZED_PARAMS_V5.copy()
        test_params.update({
            'test_mode_enabled': True,
            'bypass_contradiction_detection': True,
            'enable_trend_filter': False,  # Disable to allow more trades
            'enable_mtf': False,  # Disable to allow more trades
            'rsi_oversold': 35,  # More permissive
            'rsi_overbought': 65,  # More permissive
            'relax_entry_conditions': True,
            'max_trades_per_100': 50  # Higher limit
        })
        
        strategy = EnhancedRsiStrategyV5(**test_params)
        
        # Test entry conditions on a sample of the data
        entries_found = 0
        for i in range(50, len(data)):
            current_data = data.iloc[:i+1].copy()
            
            # Check if RSI is available
            if 'RSI' not in current_data.columns:
                current_data = backtester._calculate_rsi(current_data)
            
            # Check LONG and SHORT conditions
            long_ok, long_conditions = strategy.check_entry_conditions(current_data, PositionType.LONG)
            short_ok, short_conditions = strategy.check_entry_conditions(current_data, PositionType.SHORT)
            
            if long_ok:
                print(f"   LONG signal at {current_data.index[-1]}, RSI: {current_data['RSI'].iloc[-1]:.2f}")
                print(f"     Conditions: {long_conditions}")
                entries_found += 1
            
            if short_ok:
                print(f"   SHORT signal at {current_data.index[-1]}, RSI: {current_data['RSI'].iloc[-1]:.2f}")
                print(f"     Conditions: {short_conditions}")
                entries_found += 1
            
            if entries_found >= 10:
                break
        
        print(f"\nTotal potential entries found: {entries_found}")
        
        if entries_found == 0:
            print("\nStill no entries found. Let's try with even more permissive settings...")
            # Try with most permissive settings possible
            ultra_permissive_params = OPTIMIZED_PARAMS_V5.copy()
            ultra_permissive_params.update({
                'test_mode_enabled': True,
                'bypass_contradiction_detection': True,
                'relax_risk_filters': True,
                'relax_entry_conditions': True,
                'enable_all_signals': True,
                'enable_trend_filter': False,
                'enable_mtf': False,
                'enable_volatility_filter': False,
                'min_candles_between': 1,  # Minimal spacing
                'rsi_entry_buffer': 1,  # Minimal buffer
                'rsi_oversold': 25,  # Very permissive
                'rsi_overbought': 75,  # Very permissive
                'max_trades_per_100': 100,  # Allow maximum trades
                'max_consecutive_losses': 10,  # Higher tolerance
            })
            
            strategy_ultra = EnhancedRsiStrategyV5(**ultra_permissive_params)
            ultra_entries = 0
            for i in range(50, len(data)):
                current_data = data.iloc[:i+1].copy()
                
                if 'RSI' not in current_data.columns:
                    current_data = backtester._calculate_rsi(current_data)
                
                long_ok, _ = strategy_ultra.check_entry_conditions(current_data, PositionType.LONG)
                short_ok, _ = strategy_ultra.check_entry_conditions(current_data, PositionType.SHORT)
                
                if long_ok or short_ok:
                    ultra_entries += 1
                    if ultra_entries >= 5:
                        break
            
            print(f"Ultra permissive entries found: {ultra_entries}")
        
        # Now run a proper backtest with TestMode
        print("\nRunning actual backtest with TestMode...")
        from backtest.enhanced_rsi_backtest_v5 import EnhancedRSIBacktestV5
        
        backtester = EnhancedRSIBacktestV5(
            initial_capital=10000.0,
            commission=0.0003,
            slippage=0.0001,
            enable_plotting=False,
            detailed_logging=True,
            save_trade_logs=True,
            output_dir=os.path.join("logs", "backtests"),
            data_fetcher=None  # Will use the internal fetcher
        )
        
        # Create a custom data_fetcher that uses the provider registry
        from data.data_fetcher import DataFetcher
        custom_data_fetcher = DataFetcher(test_mode=True)
        backtester.data_fetcher = custom_data_fetcher
        
        # Use strategy with TestMode parameters
        test_strategy_params = OPTIMIZED_PARAMS_V5.copy()
        test_strategy_params.update({
            'test_mode_enabled': True,
            'bypass_contradiction_detection': True,
            'relax_risk_filters': True,
            'relax_entry_conditions': True,
            'enable_all_signals': True,
            'enable_trend_filter': False,  # Disable for testing
            'enable_mtf': False,  # Disable for testing
            'max_trades_per_100': 50,
        })
        
        # Run backtest with a small time range to see if trades happen
        results = backtester.run_backtest(
            symbol="EURUSD",
            timeframe="H1",
            days_back=5,  # Only 5 days for quick test
            strategy_params=test_strategy_params
        )
        
        metrics = results.get('performance_metrics', {})
        print(f"\nBacktest Results:")
        print(f"  Total Trades: {metrics.get('total_trades', 0)}")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.2f}%")
        print(f"  Total P&L: ${metrics.get('total_pnl', 0):,.2f}")
        print(f"  Current Portfolio Value: ${metrics.get('current_portfolio_value', 0):,.2f}")
        
        if metrics.get('total_trades', 0) > 0:
            print("\n✅ SUCCESS: Trades are now being generated!")
            print("The issue was likely the overly restrictive filters in the default configuration.")
        else:
            print("\n❌ FAILURE: Still no trades generated even with TestMode parameters")
            
    else:
        print("❌ Failed to fetch data from any provider")
        print("Errors:")
        for provider, error in result.get('errors', {}).items():
            print(f"  {provider}: {error}")

if __name__ == "__main__":
    test_providers_and_trades()