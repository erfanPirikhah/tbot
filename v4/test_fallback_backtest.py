#!/usr/bin/env python3
"""
Test backtest with fallback to ensure trades are generated
"""

import sys
import os
import logging

# Add project path
project_path = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_path)

# Set up logging to see detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from backtest.enhanced_rsi_backtest_v5 import EnhancedRSIBacktestV5
from config.parameters import OPTIMIZED_PARAMS_V5
from data.data_fetcher import DataFetcher

def test_backtest_with_fallback():
    print("Testing backtest with data fetcher fallback...")
    
    # Create a backtest with TestMode to ensure trades happen
    data_fetcher = DataFetcher(test_mode=True)
    
    backtester = EnhancedRSIBacktestV5(
        initial_capital=10000.0,
        enable_plotting=False,
        detailed_logging=True,
        data_fetcher=data_fetcher
    )
    
    # Use parameters that should generate trades
    test_params = OPTIMIZED_PARAMS_V5.copy()
    test_params.update({
        'test_mode_enabled': True,
        'bypass_contradiction_detection': True,
        'relax_risk_filters': True,
        'relax_entry_conditions': True,
        'enable_all_signals': True,
        'enable_trend_filter': False,  # Disable to allow more trades
        'enable_mtf': False,  # Disable to allow more trades
        'max_trades_per_100': 50,
        'min_candles_between': 2,
        'rsi_entry_buffer': 3,
        'rsi_oversold': 35,
        'rsi_overbought': 65,
    })
    
    print('Running backtest with fallback data...')
    results = backtester.run_backtest(
        symbol='EURUSD',
        timeframe='H1',
        days_back=5,
        strategy_params=test_params
    )
    
    metrics = results.get('performance_metrics', {})
    print(f'\nBacktest Results:')
    print(f'Total Trades: {metrics.get("total_trades", 0)}')
    print(f'Win Rate: {metrics.get("win_rate", 0):.2f}%')
    print(f'Total P&L: ${metrics.get("total_pnl", 0):.2f}')
    print(f'Portfolio Value: ${metrics.get("current_portfolio_value", 0):.2f}')
    
    if metrics.get('total_trades', 0) > 0:
        print("\n✅ SUCCESS: Trades were generated!")
        print("The data fallback system is working correctly.")
    else:
        print("\n❌ No trades were generated, checking data...")
        
        # Test data generation manually
        print("\nTesting manual data fetch...")
        try:
            data = backtester.fetch_real_data_from_mt5('EURUSD', 'H1', 5)
            print(f"Manual fetch result: {len(data)} candles, range: {data.index[0]} to {data.index[-1]}")
            if 'RSI' in data.columns:
                oversold = len(data[data['RSI'] < 35])
                overbought = len(data[data['RSI'] > 65])
                print(f"RSI < 35: {oversold} candles, RSI > 65: {overbought} candles")
        except Exception as e:
            print(f"Manual fetch failed: {e}")

if __name__ == "__main__":
    test_backtest_with_fallback()