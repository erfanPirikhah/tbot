#!/usr/bin/env python3
"""
Focused test to check backtest simulation process step-by-step
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project path
project_path = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_path)

from providers.data_provider import ImprovedSimulatedProvider
from strategies.enhanced_rsi_strategy_v5 import EnhancedRsiStrategyV5


def run_focused_backtest():
    """Run a focused backtest step-by-step to see where trades are not happening"""
    print("Running focused backtest...")
    
    # Generate data with the provider
    provider = ImprovedSimulatedProvider(seed=123)
    data = provider.fetch_data("BTCUSDT", "H1", 100)
    print(f"Generated {len(data)} candles")
    
    # Create strategy with permissive parameters
    strategy = EnhancedRsiStrategyV5(
        test_mode_enabled=True,
        bypass_contradiction_detection=True,
        rsi_oversold=40,  # Very permissive
        rsi_overbought=60,  # Very permissive  
        rsi_entry_buffer=15,  # Very loose
        enable_trend_filter=False,
        enable_mtf=False,
        min_candles_between=1,
        max_trades_per_100=200,
        risk_per_trade=0.02,
        enable_short_trades=True,
        enable_trailing_stop=False
    )
    
    print(f"Strategy initialized with test mode: {strategy.test_mode_enabled}")
    print(f"Bypass contradictions: {strategy.bypass_contradiction_detection}")
    
    # Run simulation manually like the backtester does
    initial_capital = 10000.0
    strategy._portfolio_value = initial_capital
    
    trades = []
    portfolio_values = []
    
    print(f"\nStarting simulation with {len(data)} data points...")
    
    for i in range(50, len(data)):  # Start after enough data for RSI
        current_data = data.iloc[:i+1].copy()
        
        # Generate signal
        signal = strategy.generate_signal(current_data, i)
        
        # Record portfolio value
        portfolio_values.append({
            'timestamp': current_data.index[-1],
            'portfolio_value': strategy._portfolio_value,
            'price': current_data['close'].iloc[-1]
        })
        
        # Record trade signals
        if signal['action'] != 'HOLD':
            trades.append({
                'timestamp': current_data.index[-1],
                'action': signal['action'],
                'price': signal.get('price', current_data['close'].iloc[-1]),
                'reason': signal.get('reason', ''),
                'pnl_amount': signal.get('pnl_amount', 0),
                'position': signal.get('position', 'N/A')
            })
            print(f"Trade signal at {i}: {signal['action']} at {signal.get('price', current_data['close'].iloc[-1]):.2f}, reason: {signal.get('reason', '')[:50]}")
        
        # Show progress every 20 iterations
        if i % 20 == 0:
            print(f"Processed {i}/{len(data)} candles, trades so far: {len(trades)}")
    
    print(f"\nFinal Results:")
    print(f"Total trades generated: {len(trades)}")
    print(f"Final portfolio value: ${strategy._portfolio_value:.2f}")
    
    # Check strategy metrics
    metrics = strategy.get_performance_metrics()
    print(f"Strategy metrics - Total trades: {metrics.get('total_trades', 0)}")
    print(f"Win rate: {metrics.get('win_rate', 0)}%")
    
    if len(trades) > 0:
        print("SUCCESS: Trades were generated in focused test!")
        return True, metrics, trades
    else:
        print("FAILED: No trades generated in focused test")
        return False, metrics, trades


if __name__ == "__main__":
    print("Running focused backtest analysis...")
    success, metrics, trades = run_focused_backtest()
    
    if trades:
        print(f"\nTrade details:")
        for i, trade in enumerate(trades):
            print(f"  {i+1}. {trade['action']} at {trade['price']:.2f} - {trade['reason'][:50]}")