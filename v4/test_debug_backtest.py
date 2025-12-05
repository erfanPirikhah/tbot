#!/usr/bin/env python3
"""
Debug test to see the complete signal generation process
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
from strategies.enhanced_rsi_strategy_v5 import EnhancedRsiStrategyV5, PositionType


def run_debug_backtest():
    """Run a debug backtest with detailed logging"""
    print("Running debug backtest with detailed output...")
    
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
        enable_trailing_stop=False,
        confirmation_candles=1
    )
    
    print(f"Strategy initialized with test mode: {strategy.test_mode_enabled}")
    print(f"Current position: {strategy._position}")
    
    # Track what happens step by step
    trades = []
    last_signal_action = None
    last_position = strategy._position
    
    print(f"\nStarting detailed simulation...")
    
    for i in range(50, min(70, len(data))):  # Only test first 20 points after 50
        current_data = data.iloc[:i+1].copy()
        
        print(f"\nStep {i}: Position={strategy._position.value}, Portfolio=${strategy._portfolio_value:.2f}")
        print(f"  Current price: {current_data['close'].iloc[-1]:.2f}")
        
        # Calculate and show RSI
        delta = current_data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        print(f"  RSI: {current_rsi:.2f}")
        
        # Check entry conditions manually for both directions
        if strategy._position == PositionType.OUT:
            long_ok, long_conditions = strategy.check_entry_conditions(current_data, PositionType.LONG)
            short_ok, short_conditions = strategy.check_entry_conditions(current_data, PositionType.SHORT)
            print(f"  Entry checks - LONG: {long_ok}, SHORT: {short_ok}")
            if long_ok:
                print(f"    LONG conditions: {long_conditions}")
            if short_ok:
                print(f"    SHORT conditions: {short_conditions}")
        
        # Generate signal (this is what happens during backtest)
        signal = strategy.generate_signal(current_data, i)
        
        print(f"  Signal: {signal['action']}")
        if signal['action'] != 'HOLD':
            print(f"    Price: {signal.get('price', 'N/A')}")
            print(f"    Reason: {signal.get('reason', '')[:80]}...")
        
        # Record any trades
        if signal['action'] in ['BUY', 'SELL', 'EXIT', 'PARTIAL_EXIT']:
            trades.append({
                'step': i,
                'action': signal['action'],
                'price': signal.get('price', current_data['close'].iloc[-1]),
                'reason': signal.get('reason', ''),
            })
        
        # Check if position changed
        if strategy._position != last_position:
            print(f"  >> POSITION CHANGED: {last_position.value} -> {strategy._position.value}")
            last_position = strategy._position
        
        last_signal_action = signal['action']
    
    print(f"\nSummary:")
    print(f"Started with position: OUT, ended with: {strategy._position.value}")
    print(f"Total trades recorded: {len(trades)}")
    
    for i, trade in enumerate(trades):
        print(f"  {i+1}. {trade['step']}: {trade['action']} at {trade['price']:.2f} - {trade['reason'][:50]}")
    
    # Check final strategy metrics
    metrics = strategy.get_performance_metrics()
    print(f"\nFinal metrics: {metrics.get('total_trades', 0)} total trades, {metrics.get('win_rate', 0)}% win rate")
    
    return len(trades) > 0, metrics


if __name__ == "__main__":
    print("Running debug backtest analysis...")
    success, metrics = run_debug_backtest()
    
    if success:
        print("\nSUCCESS: Trades were generated in debug test!")
    else:
        print("\nFAILED: No trades generated in debug test")