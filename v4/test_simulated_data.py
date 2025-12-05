#!/usr/bin/env python3
"""
Test to check what the simulated provider is generating
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project path
project_path = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_path)

from providers.data_provider import ImprovedSimulatedProvider


def test_simulated_data():
    """Test what data the simulated provider generates"""
    print("Testing simulated data generation...")
    
    provider = ImprovedSimulatedProvider(seed=42)
    
    # Generate data
    data = provider.fetch_data("BTCUSDT", "H1", 50)
    print(f"Generated {len(data)} candles")
    print(f"Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
    print(f"Recent prices: {data['close'].tail(10).tolist()}")
    
    # Calculate RSI manually to see values
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    print(f"\nRSI values (last 20): {rsi.tail(20).tolist()}")
    print(f"RSI range: {rsi.min():.2f} - {rsi.max():.2f}")
    print(f"RSI below 30 (oversold): {(rsi < 30).sum()} instances")
    print(f"RSI above 70 (overbought): {(rsi > 70).sum()} instances")
    print(f"RSI between 30-70: {((rsi >= 30) & (rsi <= 70)).sum()} instances")
    
    # Test with the specific permissive parameters
    print(f"\nWith permissive parameters (oversold=45, overbought=55):")
    print(f"RSI below 45: {(rsi < 45).sum()} instances")  
    print(f"RSI above 55: {(rsi > 55).sum()} instances")
    
    # Check for conditions meeting very loose thresholds
    print(f"RSI below 40: {(rsi < 40).sum()} instances")
    print(f"RSI above 60: {(rsi > 60).sum()} instances")
    
    return data


def test_strategy_on_simulated_data():
    """Test if the strategy can process the simulated data"""
    print("\nTesting strategy on simulated data...")
    
    from providers.data_provider import ImprovedSimulatedProvider
    from strategies.enhanced_rsi_strategy_v5 import EnhancedRsiStrategyV5, PositionType
    
    # Generate data with the provider
    provider = ImprovedSimulatedProvider(seed=123)
    data = provider.fetch_data("BTCUSDT", "H1", 100)
    
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
        max_trades_per_100=200
    )
    
    # Test entry conditions on multiple points in the data
    print(f"Testing entry conditions on {len(data)} data points (last 30):")
    
    for i in range(max(50, len(data)-30), len(data)):  # Test on last 30 points (after RSI can be calculated)
        current_data = data.iloc[:i+1].copy()
        
        # Check RSI
        if len(current_data) >= 15:  # Need enough data for RSI
            delta = current_data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            current_rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Test LONG conditions
            long_ok, long_conditions = strategy.check_entry_conditions(current_data, PositionType.LONG)
            short_ok, short_conditions = strategy.check_entry_conditions(current_data, PositionType.SHORT)
            
            print(f"  Index {i}: RSI={current_rsi:.2f}, LONG={long_ok}, SHORT={short_ok}")
            if long_ok or short_ok:
                print(f"    Conditions: {long_conditions if long_ok else short_conditions}")
                break
    else:
        print("  No entry conditions were met")


if __name__ == "__main__":
    print("Testing simulated provider data...")
    data = test_simulated_data()
    test_strategy_on_simulated_data()