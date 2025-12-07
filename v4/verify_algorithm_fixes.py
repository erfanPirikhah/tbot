
import sys
import os
import pandas as pd
import numpy as np
import time

# Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.enhanced_rsi_strategy_v5 import EnhancedRsiStrategyV5, PositionType
from vectorized_backtest import VectorizedBacktest

def verify_look_ahead_bias_fix():
    """Verify that adaptive thresholds do NOT use current candle data"""
    print("\n--- Verifying Look-ahead Bias Fix ---")
    strategy = EnhancedRsiStrategyV5()
    
    # Create data
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    data = pd.DataFrame({
        'close': [100.0] * 50,
        'high': [102.0] * 50,
        'low': [98.0] * 50,
        'open': [100.0] * 50,
        'RSI': [50.0] * 50
    }, index=dates)
    
    # Calculate thresholds for the last candle
    # Case 1: Standard candle
    t1_lower, t1_upper = strategy._calculate_adaptive_rsi_thresholds(data)
    
    # Case 2: Modify the CURRENT (last) candle drastically
    # If using iloc[-1] (current), thresholds would change.
    # If using shift(1) (historical), thresholds should NOT change.
    data.iloc[-1, data.columns.get_loc('high')] = 200.0  # Huge spike
    data.iloc[-1, data.columns.get_loc('low')] = 50.0    # Huge drop
    data.iloc[-1, data.columns.get_loc('close')] = 150.0
    
    t2_lower, t2_upper = strategy._calculate_adaptive_rsi_thresholds(data)
    
    print(f"Original Thresholds: {t1_lower:.4f} / {t1_upper:.4f}")
    print(f"Modified Last Candle: {t2_lower:.4f} / {t2_upper:.4f}")
    
    if abs(t1_lower - t2_lower) < 0.001 and abs(t1_upper - t2_upper) < 0.001:
        print("[PASS]: Thresholds did NOT change when current candle data changed. Look-ahead bias fixed.")
        return True
    else:
        print("[FAIL]: Thresholds changed based on current candle data! Look-ahead bias NOT fixed.")
        return False

def verify_vectorized_backtest():
    """Verify Vectorized Backtest runs and checks performance"""
    print("\n--- Verifying Vectorized Backtest ---")
    
    # Generate large dataset
    N = 5000
    dates = pd.date_range(start='2010-01-01', periods=N, freq='H')
    prices = 100 + np.cumsum(np.random.randn(N))
    data = pd.DataFrame({
        'close': prices,
        'high': prices + 1,
        'low': prices - 1,
        'open': prices,
        'volume': 1000
    }, index=dates)
    
    params = {
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'sl_atr_multiplier': 2.0,
        'risk_reward': 2.0
    }
    
    start_time = time.time()
    vb = VectorizedBacktest(data, params)
    results = vb.run()
    end_time = time.time()
    
    duration = end_time - start_time
    
    print(f"Backtest processed {N} candles in {duration:.4f} seconds.")
    print(f"Results: {results}")
    
    if duration < 1.0: # Should be extremely fast, usually < 0.1s for 5000 rows
        print("[PASS]: Vectorized Backtest is fast (< 1.0s).")
        return True
    else:
        print(f"[FAIL]: Vectorized Backtest too slow ({duration}s).")
        return False

if __name__ == "__main__":
    r1 = verify_look_ahead_bias_fix()
    r2 = verify_vectorized_backtest()
    
    if r1 and r2:
        print("\n[SUCCESS] ALL ALGORITHM FIXES VERIFIED!")
        sys.exit(0)
    else:
        print("\n[FAIL] VERIFICATION FAILED.")
        sys.exit(1)
