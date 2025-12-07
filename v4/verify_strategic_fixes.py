
import sys
import os
import pandas as pd
import numpy as np

# Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.contradiction_detector import SignalContradictionDetector
from strategies.trend_filter import AdvancedTrendFilter

def verify_divergence_strength_reduction():
    """Verify that RSI Divergence contradiction strength is reduced (halved)"""
    print("\n--- Verifying RSI Divergence Strength Reduction ---")
    detector = SignalContradictionDetector()
    
    # Create data with divergence
    # Price going UP, RSI going DOWN (Bearish Divergence) -> Contradiction for LONG
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    prices = np.linspace(100, 110, 20) # Up
    rsi = np.linspace(70, 50, 20)      # Down
    
    data = pd.DataFrame({'close': prices, 'RSI': rsi}, index=dates)
    
    # Detect
    results = detector.detect_all_contradictions(data, 'LONG')
    
    # Check details
    div_details = results['details'].get('rsi_price_divergence', {})
    
    if not div_details.get('exists'):
        print("[FAIL]: Failed to detect divergence in test data.")
        return False
        
    strength = div_details.get('strength')
    desc = div_details.get('description')
    
    print(f"Divergence detected: {desc}")
    print(f"Strength reported: {strength}")
    
    # Theoretical max strength is 1.0. With 0.5 reduction, should be <= 0.5
    # The raw calculation in detect_rsi_price_divergence is min(1.0, abs(...) * 50)
    # Then multiplied by 0.5 in detect_all_contradictions
    
    if strength <= 0.6: # Give some buffer, but expects ~0.5 max usually
        print("[PASS]: Divergence strength is reduced/capped (<= 0.6).")
        return True
    else:
        print(f"[FAIL]: Divergence strength {strength} is too high (expected <= 0.6 after reduction).")
        return False

def verify_trend_lag_reduction():
    """Verify Trend Filter favors recent price action"""
    print("\n--- Verifying Trend Filter Lag Reduction ---")
    tf = AdvancedTrendFilter()
    
    # Scenario: Long term Down, Short term Sharp Up rebound
    # If lag is reduced, this should show Bullish trend or higher score than with old weights
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    
    # 25 days down
    prices = list(np.linspace(100, 80, 25))
    # 5 days sharp up
    prices.extend(list(np.linspace(80, 95, 5)))
    
    data = pd.DataFrame({'close': prices}, index=dates)
    
    # Mock EMAs to be slow/bearish
    # We really just want to test calculate_price_trend_score logic which we changed
    score, desc = tf.calculate_price_trend_score(data)
    
    print(f"Price Trend Score: {score}")
    print(f"Description: {desc}")
    
    # Short trend: (95-80)/80 = +18.75%
    # Medium trend: (95-89)/89 = +6.7% (approx)
    # Weighted: 0.8 * 0.18 + 0.2 * 0.06 > 0.6 * 0.18 + 0.4 * 0.06
    
    # If the description says "Bullish", it logic worked (since short term is bullish)
    if "Bullish" in desc:
         print("[PASS]: Trend identified as Bullish due to recent price action.")
         return True
    else:
         print(f"[FAIL]: Trend failed to identify recent bullish move. Desc: {desc}")
         return False

if __name__ == "__main__":
    r1 = verify_divergence_strength_reduction()
    r2 = verify_trend_lag_reduction()
    
    if r1 and r2:
        print("\n[SUCCESS] ALL STRATEGIC FIXES VERIFIED!")
        sys.exit(0)
    else:
        print("\n[FAIL] VERIFICATION FAILED.")
        sys.exit(1)
