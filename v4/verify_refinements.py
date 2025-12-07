
import sys
import os
import pandas as pd
import numpy as np

# Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.enhanced_rsi_strategy_v5 import EnhancedRsiStrategyV5, PositionType, Trade, TradeEntry

def verify_obv_filter():
    """Verify OBV Filter blocks signals against the flow"""
    print("\n--- Verifying OBV Filter ---")
    strategy = EnhancedRsiStrategyV5(enable_trend_filter=False, enable_mtf=False)
    # Disable contradiction detection to isolate OBV check or use TestMode to see warnings
    strategy.bypass_contradiction_detection = True 
    
    # Create data
    dates = pd.date_range(start='2023-01-01', periods=50, freq='H')
    
    # Price Trend UP, but Volume flow DOWN (Bearish OBV)
    # Close prices up, but big volume on down days, small volume on up days
    close = []
    volume = []
    curr = 100
    for i in range(50):
        if i % 2 == 0:
            curr += 2 # Up
            vol = 100 # Low volume
        else:
            curr -= 1 # Down
            vol = 1000 # High volume
        close.append(curr)
        volume.append(vol)
        
    data = pd.DataFrame({'close': close, 'volume': volume, 'high': close, 'low': close}, index=dates)
    data['RSI'] = 10 # Very Oversold -> Should pass RSI check, allowing us to test OBV block
    
    # Check Entry Conditions for LONG
    # OBV should be bearish because heavy volume on down moves
    can_enter, msgs = strategy.check_entry_conditions(data, PositionType.LONG)
    
    print(f"Entry Result: {can_enter}")
    print(f"Messages: {msgs}")
    
    obv_msg_present = any("OBV" in m for m in msgs)
    
    if not can_enter and obv_msg_present:
        print("[PASS]: OBV Filter blocked Long signal due to bearish volume flow.")
        return True
    elif can_enter and obv_msg_present:
        # If in TestMode it might pass with warning, but by default it blocks?
        # My implementation: returns False if not TestMode.
        # Check if strategy.test_mode_enabled is False by default... yes.
        print("[FAIL]: OBV Filter warned but did not block (TestMode active?).")
        return False
    else:
        print("[FAIL]: OBV Filter did not trigger.")
        return False

def verify_dynamic_tp():
    """Verify Dynamic TP triggers on RSI extreme"""
    print("\n--- Verifying Dynamic Partial Exit (RSI) ---")
    strategy = EnhancedRsiStrategyV5()
    
    # Setup Trade
    trade = Trade()
    import pandas as pd
    # Fix: use entries list to set price
    trade.entries = [TradeEntry(price=100.0, quantity=2.0, time=pd.Timestamp.now())]
    # trade.entry_price will be 100.0 automatically
    trade.partial_exit_done = False
    trade.highest_profit = 0
    trade.stop_loss = 90.0
    trade.take_profit = 120.0 # Far away
    strategy._current_trade = trade
    strategy._position = PositionType.LONG
    strategy.enable_partial_exit = True
    strategy.partial_exit_threshold = 10.0 # High threshold (10%)
    strategy.test_mode_enabled = False # Ensure normal mode
    
    # Mock Data
    dates = pd.date_range(start='2023-01-01', periods=1, freq='h') # lowercase h
    
    # Scenario: Price = 102 (2% profit, < 10% threshold)
    # But RSI = 80 (> 75 extreme)
    # Should trigger Partial Exit
    data = pd.DataFrame({
        'close': [102.0], 
        'high': [102.0], 
        'low': [101.0], 
        'volume': [1000],
        'RSI': [80.0]
    }, index=dates)
    
    # Mock calculate_atr
    strategy.calculate_atr = lambda x: 1.0
    
    exit_signal = strategy.check_exit_conditions(data, 100)
    
    if exit_signal and exit_signal.get('action') == 'PARTIAL_EXIT':
        reason = exit_signal.get('reason', '')
        print(f"Partial Exit Triggered! Reason: {reason}")
        if "RSI_Extreme" in reason:
            print("[PASS]: Dynamic Partial Exit triggered by RSI Extreme.")
            return True
        else:
            print(f"[FAIL]: Partial Exit triggered but wrong reason: {reason}")
            return False
    else:
        print("[FAIL]: Dynamic Partial Exit NOT triggered.")
        return False

if __name__ == "__main__":
    r1 = verify_obv_filter()
    r2 = verify_dynamic_tp()
    
    if r1 and r2:
        print("\n[SUCCESS] ALL REFINEMENTS VERIFIED!")
        sys.exit(0)
    else:
        print("\n[FAIL] VERIFICATION FAILED.")
        sys.exit(1)
