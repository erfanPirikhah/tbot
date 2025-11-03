# tests/quick_backtest.py

import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advanced_backtest import AdvancedBacktestEngine

def quick_test():
    """تست سریع روی یک نماد"""
    engine = AdvancedBacktestEngine()
    
    print("🔍 Running Quick Backtest...")
    
    result = engine.run_comprehensive_backtest(
        symbol="XAUUSD",
        interval="H1",
        data_source="MT5",
        start_date=datetime.now() - timedelta(days=90),
        end_date=datetime.now(),
        initial_balance=10000.0
    )
    
    # نمایش نتایج پایه
    basic = result['basic_results']
    print(f"\n📊 QUICK TEST RESULTS for XAUUSD:")
    print(f"Total Return: {basic['Return [%]']:.2f}%")
    print(f"Sharpe Ratio: {basic['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {basic['Max. Drawdown [%]']:.2f}%")
    print(f"Win Rate: {basic['Win Rate [%]']:.2f}%")
    print(f"Total Trades: {basic['# Trades']}")
    
    return result

if __name__ == "__main__":
    quick_test()