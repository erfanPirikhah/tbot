# test_strategy_signals.py
import sys
import os
sys.path.append(os.path.dirname(__file__))

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from strategies.improved_advanced_rsi_strategy import ImprovedAdvancedRsiStrategy
from data.data_fetcher import fetch_market_data

def create_test_data():
    """ایجاد داده تست با شرایط مختلف"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='1H')
    
    # داده با RSI پایین (شرایط خرید)
    prices_low_rsi = [1000]
    for i in range(1, len(dates)):
        # ایجاد روند نزولی برای RSI پایین
        change = np.random.normal(-0.5, 0.3)
        new_price = prices_low_rsi[-1] * (1 + change/100)
        prices_low_rsi.append(max(new_price, 800))
    
    # داده با RSI بالا (شرایط فروش)
    prices_high_rsi = [1000]
    for i in range(1, len(dates)):
        # ایجاد روند صعودی برای RSI بالا
        change = np.random.normal(0.5, 0.3)
        new_price = prices_high_rsi[-1] * (1 + change/100)
        prices_high_rsi.append(min(new_price, 1200))
    
    df_low = pd.DataFrame({
        'open_time': dates,
        'open': prices_low_rsi,
        'high': [p * 1.002 for p in prices_low_rsi],
        'low': [p * 0.998 for p in prices_low_rsi],
        'close': prices_low_rsi,
        'volume': np.random.normal(1000, 200, len(dates))
    })
    
    df_high = pd.DataFrame({
        'open_time': dates,
        'open': prices_high_rsi,
        'high': [p * 1.002 for p in prices_high_rsi],
        'low': [p * 0.998 for p in prices_high_rsi],
        'close': prices_high_rsi,
        'volume': np.random.normal(1000, 200, len(dates))
    })
    
    return df_low, df_high

def test_strategy_with_real_data():
    """تست استراتژی با داده‌های واقعی"""
    print("🧪 تست استراتژی با داده‌های واقعی")
    print("=" * 50)
    
    symbols = ["XAUUSD", "XAGUSD", "EURUSD"]
    
    for symbol in symbols:
        print(f"\n📊 تست نماد: {symbol}")
        try:
            # دریافت داده واقعی
            data = fetch_market_data(symbol, "1h", 100, "MT5")
            
            if data.empty:
                print(f"   ❌ داده‌ای برای {symbol} دریافت نشد")
                continue
            
            # محاسبه RSI
            from indicators.rsi import calculate_rsi
            data_with_rsi = calculate_rsi(data)
            
            current_rsi = data_with_rsi['RSI'].iloc[-1]
            current_price = data_with_rsi['close'].iloc[-1]
            
            print(f"   💰 قیمت فعلی: {current_price:.2f}")
            print(f"   📊 RSI فعلی: {current_rsi:.2f}")
            
            # تست استراتژی
            strategy = ImprovedAdvancedRsiStrategy(
                overbought=65,      # کاهش سطح overbought
                oversold=35,        # افزایش سطح oversold
                min_conditions=2,   # کاهش حداقل شرایط
                risk_per_trade=0.02
            )
            
            signal = strategy.generate_signal(data_with_rsi)
            
            print(f"   🎯 سیگنال: {signal['action']}")
            print(f"   💪 قدرت: {signal.get('signal_strength', 'N/A')}")
            print(f"   📈 امتیاز خرید: {signal.get('buy_score', 'N/A')}")
            print(f"   📝 دلیل: {signal.get('reason', 'N/A')[:100]}...")
            
        except Exception as e:
            print(f"   ❌ خطا در تست {symbol}: {e}")

def test_strategy_with_simulated_data():
    """تست استراتژی با داده‌های شبیه‌سازی شده"""
    print("\n🎲 تست استراتژی با داده‌های شبیه‌سازی شده")
    print("=" * 50)
    
    # ایجاد داده تست
    df_low_rsi, df_high_rsi = create_test_data()
    
    # محاسبه RSI برای داده‌ها
    from indicators.rsi import calculate_rsi
    df_low_rsi = calculate_rsi(df_low_rsi)
    df_high_rsi = calculate_rsi(df_high_rsi)
    
    print(f"📊 داده RSI پایین: {df_low_rsi['RSI'].iloc[-1]:.2f}")
    print(f"📊 داده RSI بالا: {df_high_rsi['RSI'].iloc[-1]:.2f}")
    
    # استراتژی با پارامترهای آسان
    strategy = ImprovedAdvancedRsiStrategy(
        overbought=60,
        oversold=40,
        min_conditions=2,
        risk_per_trade=0.01
    )
    
    # تست با داده RSI پایین
    signal_low = strategy.generate_signal(df_low_rsi)
    print(f"\n🧪 تست با RSI پایین:")
    print(f"   🎯 سیگنال: {signal_low['action']}")
    print(f"   💪 قدرت: {signal_low.get('signal_strength', 'N/A')}")
    print(f"   📈 امتیاز خرید: {signal_low.get('buy_score', 'N/A')}")
    
    # ریست استراتژی برای تست دوم
    strategy.reset_state()
    
    # تست با داده RSI بالا
    signal_high = strategy.generate_signal(df_high_rsi)
    print(f"\n🧪 تست با RSI بالا:")
    print(f"   🎯 سیگنال: {signal_high['action']}")
    print(f"   💪 قدرت: {signal_high.get('signal_strength', 'N/A')}")
    print(f"   📈 امتیاز خرید: {signal_high.get('buy_score', 'N/A')}")

if __name__ == "__main__":
    test_strategy_with_real_data()
    test_strategy_with_simulated_data()