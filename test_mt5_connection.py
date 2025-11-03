# debug_price.py
import sys
import os
sys.path.append(os.path.dirname(__file__))

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from data.mt5_data import mt5_fetcher, MT5_AVAILABLE
from data.data_fetcher import get_current_price
from config import MT5_SYMBOL_MAP

def debug_price_issue():
    print("🔍 دیباگ مشکل قیمت لحظه‌ای")
    print("=" * 50)
    
    # تست مستقیم MT5
    print("1. تست مستقیم MT5 Data Fetcher:")
    if MT5_AVAILABLE and mt5_fetcher:
        print(f"   ✅ MT5 Connected: {mt5_fetcher.connected}")
        
        # تست قیمت XAGUSD
        price = mt5_fetcher.get_current_price("XAGUSD")
        print(f"   💰 Direct MT5 price for XAGUSD: {price}")
    else:
        print("   ❌ MT5 not available")
    
    # تست از طریق data_fetcher
    print("\n2. تست از طریق Data Fetcher:")
    try:
        price = get_current_price("XAGUSD", "MT5")
        print(f"   💰 Data Fetcher price for XAGUSD: {price}")
    except Exception as e:
        print(f"   ❌ Error in Data Fetcher: {e}")
    
    # تست نمادها
    print("\n3. بررسی مپ نمادها:")
    symbol_display = "نقره (XAGUSD)"
    symbol_code = MT5_SYMBOL_MAP.get(symbol_display)
    print(f"   📋 Symbol display: '{symbol_display}'")
    print(f"   🔤 Symbol code: '{symbol_code}'")
    print(f"   ✅ In MT5_SYMBOL_MAP: {symbol_display in MT5_SYMBOL_MAP}")

if __name__ == "__main__":
    debug_price_issue()