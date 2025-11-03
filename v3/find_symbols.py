# find_symbols.py
import sys
import os

# اضافه کردن پوشه اصلی پروژه به sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'data'))

from mt5_data import mt5_fetcher

def find_crypto_symbols():
    """
    این تابع به MT5 متصل شده و نمادهای مرتبط با کریپتو را پیدا می‌کند
    """
    if not mt5_fetcher or not mt5_fetcher.ensure_connected():
        print("❌ اتصال به MetaTrader 5 برقرار نیست. لطفاً مطمئن شوید MT5 باز است.")
        return

    print("✅ به MetaTrader 5 متصل شدیم.")
    print("🔍 در حال جستجو برای نمادهای کریپتو...")
    
    # دریافت لیست تمام نمادها
    all_symbols = mt5_fetcher.get_available_symbols(limit=500) # افزایش محدودیت
    
    if not all_symbols:
        print("❌ هیچ نمادی پیدا نشد.")
        return

    # فیلتر کردن نمادهای مرتبط با بیت‌کوین
    btc_symbols = [s for s in all_symbols if 'BTC' in s.upper()]
    
    # فیلتر کردن نمادهای مرتبط با اتریوم
    eth_symbols = [s for s in all_symbols if 'ETH' in s.upper()]

    print("\n--- نمادهای مرتبط با بیت‌کوین ---")
    if btc_symbols:
        for symbol in btc_symbols:
            print(f"  - {symbol}")
    else:
        print("  هیچ نمادی با 'BTC' پیدا نشد.")

    print("\n--- نمادهای مرتبط با اتریوم ---")
    if eth_symbols:
        for symbol in eth_symbols:
            print(f"  - {symbol}")
    else:
        print("  هیچ نمادی با 'ETH' پیدا نشد.")

    print("\n--- ۱۰ نماد اول در لیست ---")
    for symbol in all_symbols[:10]:
        print(f"  - {symbol}")
        
    print(f"\nدر مجموع {len(all_symbols)} نماد پیدا شد.")

if __name__ == "__main__":
    find_crypto_symbols()