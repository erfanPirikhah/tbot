
import pandas as pd
import cryptocompare
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def fetch_market_data(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
    """
    این تابع داده‌های بازار را از API CryptoCompare دریافت می‌کند.

    Args:
        symbol (str): نماد معاملاتی (مثلاً 'BTC', 'ETH').
        interval (str): تایم‌فریم (مثلاً '۱ ساعت', '۱ روز').
        limit (int): تعداد کندل‌های مورد نیاز.

    Returns:
        pd.DataFrame: دیتافریم شامل داده‌های OHLCV.
    
    Raises:
        ValueError: اگر نماد یا تایم‌فریم نامعتبر باشد.
        Exception: برای خطاهای مربوط به درخواست API.
    """
    try:
        # دریافت تنظیمات interval از مپ جدید
        from config import CRYPTOCOMPARE_INTERVAL_MAP
        interval_param = CRYPTOCOMPARE_INTERVAL_MAP.get(interval)
        
        if not interval_param:
            raise ValueError(f"تایم‌فریم '{interval}' پشتیبانی نمی‌شود.")

        # تعیین پارامترها بر اساس تایم‌فریم
        if interval_param == '1h':
            # داده‌های ساعتی - 200 ساعت گذشته (حدود 8 روز)
            data = cryptocompare.get_historical_price_hour(
                symbol, 
                currency='USD', 
                limit=200,
                toTs=datetime.now()
            )
        elif interval_param == '1d':
            # داده‌های روزانه - 365 روز گذشته
            data = cryptocompare.get_historical_price_day(
                symbol,
                currency='USD',
                limit=365,
                toTs=datetime.now()
            )
        else:  # '1w'
            # داده‌های هفتگی - 200 هفته گذشته
            data = cryptocompare.get_historical_price_day(
                symbol,
                currency='USD', 
                limit=200 * 7,  # تقریبی برای هفته
                toTs=datetime.now()
            )

        if not data:
            raise Exception("هیچ داده‌ای از API دریافت نشد.")

        # تبدیل داده به DataFrame
        df = pd.DataFrame(data)
        
        # استانداردسازی نام ستون‌ها
        column_mapping = {
            'time': 'open_time',
            'open': 'open', 
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volumefrom': 'volume',
            'volumeto': 'volume_usd'
        }
        
        # تغییر نام ستون‌های موجود
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # تبدیل timestamp به datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='s')
        
        # اطمینان از عددی بودن ستون‌های قیمت
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # حذف ردیف‌های با داده نامعتبر
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        # مرتب‌سازی بر اساس زمان (قدیمی به جدید)
        df = df.sort_values('open_time').reset_index(drop=True)
        
        # محدود کردن به تعداد درخواستی
        if limit and len(df) > limit:
            df = df.tail(limit)
        
        logger.info("Received %d records for %s with interval %s", len(df), symbol, interval)
        return df

    except Exception as e:
        logger.error("Error fetching data from CryptoCompare: %s", str(e))
        raise Exception(f"Error fetching data: {str(e)}")

def get_current_price(symbol: str) -> float:
    """
    دریافت قیمت لحظه‌ای ارز
    """
    try:
        price_data = cryptocompare.get_price(symbol, currency='USD')
        if price_data and symbol in price_data:
            return float(price_data[symbol]['USD'])
        return 0.0
    except Exception as e:
        logger.error("Error getting current price for %s: %s", symbol, str(e))
        return 0.0

def set_cryptocompare_api_key(api_key: str):
    """
    تنظیم API Key برای CryptoCompare
    """
    cryptocompare.cryptocompare._set_api_key_parameter(api_key)
    logger.info("CryptoCompare API Key set successfully")
