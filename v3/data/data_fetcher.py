import pandas as pd
import cryptocompare
from datetime import datetime
import logging
from typing import Optional
import sys
import os

# --- شروع کد اصلاحی ---
# اضافه کردن پوشه اصلی پروژه به sys.path تا فایل config.py پیدا شود
# این کد به پایتون می‌گوید که در پوشه والد هم به دنبال ماژول‌ها بگردد
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# --- پایان کد اصلاحی ---

# ایمپورت ماژول جدید MT5
try:
    from .mt5_data import mt5_fetcher, MT5_AVAILABLE
except ImportError as e:
    MT5_AVAILABLE = False
    mt5_fetcher = None
    logging.warning(f"MetaTrader5 not available: {e}")

logger = logging.getLogger(__name__)

def fetch_market_data(symbol: str, interval: str, limit: int = 100, data_source: str = "AUTO") -> pd.DataFrame:
    """
    دریافت داده‌های بازار از منابع مختلف
    """
    from config import CRYPTOCOMPARE_SYMBOL_MAP, MT5_SYMBOL_MAP, CRYPTOCOMPARE_INTERVAL_MAP, MT5_INTERVAL_MAP
    
    # تشخیص خودکار منبع داده
    if data_source == "AUTO":
        if symbol in MT5_SYMBOL_MAP.values():
            data_source = "MT5"
        elif symbol in CRYPTOCOMPARE_SYMBOL_MAP.values():
            data_source = "CRYPTOCOMPARE"
        else:
            data_source = "MT5"
    
    logger.info(f"دریافت داده برای {symbol} از {data_source} با تایم‌فریم {interval}")
    
    if data_source == "MT5" and MT5_AVAILABLE:
        return fetch_mt5_data(symbol, interval, limit)
    else:
        return fetch_cryptocompare_data(symbol, interval, limit)

def fetch_mt5_data(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
    """دریافت داده از MetaTrader5"""
    try:
        # مپ تایم‌فریم
        interval_map = {
            "۱ دقیقه": "M1",
            "۵ دقیقه": "M5", 
            "۱۵ دقیقه": "M15",
            "۳۰ دقیقه": "M30",
            "۱ ساعت": "H1",
            "۴ ساعت": "H4",
            "۱ روز": "D1",
            "۱ هفته": "W1"
        }
        
        mt5_interval = interval_map.get(interval, "H1")
        logger.info(f"دریافت داده MT5 برای {symbol} با تایم‌فریم {mt5_interval}")
        
        if not MT5_AVAILABLE or not mt5_fetcher:
            raise ValueError("MT5 در دسترس نیست")
            
        data = mt5_fetcher.fetch_market_data(symbol, mt5_interval, limit)
        
        if data.empty:
            raise ValueError(f"داده‌ای از MT5 برای {symbol} دریافت نشد")
            
        return data
        
    except Exception as e:
        error_msg = f"خطا در دریافت داده از MT5 برای {symbol}: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def fetch_cryptocompare_data(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
    """دریافت داده از CryptoCompare"""
    try:
        from config import CRYPTOCOMPARE_INTERVAL_MAP
        
        interval_param = CRYPTOCOMPARE_INTERVAL_MAP.get(interval)
        
        if not interval_param:
            raise ValueError(f"تایم‌فریم '{interval}' پشتیبانی نمی‌شود.")

        logger.info(f"دریافت داده از CryptoCompare برای {symbol} ({interval_param})")

        if interval_param == '1h':
            data = cryptocompare.get_historical_price_hour(
                symbol, 
                currency='USD', 
                limit=min(limit, 200),
                toTs=datetime.now()
            )
        elif interval_param == '1d':
            data = cryptocompare.get_historical_price_day(
                symbol,
                currency='USD',
                limit=min(limit, 365),
                toTs=datetime.now()
            )
        else:  # '1w'
            data = cryptocompare.get_historical_price_day(
                symbol,
                currency='USD', 
                limit=min(limit, 200 * 7),
                toTs=datetime.now()
            )

        if not data:
            raise Exception("هیچ داده‌ای از API دریافت نشد.")

        df = pd.DataFrame(data)
        
        column_mapping = {
            'time': 'open_time',
            'open': 'open', 
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volumefrom': 'volume',
            'volumeto': 'volume_usd'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='s')
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        df = df.sort_values('open_time').reset_index(drop=True)
        
        if limit and len(df) > limit:
            df = df.tail(limit)
        
        logger.info(f"دریافت {len(df)} رکورد از CryptoCompare برای {symbol}")
        return df

    except Exception as e:
        logger.error(f"Error fetching data from CryptoCompare: {str(e)}")
        raise

def get_current_price(symbol: str, data_source: str = "AUTO") -> float:
    """دریافت قیمت لحظه‌ای - نسخه تصحیح شده"""
    from config import MT5_SYMBOL_MAP, CRYPTOCOMPARE_SYMBOL_MAP
    
    logger.info(f"💰 دریافت قیمت لحظه‌ای برای {symbol} از {data_source}")
    
    # تشخیص خودکار منبع داده
    if data_source == "AUTO":
        if symbol in MT5_SYMBOL_MAP.values() and MT5_AVAILABLE:
            data_source = "MT5"
        else:
            data_source = "CRYPTOCOMPARE"
    
    logger.info(f"🔍 استفاده از منبع داده: {data_source} برای نماد: {symbol}")
    
    if data_source == "MT5" and MT5_AVAILABLE:
        try:
            price = get_mt5_price(symbol)
            if price > 0:
                logger.info(f"✅ قیمت MT5 برای {symbol}: {price}")
                return price
            else:
                logger.warning(f"⚠️ قیمت MT5 برای {symbol} صفر یا نامعتبر است")
                # Fallback: سعی کن از داده‌های تاریخی استفاده کنی
                return get_fallback_price(symbol, data_source)
        except Exception as e:
            logger.error(f"❌ خطا در دریافت قیمت MT5 برای {symbol}: {e}")
            return get_fallback_price(symbol, data_source)
    
    # Fallback به CryptoCompare
    if symbol in CRYPTOCOMPARE_SYMBOL_MAP.values():
        try:
            price = get_cryptocompare_price(symbol)
            if price > 0:
                logger.info(f"✅ قیمت CryptoCompare برای {symbol}: {price}")
                return price
        except Exception as e:
            logger.error(f"❌ خطا در دریافت قیمت CryptoCompare برای {symbol}: {e}")
    
    # Fallback نهایی
    return get_fallback_price(symbol, data_source)

def get_fallback_price(symbol: str, data_source: str) -> float:
    """دریافت قیمت جایگزین از داده‌های تاریخی"""
    try:
        logger.info(f"🔄 استفاده از Fallback برای قیمت {symbol}")
        
        # دریافت داده‌های تاریخی اخیر
        data = fetch_market_data(symbol, "H1", 1, data_source)
        
        if not data.empty and 'close' in data.columns:
            price = data['close'].iloc[-1]
            logger.info(f"✅ قیمت Fallback از داده تاریخی برای {symbol}: {price}")
            return price
        else:
            logger.error(f"❌ Fallback نیز برای {symbol} شکست خورد")
            return 0.0
            
    except Exception as e:
        logger.error(f"❌ خطا در Fallback قیمت برای {symbol}: {e}")
        return 0.0

def get_mt5_price(symbol: str) -> float:
    """دریافت قیمت از MT5 - نسخه بهبود یافته"""
    if not MT5_AVAILABLE or not mt5_fetcher:
        logger.error("❌ MT5 در دسترس نیست")
        return 0.0
        
    try:
        # مطمئن شویم متصل هستیم
        if not mt5_fetcher.ensure_connected():
            logger.error("❌ اتصال به MT5 برقرار نیست")
            return 0.0
        
        # دریافت قیمت
        price = mt5_fetcher.get_current_price(symbol)
        
        if price <= 0:
            logger.warning(f"⚠️ قیمت دریافتی از MT5 برای {symbol} نامعتبر است: {price}")
            
            # روش جایگزین: استفاده از symbol_info مستقیم
            import MetaTrader5 as mt5
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info and hasattr(symbol_info, 'bid') and symbol_info.bid > 0:
                price = float(symbol_info.bid)
                logger.info(f"✅ قیمت جایگزین از symbol_info برای {symbol}: {price}")
                return price
        
        return price
        
    except Exception as e:
        logger.error(f"❌ خطا در دریافت قیمت MT5 برای {symbol}: {e}")
        return 0.0

def get_cryptocompare_price(symbol: str) -> float:
    """دریافت قیمت از CryptoCompare"""
    try:
        price_data = cryptocompare.get_price(symbol, currency='USD')
        if price_data and symbol in price_data:
            return float(price_data[symbol]['USD'])
        return 0.0
    except Exception as e:
        logger.error(f"Error getting current price for {symbol}: {str(e)}")
        return 0.0

def set_cryptocompare_api_key(api_key: str):
    """تنظیم API Key برای CryptoCompare"""
    cryptocompare.cryptocompare._set_api_key_parameter(api_key)
    logger.info("CryptoCompare API Key set successfully")

def get_price_from_historical(symbol: str, data_source: str) -> float:
    """دریافت قیمت از آخرین داده تاریخی - راه‌حل قطعی"""
    try:
        logger.info(f"📊 دریافت قیمت از داده تاریخی برای {symbol}")
        
        # دریافت آخرین داده
        data = fetch_market_data(symbol, "H1", 2, data_source)
        
        if not data.empty and 'close' in data.columns:
            price = data['close'].iloc[-1]
            logger.info(f"✅ قیمت از داده تاریخی برای {symbol}: {price}")
            return price
        else:
            logger.error(f"❌ دریافت قیمت از داده تاریخی برای {symbol} شکست خورد")
            return 0.0
            
    except Exception as e:
        logger.error(f"❌ خطا در دریافت قیمت تاریخی برای {symbol}: {e}")
        return 0.0