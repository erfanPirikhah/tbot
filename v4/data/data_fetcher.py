# data/data_fetcher.py

import pandas as pd
import cryptocompare
from datetime import datetime
import logging
from typing import Optional, Dict, Any
import sys
import os
import warnings

# تنظیم مسیرهای پروژه
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

warnings.filterwarnings('ignore')

# ایمپورت ماژول MT5
try:
    from .mt5_data import mt5_fetcher, MT5_AVAILABLE
except ImportError as e:
    MT5_AVAILABLE = False
    mt5_fetcher = None
    logging.warning(f"MetaTrader5 not available: {e}")

logger = logging.getLogger(__name__)

class DataFetcher:
    """کلاس اصلی برای دریافت داده از منابع مختلف"""
    
    def __init__(self, crypto_api_key: Optional[str] = None):
        self.crypto_api_key = crypto_api_key
        if crypto_api_key:
            self.set_cryptocompare_api_key(crypto_api_key)
        
        logger.info("✅ DataFetcher initialized successfully")

    def fetch_market_data(
        self, 
        symbol: str, 
        interval: str, 
        limit: int = 100, 
        data_source: str = "AUTO"
    ) -> pd.DataFrame:
        """
        دریافت داده‌های بازار از منابع مختلف
        
        Args:
            symbol: نماد مورد نظر
            interval: تایم‌فریم داده
            limit: تعداد داده‌های مورد نیاز
            data_source: منبع داده (AUTO, MT5, CRYPTOCOMPARE)
        """
        try:
            from config.market_config import SYMBOL_MAPPING, TIMEFRAME_MAPPING
            
            # تشخیص خودکار منبع داده
            if data_source == "AUTO":
                data_source = self._detect_data_source(symbol)
            
            logger.info(f"📥 دریافت داده برای {symbol} از {data_source} با تایم‌فریم {interval}")
            
            if data_source == "MT5" and MT5_AVAILABLE:
                return self.fetch_mt5_data(symbol, interval, limit)
            else:
                return self.fetch_cryptocompare_data(symbol, interval, limit)
                
        except Exception as e:
            logger.error(f"❌ خطا در دریافت داده: {e}")
            raise

    def _detect_data_source(self, symbol: str) -> str:
        """تشخیص خودکار منبع داده بر اساس نماد"""
        from config.market_config import SYMBOL_MAPPING
        
        # بررسی نمادهای MT5
        mt5_symbols = SYMBOL_MAPPING["MT5"]
        if symbol.upper() in mt5_symbols.values() or symbol in mt5_symbols.values():
            if MT5_AVAILABLE and mt5_fetcher and mt5_fetcher.ensure_connected():
                return "MT5"
        
        # پیش‌فرض CryptoCompare
        return "CRYPTOCOMPARE"

    def fetch_mt5_data(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """دریافت داده از MetaTrader5"""
        try:
            # نگاشت تایم‌فریم
            interval_map = {
                "۱ دقیقه": "M1", "1m": "M1", "M1": "M1",
                "۵ دقیقه": "M5", "5m": "M5", "M5": "M5",
                "۱۵ دقیقه": "M15", "15m": "M15", "M15": "M15",
                "۳۰ دقیقه": "M30", "30m": "M30", "M30": "M30",
                "۱ ساعت": "H1", "1h": "H1", "H1": "H1",
                "۴ ساعت": "H4", "4h": "H4", "H4": "H4",
                "۱ روز": "D1", "1d": "D1", "D1": "D1",
                "۱ هفته": "W1", "1w": "W1", "W1": "W1"
            }
            
            mt5_interval = interval_map.get(interval, "H1")
            logger.info(f"🔍 دریافت داده MT5 برای {symbol} با تایم‌فریم {mt5_interval}")
            
            if not MT5_AVAILABLE or not mt5_fetcher:
                raise ValueError("❌ MT5 در دسترس نیست")
                
            data = mt5_fetcher.fetch_market_data(symbol, mt5_interval, limit)
            
            if data.empty:
                raise ValueError(f"❌ داده‌ای از MT5 برای {symbol} دریافت نشد")
                
            logger.info(f"✅ دریافت {len(data)} رکورد از MT5 برای {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"❌ خطا در دریافت داده از MT5 برای {symbol}: {str(e)}")
            raise

    def fetch_cryptocompare_data(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """دریافت داده از CryptoCompare"""
        try:
            from config.market_config import CRYPTOCOMPARE_INTERVAL_MAP
            
            interval_param = CRYPTOCOMPARE_INTERVAL_MAP.get(interval)
            
            if not interval_param:
                raise ValueError(f"❌ تایم‌فریم '{interval}' پشتیبانی نمی‌شود.")

            logger.info(f"🔍 دریافت داده از CryptoCompare برای {symbol} ({interval_param})")

            # محدودیت تعداد داده‌ها
            actual_limit = min(limit, 2000)  # محدودیت API
            
            if interval_param in ['1m', '5m', '15m', '30m']:
                data = cryptocompare.get_historical_price_minute(
                    symbol, 
                    currency='USD', 
                    limit=actual_limit,
                    toTs=datetime.now()
                )
            elif interval_param == '1h':
                data = cryptocompare.get_historical_price_hour(
                    symbol, 
                    currency='USD', 
                    limit=actual_limit,
                    toTs=datetime.now()
                )
            elif interval_param == '1d':
                data = cryptocompare.get_historical_price_day(
                    symbol,
                    currency='USD',
                    limit=actual_limit,
                    toTs=datetime.now()
                )
            else:  # '1w'
                data = cryptocompare.get_historical_price_day(
                    symbol,
                    currency='USD', 
                    limit=min(actual_limit * 7, 2000),
                    toTs=datetime.now()
                )

            if not data:
                raise Exception("❌ هیچ داده‌ای از API دریافت نشد.")

            df = pd.DataFrame(data)
            
            # پردازش و استانداردسازی داده‌ها
            df = self._process_cryptocompare_data(df, interval_param)
            
            if limit and len(df) > limit:
                df = df.tail(limit)
            
            logger.info(f"✅ دریافت {len(df)} رکورد از CryptoCompare برای {symbol}")
            return df

        except Exception as e:
            logger.error(f"❌ Error fetching data from CryptoCompare: {str(e)}")
            raise

    def _process_cryptocompare_data(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """پردازش داده‌های CryptoCompare"""
        try:
            # نگاشت ستون‌ها
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
            
            # تبدیل زمان
            if 'open_time' in df.columns:
                df['open_time'] = pd.to_datetime(df['open_time'], unit='s')
                df.set_index('open_time', inplace=True)
            
            # تبدیل انواع داده‌ای
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # حذف داده‌های نامعتبر
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            df = df.sort_index().reset_index(drop=False)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error processing CryptoCompare data: {e}")
            return df

    def get_current_price(self, symbol: str, data_source: str = "AUTO") -> float:
        """دریافت قیمت لحظه‌ای - نسخه مقاوم در برابر خطا"""
        logger.info(f"💰 دریافت قیمت لحظه‌ای برای {symbol}")
        
        # تشخیص منبع داده
        if data_source == "AUTO":
            data_source = self._detect_data_source(symbol)
        
        logger.info(f"🔍 استفاده از منبع داده: {data_source} برای نماد: {symbol}")
        
        if data_source == "MT5" and MT5_AVAILABLE:
            return self._get_mt5_current_price(symbol)
        else:
            return self._get_cryptocompare_current_price(symbol)

    def _get_mt5_current_price(self, symbol: str) -> float:
        """دریافت قیمت از MT5"""
        try:
            price = mt5_fetcher.get_current_price(symbol)
            if price > 0:
                logger.info(f"✅ قیمت MT5 برای {symbol}: {price:.5f}")
                return price
            else:
                logger.warning(f"⚠️ قیمت MT5 برای {symbol} نامعتبر است")
                return self._get_fallback_price(symbol, "MT5")
                
        except Exception as e:
            logger.error(f"❌ خطا در دریافت قیمت MT5 برای {symbol}: {e}")
            return self._get_fallback_price(symbol, "MT5")

    def _get_cryptocompare_current_price(self, symbol: str) -> float:
        """دریافت قیمت از CryptoCompare"""
        try:
            price_data = cryptocompare.get_price(symbol, currency='USD')
            if price_data and symbol in price_data:
                price = float(price_data[symbol]['USD'])
                logger.info(f"✅ قیمت CryptoCompare برای {symbol}: {price:.5f}")
                return price
            else:
                logger.warning(f"⚠️ قیمت CryptoCompare برای {symbol} یافت نشد")
                return self._get_fallback_price(symbol, "CRYPTOCOMPARE")
                
        except Exception as e:
            logger.error(f"❌ Error getting current price for {symbol}: {str(e)}")
            return self._get_fallback_price(symbol, "CRYPTOCOMPARE")

    def _get_fallback_price(self, symbol: str, data_source: str) -> float:
        """دریافت قیمت جایگزین از داده‌های تاریخی"""
        try:
            logger.info(f"🔄 استفاده از Fallback برای قیمت {symbol}")
            
            # دریافت داده‌های تاریخی اخیر
            data = self.fetch_market_data(symbol, "H1", 2, data_source)
            
            if not data.empty and 'close' in data.columns:
                price = float(data['close'].iloc[-1])
                logger.info(f"✅ قیمت Fallback از داده تاریخی برای {symbol}: {price:.5f}")
                return price
            else:
                logger.error(f"❌ Fallback نیز برای {symbol} شکست خورد")
                return 0.0
                
        except Exception as e:
            logger.error(f"❌ خطا در Fallback قیمت برای {symbol}: {e}")
            return 0.0

    def get_symbol_info(self, symbol: str, data_source: str = "AUTO") -> Optional[Dict[str, Any]]:
        """دریافت اطلاعات نماد"""
        if data_source == "AUTO":
            data_source = self._detect_data_source(symbol)
        
        if data_source == "MT5" and MT5_AVAILABLE:
            return mt5_fetcher.get_symbol_info(symbol)
        else:
            return self._get_cryptocompare_symbol_info(symbol)

    def _get_cryptocompare_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """دریافت اطلاعات نماد از CryptoCompare"""
        try:
            # این تابع می‌تواند با API دیگر تکمیل شود
            return {
                'name': symbol,
                'description': f'Cryptocurrency {symbol}',
                'source': 'CRYPTOCOMPARE'
            }
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None

    def get_available_symbols(self, data_source: str = "MT5") -> list:
        """دریافت لیست نمادهای موجود"""
        if data_source == "MT5" and MT5_AVAILABLE:
            return mt5_fetcher.get_available_symbols()
        else:
            from config.market_config import SYMBOL_MAPPING
            return list(SYMBOL_MAPPING["CRYPTOCOMPARE"].values())

    def set_cryptocompare_api_key(self, api_key: str):
        """تنظیم API Key برای CryptoCompare"""
        try:
            cryptocompare.cryptocompare._set_api_key_parameter(api_key)
            self.crypto_api_key = api_key
            logger.info("✅ CryptoCompare API Key set successfully")
        except Exception as e:
            logger.error(f"❌ Error setting CryptoCompare API key: {e}")

    def test_connection(self, data_source: str = "AUTO") -> bool:
        """تست اتصال به منابع داده"""
        try:
            if data_source in ["AUTO", "MT5"] and MT5_AVAILABLE:
                if mt5_fetcher and mt5_fetcher.ensure_connected():
                    logger.info("✅ MT5 connection test: PASSED")
                    return True
                else:
                    logger.warning("⚠️ MT5 connection test: FAILED")
            
            if data_source in ["AUTO", "CRYPTOCOMPARE"]:
                # تست ساده با دریافت قیمت بیت‌کوین
                price = self.get_current_price("BTC", "CRYPTOCOMPARE")
                if price > 0:
                    logger.info("✅ CryptoCompare connection test: PASSED")
                    return True
                else:
                    logger.warning("⚠️ CryptoCompare connection test: FAILED")
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False

# ایجاد نمونه جهانی
data_fetcher = DataFetcher()