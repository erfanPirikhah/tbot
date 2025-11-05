# data/mt5_data.py

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import time
import os

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    logger.info("✅ MetaTrader5 package imported successfully")
except ImportError as e:
    MT5_AVAILABLE = False
    mt5 = None
    logger.error(f"❌ MetaTrader5 not installed: {e}")

class MT5DataFetcher:
    """کلاس مدیریت اتصال و دریافت داده از MetaTrader5 - نسخه بهبود یافته"""
    
    def __init__(self, max_retries: int = 5, retry_delay: int = 3):
        self.connected = False
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._initialize_with_retry()
    
    def _initialize_with_retry(self):
        """راه‌اندازی با قابلیت تلاش مجدد پیشرفته"""
        for attempt in range(self.max_retries):
            try:
                if self.initialize_mt5_advanced():
                    logger.info(f"✅ MT5 connected successfully on attempt {attempt + 1}")
                    return
                else:
                    logger.warning(f"⚠️ MT5 connection failed on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"❌ MT5 initialization error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        logger.error("❌ Failed to connect to MT5 after all retries")
    
    def initialize_mt5_advanced(self) -> bool:
        """راه‌اندازی پیشرفته اتصال به MT5"""
        if not MT5_AVAILABLE:
            logger.error("❌ MetaTrader5 is not installed")
            return False
            
        try:
            # بستن اتصال قبلی
            try:
                mt5.shutdown()
                logger.info("🔁 Previous MT5 connection shut down")
                time.sleep(1)
            except:
                pass
            
            logger.info("🔄 Attempting advanced MT5 connection...")
            
            # تنظیمات پیشرفته برای اتصال
            mt5_initialize_params = {
                'path': self._detect_mt5_path(),
                'login': 0,  # استفاده از حساب دمو به صورت پیش‌فرض
                'password': "",
                'server': "",
                'timeout': 60000,
                'portable': False
            }
            
            if mt5.initialize(**{k: v for k, v in mt5_initialize_params.items() if v is not None}):
                terminal_info = mt5.terminal_info()
                if terminal_info:
                    self.connected = True
                    terminal_name = getattr(terminal_info, 'name', 'Unknown')
                    terminal_version = getattr(terminal_info, 'version', 'Unknown')
                    logger.info(f"✅ MT5 connected successfully - Terminal: {terminal_name}, Version: {terminal_version}")
                    
                    # تست اتصال با دریافت اطلاعات پایه
                    symbols_count = mt5.symbols_total()
                    if symbols_count > 0:
                        logger.info(f"📊 MT5 initialized with {symbols_count} symbols available")
                        return True
                    else:
                        logger.warning("⚠️ MT5 connected but no symbols available")
                        return False
                else:
                    logger.error("❌ MT5 connected but terminal info is None")
                    return False
            else:
                error_code = mt5.last_error()
                error_msg = self._get_mt5_error_message(error_code)
                logger.error(f"❌ MT5 initialization failed. Error {error_code}: {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error in advanced MT5 initialization: {str(e)}")
            self.connected = False
            return False

    def _detect_mt5_path(self) -> Optional[str]:
        """تشخیص مسیر نصب MT5"""
        possible_paths = [
            r"C:\Program Files\MetaTrader 5\terminal64.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
            r"C:\Program Files\MetaTrader 5\terminal.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal.exe",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"🔍 Found MT5 at: {path}")
                return path
        
        logger.warning("🔍 MT5 path not found in common locations")
        return None

    def _get_mt5_error_message(self, error_code: int) -> str:
        """دریافت پیام خطای MT5"""
        error_messages = {
            1: "ERR_SUCCESS - Successful execution",
            -1: "ERR_INTERNAL_ERROR - Common error",
            -2: "ERR_INTERNAL_ERROR - Internal error",
            -3: "ERR_INVALID_PARAMETER - Invalid parameters",
            -4: "ERR_NOT_ENOUGH_MEMORY - Not enough memory",
            -5: "ERR_NO_HISTORY_DATA - No history data",
            -6: "ERR_MALFUNCTIONAL_TRADE - Trade context is busy",
            -10000: "ERR_TRADE_DISABLED - Trading disabled",
            -10001: "ERR_OLD_VERSION - Old terminal version",
            -10002: "ERR_NO_CONNECTION - No connection to server",
        }
        return error_messages.get(error_code, f"Unknown error code: {error_code}")

    def ensure_connected(self) -> bool:
        """اطمینان از اتصال - نسخه مقاوم در برابر خطا"""
        if not MT5_AVAILABLE:
            logger.error("❌ MetaTrader5 is not available")
            return False
            
        if self.connected:
            # بررسی سلامت اتصال
            try:
                account_info = mt5.account_info()
                if account_info:
                    return True
                else:
                    logger.warning("⚠️ Connection check failed, reconnecting...")
                    self.connected = False
            except:
                logger.warning("⚠️ Connection check failed, reconnecting...")
                self.connected = False
        
        # تلاش برای اتصال مجدد
        return self.initialize_mt5_advanced()
    
    def ensure_symbol_selected(self, symbol: str) -> bool:
        """اطمینان از انتخاب نماد در MT5 با مدیریت خطا"""
        if not self.ensure_connected():
            logger.error("❌ MT5 not connected")
            return False
            
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"❌ Symbol {symbol} not found in MT5")
                return False
                
            if not symbol_info.visible:
                logger.info(f"👀 Symbol {symbol} not visible, selecting...")
                if not mt5.symbol_select(symbol, True):
                    error_code = mt5.last_error()
                    logger.error(f"❌ Failed to select symbol {symbol}. Error: {error_code}")
                    return False
                logger.info(f"✅ Symbol {symbol} selected successfully")
                    
            logger.info(f"✅ Symbol {symbol} is available in MT5")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error ensuring symbol selection for {symbol}: {str(e)}")
            return False
    
    def fetch_market_data(self, symbol: str, interval: str, count: int = 100) -> pd.DataFrame:
        """دریافت داده‌های بازار از MT5 با مدیریت خطاهای پیشرفته"""
        logger.info(f"📥 Fetching market data for {symbol}, timeframe: {interval}, count: {count}")
        
        if not MT5_AVAILABLE:
            raise ImportError("❌ MetaTrader5 is not installed.")
            
        if not self.ensure_connected():
            raise ConnectionError("❌ اتصال به MT5 برقرار نشد.")
        
        try:
            if not self.ensure_symbol_selected(symbol):
                raise ValueError(f"❌ نماد {symbol} در MT5 موجود نیست")
            
            # نگاشت تایم‌فریم
            timeframe_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
                "W1": mt5.TIMEFRAME_W1,
                "MN1": mt5.TIMEFRAME_MN1
            }
            
            timeframe = timeframe_map.get(interval.upper())
            if timeframe is None:
                raise ValueError(f"❌ تایم‌فریم {interval} پشتیبانی نمی‌شود")
            
            logger.info(f"📊 Requesting {count} candles for {symbol} with timeframe {timeframe}")
            
            # روش اول: دریافت از موقعیت فعلی
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            
            # روش دوم: اگر روش اول شکست خورد، دریافت از بازه زمانی
            if rates is None:
                logger.warning("⚠️ No data with copy_rates_from_pos, trying copy_rates_range...")
                utc_from = datetime.now() - timedelta(days=30)
                rates = mt5.copy_rates_range(symbol, timeframe, utc_from, datetime.now())
                
            # روش سوم: اگر هنوز داده‌ای دریافت نشد، تلاش نهایی
            if rates is None:
                logger.warning("⚠️ No data with copy_rates_range, trying alternative method...")
                rates = self._fetch_data_alternative(symbol, timeframe, count)
                
            if rates is None or len(rates) == 0:
                raise ValueError(f"❌ هیچ داده‌ای برای نماد {symbol} دریافت نشد")
            
            df = pd.DataFrame(rates)
            logger.info(f"📈 Raw data received: {len(df)} rows")
            
            # پردازش داده‌ها
            df = self._process_mt5_data(df)
            
            if df.empty:
                raise ValueError(f"❌ داده‌های دریافت شده برای {symbol} خالی است")
            
            logger.info(f"✅ دریافت {len(df)} کندل برای {symbol} ({interval})")
            logger.info(f"💰 آخرین قیمت: {df['close'].iloc[-1]:.4f}")
            logger.info(f"📅 بازه زمانی: {df.index[0]} تا {df.index[-1]}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ خطا در دریافت داده از MT5 برای {symbol}: {str(e)}")
            raise
    
    def _fetch_data_alternative(self, symbol: str, timeframe: int, count: int):
        """روش جایگزین برای دریافت داده"""
        try:
            # دریافت داده با offset
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, count)
            if rates is not None:
                return rates
            
            # کاهش تعداد درخواست
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, min(count, 500))
            return rates
            
        except Exception as e:
            logger.error(f"❌ Alternative data fetch failed: {e}")
            return None

    def _process_mt5_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """پردازش داده‌های دریافتی از MT5"""
        try:
            # تبدیل زمان به datetime و تنظیم به عنوان ایندکس
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.index.name = 'open_time'
            
            # تغییر نام ستون‌ها به فرمت استاندارد
            column_mapping = {
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume',
                'real_volume': 'real_volume',
                'spread': 'spread'
            }
            
            df = df.rename(columns=column_mapping)
            
            # حذف ستون‌های اضافی
            keep_columns = ['open', 'high', 'low', 'close', 'volume']
            available_columns = [col for col in keep_columns if col in df.columns]
            df = df[available_columns]
            
            # مرتب‌سازی بر اساس زمان
            df = df.sort_index()
            
            # حذف داده‌های تکراری
            df = df[~df.index.duplicated(keep='first')]
            
            # اطمینان از انواع داده‌ای صحیح
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # حذف سطرهای با داده‌های نامعتبر
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error processing MT5 data: {e}")
            return df

    def get_current_price(self, symbol: str) -> float:
        """دریافت قیمت لحظه‌ای - نسخه مقاوم در برابر خطا"""
        logger.info(f"💰 Getting current price for {symbol}")
        
        if not self.ensure_connected():
            logger.error("❌ MT5 not connected")
            return 0.0
            
        try:
            if not self.ensure_symbol_selected(symbol):
                return 0.0
            
            # روش اول: استفاده از tick data
            tick = mt5.symbol_info_tick(symbol)
            if tick and hasattr(tick, 'bid') and tick.bid > 0:
                price = float(tick.bid)
                logger.info(f"✅ قیمت {symbol} (bid): {price:.5f}")
                return price
            
            # روش دوم: استفاده از symbol_info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info and hasattr(symbol_info, 'bid') and symbol_info.bid > 0:
                price = float(symbol_info.bid)
                logger.info(f"✅ قیمت {symbol} از symbol_info: {price:.5f}")
                return price
            
            # روش سوم: استفاده از آخرین داده تاریخی
            logger.warning(f"⚠️ No current price available for {symbol}, using historical data")
            historical_data = self.fetch_market_data(symbol, "M1", 1)
            if not historical_data.empty:
                price = float(historical_data['close'].iloc[-1])
                logger.info(f"✅ قیمت {symbol} از داده تاریخی: {price:.5f}")
                return price
                
            logger.error(f"❌ هیچ قیمت معتبری برای {symbol} یافت نشد")
            return 0.0
                
        except Exception as e:
            logger.error(f"❌ خطا در دریافت قیمت لحظه‌ای {symbol}: {str(e)}")
            return 0.0
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """دریافت اطلاعات کامل نماد از MT5"""
        if not self.ensure_connected():
            return None
            
        try:
            if not self.ensure_symbol_selected(symbol):
                return None
                
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                info_dict = {
                    'name': getattr(symbol_info, 'name', ''),
                    'description': getattr(symbol_info, 'description', ''),
                    'bid': getattr(symbol_info, 'bid', 0),
                    'ask': getattr(symbol_info, 'ask', 0),
                    'spread': getattr(symbol_info, 'spread', 0),
                    'digits': getattr(symbol_info, 'digits', 0),
                    'trade_mode': getattr(symbol_info, 'trade_mode', 0),
                    'trade_contract_size': getattr(symbol_info, 'trade_contract_size', 0),
                    'currency_base': getattr(symbol_info, 'currency_base', ''),
                    'currency_profit': getattr(symbol_info, 'currency_profit', ''),
                    'point': getattr(symbol_info, 'point', 0),
                    'volume_min': getattr(symbol_info, 'volume_min', 0),
                    'volume_max': getattr(symbol_info, 'volume_max', 0),
                    'volume_step': getattr(symbol_info, 'volume_step', 0),
                }
                return info_dict
            return None
        except Exception as e:
            logger.error(f"خطا در دریافت اطلاعات نماد {symbol}: {e}")
            return None
    
    def get_available_symbols(self, filter_visible: bool = True, limit: int = 100) -> list:
        """دریافت لیست نمادهای موجود با فیلترهای پیشرفته"""
        if not self.ensure_connected():
            return []
            
        try:
            symbols = mt5.symbols_get()
            filtered_symbols = []
            
            for symbol in symbols:
                if filter_visible and not getattr(symbol, 'visible', False):
                    continue
                
                # فیلتر نمادهای اصلی
                symbol_name = getattr(symbol, 'name', '')
                if symbol_name and not symbol_name.startswith('.'):
                    filtered_symbols.append(symbol_name)
                
                if len(filtered_symbols) >= limit:
                    break
            
            logger.info(f"📋 تعداد نمادهای موجود: {len(filtered_symbols)}")
            return filtered_symbols
        except Exception as e:
            logger.error(f"❌ خطا در دریافت لیست نمادها: {str(e)}")
            return []

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """دریافت اطلاعات حساب"""
        if not self.ensure_connected():
            return None
            
        try:
            account_info = mt5.account_info()
            if account_info:
                return {
                    'login': getattr(account_info, 'login', 0),
                    'name': getattr(account_info, 'name', ''),
                    'server': getattr(account_info, 'server', ''),
                    'currency': getattr(account_info, 'currency', ''),
                    'leverage': getattr(account_info, 'leverage', 0),
                    'balance': getattr(account_info, 'balance', 0),
                    'equity': getattr(account_info, 'equity', 0),
                    'margin': getattr(account_info, 'margin', 0),
                    'free_margin': getattr(account_info, 'free_margin', 0),
                    'profit': getattr(account_info, 'profit', 0),
                }
            return None
        except Exception as e:
            logger.error(f"❌ Error getting account info: {e}")
            return None

    def shutdown_mt5(self):
        """قطع اتصال از MT5 با مدیریت خطا"""
        if MT5_AVAILABLE:
            try:
                mt5.shutdown()
                self.connected = False
                logger.info("🔌 MT5 connection closed successfully")
            except Exception as e:
                logger.error(f"❌ Error shutting down MT5: {e}")

    def __del__(self):
        """دمسترکتور برای بستن اتصال"""
        self.shutdown_mt5()

# ایجاد نمونه جهانی
if MT5_AVAILABLE:
    mt5_fetcher = MT5DataFetcher()
else:
    mt5_fetcher = None
    logger.warning("❌ MT5 fetcher not created because MetaTrader5 is not available")