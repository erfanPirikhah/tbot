# data/mt5_data.py

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any

# بررسی وجود MetaTrader5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None
    logging.warning("MetaTrader5 not installed. Please install with: pip install MetaTrader5")

logger = logging.getLogger(__name__)

class MT5DataFetcher:
    """کلاس مدیریت اتصال و دریافت داده از MetaTrader5"""
    
    def __init__(self):
        self.connected = False
        if MT5_AVAILABLE:
            self.initialize_mt5()
        else:
            logger.warning("MetaTrader5 is not available. Install it to use MT5 features.")
    
    def initialize_mt5(self) -> bool:
        """راه‌اندازی اتصال به MT5"""
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 is not installed")
            return False
            
        try:
            if not mt5.initialize():
                logger.warning("MT5 initialize() failed, trying with portable mode")
                if not mt5.initialize(portable=True):
                    logger.error("MT5 initialize() failed completely")
                    return False
            
            self.connected = True
            logger.info("✅ MT5 connected successfully")
            
            # نمایش اطلاعات نسخه
            version = mt5.version()
            logger.info(f"MT5 version: {version}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing MT5: {str(e)}")
            self.connected = False
            return False
    
    def shutdown_mt5(self):
        """قطع اتصال از MT5"""
        if self.connected and MT5_AVAILABLE:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 connection closed")
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """دریافت اطلاعات نماد"""
        if not self.connected or not MT5_AVAILABLE:
            return None
            
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                logger.error(f"Symbol {symbol} not found")
                return None
            
            return {
                'name': info.name,
                'description': info.description,
                'digits': info.digits,
                'spread': info.spread,
                'trade_mode': info.trade_mode,
                'point': info.point,
                'volume_min': info.volume_min,
                'volume_max': info.volume_max
            }
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {str(e)}")
            return None
    
    def fetch_market_data(self, symbol: str, interval: str, count: int = 500) -> pd.DataFrame:
        """
        دریافت داده‌های بازار از MT5
        """
        if not MT5_AVAILABLE:
            raise ImportError("MetaTrader5 is not installed. Please install it with: pip install MetaTrader5")
            
        if not self.connected:
            if not self.initialize_mt5():
                raise ConnectionError("اتصال به MT5 برقرار نشد. لطفاً مطمئن شوید MetaTrader5 اجرا است.")
        
        try:
            # انتخاب نماد
            if not mt5.symbol_select(symbol, True):
                logger.error(f"نماد {symbol} یافت نشد")
                raise ValueError(f"نماد {symbol} در MT5 موجود نیست")
            
            # مپ تایم‌فریم
            timeframe_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
                "W1": mt5.TIMEFRAME_W1
            }
            
            timeframe = timeframe_map.get(interval)
            if timeframe is None:
                raise ValueError(f"تایم‌فریم {interval} پشتیبانی نمی‌شود")
            
            # دریافت داده‌ها
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            
            if rates is None or len(rates) == 0:
                raise ValueError(f"هیچ داده‌ای برای نماد {symbol} دریافت نشد")
            
            # تبدیل به DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # تغییر نام ستون‌ها برای سازگاری
            df = df.rename(columns={
                'time': 'open_time',
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            })
            
            # مرتب‌سازی بر اساس زمان
            df = df.sort_values('open_time').reset_index(drop=True)
            
            logger.info(f"✅ دریافت {len(df)} کندل برای {symbol} ({interval})")
            return df
            
        except Exception as e:
            logger.error(f"خطا در دریافت داده از MT5 برای {symbol}: {str(e)}")
            raise
    
    def get_current_price(self, symbol: str) -> float:
        """دریافت قیمت لحظه‌ای با مدیریت خطای بهتر"""
        if not self.connected or not MT5_AVAILABLE:
            logger.warning(f"MT5 not connected or available for {symbol}")
            return 0.0
            
        try:
            # مطمئن شویم نماد انتخاب شده است
            if not mt5.symbol_select(symbol, True):
                logger.error(f"نماد {symbol} در MT5 انتخاب نشد")
                return 0.0
            
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"اطلاعات تیک برای {symbol} دریافت نشد")
                return 0.0
            
            # استفاده از bid یا last قیمت
            if tick.bid > 0:
                price = float(tick.bid)
                logger.info(f"قیمت {symbol}: {price}")
                return price
            elif tick.last > 0:
                price = float(tick.last)
                logger.info(f"قیمت {symbol} (last): {price}")
                return price
            else:
                logger.error(f"قیمت معتبر برای {symbol} یافت نشد. bid: {tick.bid}, last: {tick.last}")
                return 0.0
                
        except Exception as e:
            logger.error(f"خطا در دریافت قیمت لحظه‌ای {symbol}: {str(e)}")
            return 0.0
    
    def get_available_symbols(self) -> list:
        """دریافت لیست نمادهای موجود"""
        if not self.connected or not MT5_AVAILABLE:
            return []
            
        try:
            symbols = mt5.symbols_get()
            symbol_names = [s.name for s in symbols]
            logger.info(f"تعداد نمادهای موجود: {len(symbol_names)}")
            return symbol_names
        except Exception as e:
            logger.error(f"خطا در دریافت لیست نمادها: {str(e)}")
            return []

# نمونه جهانی - فقط اگر MT5 در دسترس باشد ایجاد می‌شود
if MT5_AVAILABLE:
    mt5_fetcher = MT5DataFetcher()
else:
    mt5_fetcher = None
    logger.warning("MT5 fetcher not created because MetaTrader5 is not available")