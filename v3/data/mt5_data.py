# data/mt5_data.py

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import time

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
    """کلاس مدیریت اتصال و دریافت داده از MetaTrader5"""
    
    def __init__(self):
        self.connected = False
        self._initialize_with_retry()
    
    def _initialize_with_retry(self, max_retries=3):
        """راه‌اندازی با قابلیت تلاش مجدد"""
        for attempt in range(max_retries):
            try:
                if self.initialize_mt5_simple():
                    logger.info(f"✅ MT5 connected on attempt {attempt + 1}")
                    return
                else:
                    logger.warning(f"⚠️ MT5 connection failed on attempt {attempt + 1}")
                    time.sleep(2)
            except Exception as e:
                logger.error(f"❌ MT5 initialization error on attempt {attempt + 1}: {e}")
                time.sleep(2)
        
        logger.error("❌ Failed to connect to MT5 after all retries")
    
    def initialize_mt5_simple(self) -> bool:
        """راه‌اندازی ساده و مستقیم اتصال به MT5"""
        if not MT5_AVAILABLE:
            logger.error("❌ MetaTrader5 is not installed")
            return False
            
        try:
            try:
                mt5.shutdown()
                logger.info("🔁 Previous MT5 connection shut down")
            except:
                pass
            
            logger.info("🔄 Attempting simple MT5 connection...")
            
            if mt5.initialize():
                terminal_info = mt5.terminal_info()
                if terminal_info:
                    self.connected = True
                    terminal_name = getattr(terminal_info, 'name', 'Unknown')
                    logger.info(f"✅ MT5 connected successfully - Terminal: {terminal_name}")
                    return True
                else:
                    logger.error("❌ MT5 connected but terminal info is None")
                    return False
            else:
                error_code = mt5.last_error()
                logger.error(f"❌ MT5 initialization failed. Error code: {error_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error in simple MT5 initialization: {str(e)}")
            self.connected = False
            return False

    def ensure_connected(self) -> bool:
        """اطمینان از اتصال - نسخه ساده و قابل اعتماد"""
        if not MT5_AVAILABLE:
            return False
            
        if self.connected:
            return True
            
        return self.initialize_mt5_simple()
    
    def ensure_symbol_selected(self, symbol: str) -> bool:
        """اطمینان از انتخاب نماد در MT5"""
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
                    logger.error(f"❌ Failed to select symbol {symbol}")
                    return False
                logger.info(f"✅ Symbol {symbol} selected successfully")
                    
            logger.info(f"✅ Symbol {symbol} is available in MT5")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error ensuring symbol selection for {symbol}: {str(e)}")
            return False
    
    def fetch_market_data(self, symbol: str, interval: str, count: int = 100) -> pd.DataFrame:
        """دریافت داده‌های بازار از MT5"""
        logger.info(f"📥 Fetching market data for {symbol}, timeframe: {interval}, count: {count}")
        
        if not MT5_AVAILABLE:
            raise ImportError("❌ MetaTrader5 is not installed.")
            
        if not self.ensure_connected():
            raise ConnectionError("❌ اتصال به MT5 برقرار نشد.")
        
        try:
            if not self.ensure_symbol_selected(symbol):
                raise ValueError(f"❌ نماد {symbol} در MT5 موجود نیست")
            
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
                raise ValueError(f"❌ تایم‌فریم {interval} پشتیبانی نمی‌شود")
            
            logger.info(f"📊 Requesting {count} candles for {symbol} with timeframe {timeframe}")
            
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            
            if rates is None:
                logger.warning("⚠️ No data with copy_rates_from_pos, trying copy_rates_range...")
                utc_from = datetime.now() - timedelta(days=30)
                rates = mt5.copy_rates_range(symbol, timeframe, utc_from, datetime.now())
                
            if rates is None:
                raise ValueError(f"❌ هیچ داده‌ای برای نماد {symbol} دریافت نشد")
            
            df = pd.DataFrame(rates)
            logger.info(f"📈 Raw data received: {len(df)} rows")
            
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            df = df.rename(columns={
                'time': 'open_time',
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            })
            
            df = df.sort_values('open_time').reset_index(drop=True)
            
            if df.empty:
                raise ValueError(f"❌ داده‌های دریافت شده برای {symbol} خالی است")
            
            logger.info(f"✅ دریافت {len(df)} کندل برای {symbol} ({interval})")
            logger.info(f"💰 آخرین قیمت: {df['close'].iloc[-1]:.4f}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ خطا در دریافت داده از MT5 برای {symbol}: {str(e)}")
            raise
    
    def get_current_price(self, symbol: str) -> float:
        """دریافت قیمت لحظه‌ای - نسخه تصحیح شده"""
        logger.info(f"💰 Getting current price for {symbol}")
        
        if not self.ensure_connected():
            logger.error("❌ MT5 not connected")
            return 0.0
            
        try:
            if not self.ensure_symbol_selected(symbol):
                return 0.0
            
            tick = mt5.symbol_info_tick(symbol)
            if tick and tick.bid > 0:
                price = float(tick.bid)
                logger.info(f"✅ قیمت {symbol} (bid): {price}")
                return price
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info and hasattr(symbol_info, 'bid') and symbol_info.bid > 0:
                price = float(symbol_info.bid)
                logger.info(f"✅ قیمت {symbol} از symbol_info: {price}")
                return price
                
            logger.error(f"❌ هیچ قیمت معتبری برای {symbol} یافت نشد")
            return 0.0
                
        except Exception as e:
            logger.error(f"❌ خطا در دریافت قیمت لحظه‌ای {symbol}: {str(e)}")
            return 0.0
    
    def get_available_symbols(self, limit: int = 50) -> list:
        """دریافت لیست نمادهای موجود"""
        if not self.ensure_connected():
            return []
            
        try:
            symbols = mt5.symbols_get()
            symbol_names = [s.name for s in symbols if getattr(s, 'visible', False)][:limit]
            logger.info(f"📋 تعداد نمادهای موجود: {len(symbol_names)}")
            return symbol_names
        except Exception as e:
            logger.error(f"❌ خطا در دریافت لیست نمادها: {str(e)}")
            return []

    def shutdown_mt5(self):
        """قطع اتصال از MT5"""
        if MT5_AVAILABLE:
            try:
                mt5.shutdown()
                self.connected = False
                logger.info("🔌 MT5 connection closed")
            except Exception as e:
                logger.error(f"❌ Error shutting down MT5: {e}")

if MT5_AVAILABLE:
    mt5_fetcher = MT5DataFetcher()
else:
    mt5_fetcher = None
    logger.warning("❌ MT5 fetcher not created because MetaTrader5 is not available")