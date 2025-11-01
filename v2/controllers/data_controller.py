# controllers/data_controller.py

import logging
from data.data_fetcher import get_current_price
from data.mt5_data import MT5_AVAILABLE, mt5_fetcher

logger = logging.getLogger(__name__)

class DataController:
    """کنترلر مدیریت داده‌ها"""
    
    def __init__(self):
        self.current_price = 0.0
        self.market_data = None
        
    def get_current_price(self, symbol, data_source):
        """دریافت قیمت لحظه‌ای"""
        try:
            self.current_price = get_current_price(symbol, data_source)
            if self.current_price == 0:
                logger.warning(f"قیمت صفر دریافت شده برای {symbol} از {data_source}")
            return self.current_price
        except Exception as e:
            logger.error(f"خطا در دریافت قیمت برای {symbol}: {e}")
            return 0.0
        
    def check_mt5_connection(self):
        """بررسی اتصال MT5"""
        if not MT5_AVAILABLE:
            return False, "MetaTrader5 در دسترس نیست"
            
        if mt5_fetcher and mt5_fetcher.connected:
            return True, "متصل"
        else:
            return False, "قطع"
            
    def cleanup(self):
        """تمیزکاری منابع"""
        # بستن اتصالات اگر نیاز باشد
        if MT5_AVAILABLE and mt5_fetcher:
            mt5_fetcher.shutdown_mt5()