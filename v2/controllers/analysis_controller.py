# controllers/analysis_controller.py

import logging
import pandas as pd
from strategies.improved_advanced_rsi_strategy import ImprovedAdvancedRsiStrategy
from data.data_fetcher import fetch_market_data
from indicators.rsi import calculate_rsi
from config import RSI_PERIOD, MT5_SYMBOL_MAP, CRYPTOCOMPARE_SYMBOL_MAP

logger = logging.getLogger(__name__)

class AnalysisController:
    """کنترلر مدیریت تحلیل بازار"""
    
    def __init__(self):
        self.strategy = ImprovedAdvancedRsiStrategy()
        self.current_data = None
        self.current_signal = None
        
    def analyze_market(self, data_source, symbol_display, interval_display):
        """انجام تحلیل بازار"""
        logger.info(f"شروع تحلیل برای {symbol_display} از {data_source}")
        
        # دریافت کد نماد
        symbol = self.get_symbol_code(symbol_display, data_source)
        if not symbol:
            raise ValueError(f"نماد {symbol_display} یافت نشد")
        
        # دریافت داده‌ها
        raw_data = fetch_market_data(symbol, interval_display, 
                                   data_source="MT5" if data_source == "MetaTrader5" else "CRYPTOCOMPARE")
        
        if raw_data.empty:
            raise ValueError("داده‌های تاریخی دریافت نشد")
            
        # محاسبه اندیکاتورها
        data_with_rsi = calculate_rsi(raw_data, period=RSI_PERIOD)
        
        # تولید سیگنال
        signal_info = self.strategy.generate_signal(data_with_rsi)
        
        # ذخیره نتایج
        self.current_data = data_with_rsi
        self.current_signal = signal_info
        
        logger.info(f"تحلیل با موفقیت انجام شد. سیگنال: {signal_info['action']}")
        return signal_info
        
    def get_symbol_code(self, symbol_display, data_source):
        """دریافت کد نماد از نمایش آن"""
        if data_source == "MetaTrader5":
            return MT5_SYMBOL_MAP.get(symbol_display)
        else:
            return CRYPTOCOMPARE_SYMBOL_MAP.get(symbol_display)
        
    def get_performance_metrics(self):
        """دریافت معیارهای عملکرد"""
        return self.strategy.get_performance_metrics()
        
    def get_trade_history(self):
        """دریافت تاریخچه معاملات"""
        return self.strategy.trade_history
        
    def get_current_data(self):
        """دریافت داده فعلی برای Fallback قیمت"""
        return self.current_data
        
    def update_strategy_params(self, new_params):
        """به‌روزرسانی پارامترهای استراتژی"""
        self.strategy = ImprovedAdvancedRsiStrategy(**new_params)
        logger.info("✅ پارامترهای استراتژی به‌روزرسانی شد")
        
    def cleanup(self):
        """تمیزکاری منابع"""
        # در صورت نیاز پاک‌سازی منابع
        pass