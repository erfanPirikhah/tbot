# core/config_manager.py

import logging
from PyQt5.QtCore import QSettings
from config import (
    CRYPTOCOMPARE_API_KEY, IMPROVED_STRATEGY_PARAMS,
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT
)

logger = logging.getLogger(__name__)

class ConfigManager:
    """مدیریت تنظیمات برنامه"""
    
    def __init__(self):
        self.settings = QSettings("TradeBotPro", "v3")
        self.load_settings()
        
    def load_settings(self):
        """بارگذاری تنظیمات از ذخیره‌سازی"""
        self.api_key = self.settings.value("api_key", CRYPTOCOMPARE_API_KEY)
        
        # بارگذاری پارامترهای استراتژی
        strategy_params = {}
        for key, default_value in IMPROVED_STRATEGY_PARAMS.items():
            strategy_params[key] = self.settings.value(f"strategy/{key}", default_value, type=type(default_value))
        
        self.strategy_params = strategy_params
        
    def save_settings(self):
        """ذخیره تنظیمات"""
        self.settings.setValue("api_key", self.api_key)
        
        # ذخیره پارامترهای استراتژی
        for key, value in self.strategy_params.items():
            self.settings.setValue(f"strategy/{key}", value)
            
        self.settings.sync()
        logger.info("✅ تنظیمات ذخیره شد")
        
    def get_strategy_param(self, key, default=None):
        """دریافت پارامتر استراتژی"""
        return self.strategy_params.get(key, default)
        
    def set_strategy_param(self, key, value):
        """تنظیم پارامتر استراتژی"""
        self.strategy_params[key] = value
        
    def reset_to_defaults(self):
        """بازنشانی به تنظیمات پیش‌فرض"""
        self.api_key = CRYPTOCOMPARE_API_KEY
        self.strategy_params = IMPROVED_STRATEGY_PARAMS.copy()
        self.save_settings()
        logger.info("🔄 تنظیمات به پیش‌فرض بازنشانی شد")
        
    def get_all_strategy_params(self):
        """دریافت تمام پارامترهای استراتژی"""
        return self.strategy_params.copy()