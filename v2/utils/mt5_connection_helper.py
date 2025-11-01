# utils/mt5_connection_helper.py

import logging
import subprocess
import sys
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class MT5ConnectionHelper:
    """کلاس کمکی برای اتصال به MetaTrader5"""
    
    @staticmethod
    def find_mt5_installation():
        """یافتن مسیر نصب MetaTrader5"""
        common_paths = [
            # مسیرهای معمول نصب MT5 در ویندوز
            Path("C:/Program Files/MetaTrader 5/terminal64.exe"),
            Path("C:/Program Files (x86)/MetaTrader 5/terminal64.exe"),
            Path("C:/Program Files/MetaTrader 5/terminal.exe"),
            Path("C:/Program Files (x86)/MetaTrader 5/terminal.exe"),
            Path(os.path.expanduser("~/AppData/Local/Programs/MetaTrader 5/terminal64.exe")),
            Path(os.path.expanduser("~/AppData/Local/Programs/MetaTrader 5/terminal.exe")),
        ]
        
        for path in common_paths:
            if path.exists():
                logger.info(f"MT5 found at: {path}")
                return path
        return None
    
    @staticmethod
    def launch_mt5():
        """اجرای MetaTrader5"""
        mt5_path = MT5ConnectionHelper.find_mt5_installation()
        if mt5_path:
            try:
                subprocess.Popen([str(mt5_path)])
                logger.info("MetaTrader5 launched successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to launch MT5: {e}")
                return False
        else:
            logger.error("MetaTrader5 not found. Please install it first.")
            return False
    
    @staticmethod
    def check_mt5_requirements():
        """بررسی پیش‌نیازهای MT5"""
        try:
            import MetaTrader5 as mt5
            return True, "MetaTrader5 package is installed"
        except ImportError:
            return False, "MetaTrader5 package not installed. Run: pip install MetaTrader5"
    
    @staticmethod
    def get_connection_guide():
        """راهنمای اتصال به MT5"""
        return """
        🔧 راهنمای اتصال به MetaTrader5:
        
        ۱. **نصب MetaTrader5:**
           - از سایت رسمی MetaTrader5 دانلود و نصب کنید
           - یا با دستور زیر کتابخانه را نصب کنید: 
             pip install MetaTrader5
        
        ۲. **اجرای MT5:**
           - برنامه MetaTrader5 را اجرا کنید
           - با حساب دمو یا واقعی وارد شوید
        
        ۳. **فعال‌سازی نمادها:**
           - در MT5، به Market Watch بروید (Ctrl+M)
           - راست‌کلیک → Symbols
           - نمادهای مورد نیاز مانند XAUUSD, EURUSD را انتخاب کنید
        
        ۴. **بررسی اتصال:**
           - در نرم‌افزار، دکمه "تنظیمات MT5" → "تست اتصال" را بزنید
        
        ۵. **راه‌حل‌های جایگزین:**
           - اگر مشکل persists، از CryptoCompare برای تحلیل ارزهای دیجیتال استفاده کنید
        """