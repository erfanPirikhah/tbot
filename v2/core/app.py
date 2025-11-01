# core/app.py

import sys
import logging
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from ui.main_window import MainWindow
from utils.font_manager import FontManager
from core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class TradeBotApp:
    """کلاس اصلی مدیریت برنامه"""
    
    def __init__(self, argv):
        self.argv = argv
        self.app = None
        self.main_window = None
        self.config = ConfigManager()
        
    def run(self):
        """اجرای برنامه"""
        try:
            # ایجاد برنامه Qt
            self.app = QApplication(self.argv)
            self.app.setApplicationName("TradeBot Pro")
            self.app.setApplicationVersion("3.0.0")
            
            # تنظیم فونت
            FontManager.setup_application_fonts(self.app)
            
            # ایجاد پنجره اصلی
            self.main_window = MainWindow(self.config)
            self.main_window.show()
            
            logger.info("✅ برنامه TradeBot Pro با موفقیت راه‌اندازی شد")
            
            # اجرای حلقه رویداد
            return self.app.exec_()
            
        except Exception as e:
            logger.error(f"❌ خطا در اجرای برنامه: {e}")
            return 1
            
    def shutdown(self):
        """خاموش کردن برنامه"""
        if self.main_window:
            self.main_window.cleanup()
        logger.info("🛑 برنامه TradeBot Pro خاموش شد")