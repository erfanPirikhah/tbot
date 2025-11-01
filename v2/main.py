# -*- coding: utf-8 -*-
"""
TradeBot Pro - نرم افزار تحلیل پیشرفته بازار ارزهای دیجیتال و فارکس
نسخه: ۳.۰.۰
توسعه دهنده: تیم تحلیل بازار
"""

import sys
import logging
from core.app import TradeBotApp

def setup_logging():
    """تنظیمات پیشرفته لاگینگ"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_bot.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def main():
    """تابع اصلی برنامه"""
    try:
        # تنظیم لاگینگ
        setup_logging()
        
        # ایجاد و اجرای برنامه
        app = TradeBotApp(sys.argv)
        return app.run()
        
    except Exception as e:
        logging.critical(f"خطای بحرانی در اجرای برنامه: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())