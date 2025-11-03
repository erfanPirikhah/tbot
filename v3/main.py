# main.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# وارد کردن کامپوننت‌های جدید
from backtesting import Backtest
from advanced_swing_strategy import AdvancedSwingStrategy
from data.data_fetcher import fetch_market_data

# کد اصلاح شده برای لاگ‌گیری با پشتیبانی از فارسی
import sys

log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('trading_bot.log', encoding='utf-8')
file_handler.setFormatter(log_format)
console_handler = logging.StreamHandler(sys.stdout)
try:
    console_handler.stream.reconfigure(encoding='utf-8')
except AttributeError:
    pass
console_handler.setFormatter(log_format)
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
logger = logging.getLogger(__name__)

def prepare_data_for_backtest(symbol='EURUSD', interval='H1', limit=1000, data_source="MT5"):
    """
    آماده‌سازی داده‌ها برای بک‌تست.
    """
    try:
        logger.info(f"در حال دریافت داده برای {symbol} با تایم‌فریم {interval} از {data_source}")
        data = fetch_market_data(symbol, interval, limit, data_source)
        
        if data.empty:
            logger.error(f"داده‌ای برای {symbol} دریافت نشد")
            return None
        
        # اطمینان از وجود ستون‌های مورد نیاز
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            logger.error("ستون‌های مورد نیاز (OHLCV) در داده‌ها وجود ندارد.")
            return None

        # کتابخانه backtesting.py انتظار ستون‌های با حرف بزرگ دارد
        data = data.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'open_time': 'timestamp'
        })
        
        # --- محاسبه RSI از اینجا حذف شد ---
        # استراتژی به صورت خودکار RSI را محاسبه می‌کند
        
        data.dropna(inplace=True)
        data.set_index('timestamp', inplace=True)
        
        logger.info(f"✅ داده‌های آماده شده: {len(data)} کندل")
        logger.info(f"📅 بازه زمانی: {data.index[0]} تا {data.index[-1]}")
        
        return data
        
    except Exception as e:
        logger.error(f"خطا در آماده‌سازی داده: {e}")
        return None

def run_complete_backtest():
    """
    اجرای کامل فرآیند بک‌تست با استفاده از کتابخانه backtesting.py
    """
    config = {
        'symbol': 'EURUSD',
        'interval': 'H1',
        'limit': 2000,
        'data_source': 'MT5',
        'initial_cash': 10000,
        'commission': 0.001,
    }
    
    # پارامترهای استراتژی را دیگر اینجا تعریف نمی‌کنیم
    # چون در خود فایل استراتژی مقادیر پیش‌فرض و بهتری داریم
    strategy_params = {} # خالی می‌گذاریم
    
    data = prepare_data_for_backtest(
        config['symbol'], 
        config['interval'], 
        config['limit'], 
        config['data_source']
    )
    
    if data is None:
        logger.error("❌ عدم امکان ادامه بک‌تست به دلیل عدم دریافت داده")
        return

    print("\n" + "="*50)
    print("🚀 شروع اجرای بک‌تست با backtesting.py...")
    print("="*50)
    
    bt = Backtest(
        data, 
        AdvancedSwingStrategy, 
        cash=config['initial_cash'],
        commission=config['commission'],
        trade_on_close=False,
        exclusive_orders=True
    )
    
    stats = bt.run(**strategy_params)
    
    print("\n📊 نتایج بک‌تست:")
    print(stats)
    
    print("\n📈 رسم نمودار نتایج...")
    bt.plot(filename=f"{config['symbol']}_backtest_chart.html", open_browser=False)
    logger.info(f"✅ نمودار نتایج در فایل '{config['symbol']}_backtest_chart.html' ذخیره شد.")

    trades_df = stats['_trades']
    equity_df = stats['_equity_curve']
    
    trades_df.to_csv(f"{config['symbol']}_trades.csv")
    equity_df.to_csv(f"{config['symbol']}_equity.csv")
    
    logger.info("✅ بک‌تست با موفقیت تکمیل شد")
    logger.info(f"📁 نتایج در فایل‌های '{config['symbol']}_trades.csv' و '{config['symbol']}_equity.csv' ذخیره شدند")

if __name__ == "__main__":
    run_complete_backtest()