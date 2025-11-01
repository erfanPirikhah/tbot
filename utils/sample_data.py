# utils/sample_data.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def create_sample_data(symbol: str, periods: int = 100) -> pd.DataFrame:
    """ایجاد داده نمونه برای تست"""
    logger.info(f"ایجاد داده نمونه برای {symbol} با {periods} دوره")
    
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=periods),
        end=datetime.now(),
        freq='1H'
    )[-periods:]
    
    # قیمت پایه بر اساس نماد
    base_prices = {
        'XAUUSD': 1950.0, 'XAGUSD': 23.50, 'EURUSD': 1.0800, 'GBPUSD': 1.2600,
        'USDJPY': 150.00, 'USDCHF': 0.8800, 'USDCAD': 1.3500, 'AUDUSD': 0.6500,
        'EURJPY': 162.00, 'BTCUSD': 45000.0, 'ETHUSD': 2500.0, 'XTIUSD': 75.00,
        'US30': 33000.0, 'NAS100': 15000.0, 'SPX500': 4500.0,
        'BTC': 45000.0, 'ETH': 2500.0, 'BNB': 300.0, 'SOL': 100.0,
        'XRP': 0.6000, 'ADA': 0.4500, 'DOGE': 0.0800, 'TRX': 0.1000,
        'AVAX': 35.00, 'MATIC': 0.8000
    }
    
    base_price = base_prices.get(symbol, 100.0)
    volatility = base_price * 0.02  # 2% نوسان
    
    np.random.seed(42)  # برای نتایج ثابت
    
    # ایجاد داده‌های تصادفی اما واقعی‌تر
    returns = np.random.normal(0, 0.01, periods)  # بازده‌های تصادفی
    
    prices = [base_price]
    for i in range(1, periods):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(new_price)
    
    data = {
        'open_time': dates,
        'open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0.005, 0.002))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0.005, 0.002))) for p in prices],
        'close': prices,
        'volume': np.random.normal(1000, 200, periods)
    }
    
    df = pd.DataFrame(data)
    
    # اطمینان از مثبت بودن قیمت‌ها
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].abs()
    
    # مرتب‌سازی
    df = df.sort_values('open_time').reset_index(drop=True)
    
    logger.info(f"داده نمونه برای {symbol} ایجاد شد. آخرین قیمت: {df['close'].iloc[-1]:.2f}")
    return df