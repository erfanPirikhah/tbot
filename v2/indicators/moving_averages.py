# indicators/moving_averages.py

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

def calculate_moving_averages(data: pd.DataFrame, periods: list = [20, 50, 200]) -> pd.DataFrame:
    """
    محاسبه میانگین متحرک برای دوره‌های مختلف
    
    Args:
        data (pd.DataFrame): دیتافریم حاوی ستون 'close'
        periods (list): لیست دوره‌های زمانی برای محاسبه میانگین متحرک
        
    Returns:
        pd.DataFrame: دیتافریم با ستون‌های جدید میانگین متحرک
    """
    if 'close' not in data.columns:
        raise ValueError("دیتافریم ورودی باید ستون 'close' داشته باشد.")
    
    df = data.copy()
    
    for period in periods:
        ma_column = f'MA_{period}'
        df[ma_column] = df['close'].rolling(window=period, min_periods=1).mean()
    
    return df

def calculate_ema(data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    محاسبه میانگین متحرک نمایی (EMA)
    
    Args:
        data (pd.DataFrame): دیتافریم حاوی ستون 'close'
        period (int): دوره زمانی برای محاسبه EMA
        
    Returns:
        pd.DataFrame: دیتافریم با ستون جدید EMA
    """
    if 'close' not in data.columns:
        raise ValueError("دیتافریم ورودی باید ستون 'close' داشته باشد.")
    
    df = data.copy()
    df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    return df

def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
    """
    محاسبه باندهای بولینگر
    
    Args:
        data (pd.DataFrame): دیتافریم حاوی ستون 'close'
        period (int): دوره زمانی برای محاسبه
        std_dev (int): تعداد انحراف معیار
        
    Returns:
        pd.DataFrame: دیتافریم با ستون‌های بولینگر باند
    """
    if 'close' not in data.columns:
        raise ValueError("دیتافریم ورودی باید ستون 'close' داشته باشد.")
    
    df = data.copy()
    
    # میانگین متحرک
    df['BB_Middle'] = df['close'].rolling(window=period).mean()
    
    # انحراف معیار
    df['BB_Std'] = df['close'].rolling(window=period).std()
    
    # باندهای بالا و پایین
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * std_dev)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * std_dev)
    
    # موقعیت قیمت نسبت به باندها
    df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    return df

def get_trend_strength(data: pd.DataFrame) -> Dict[str, Any]:
    """
    تحلیل قدرت روند بر اساس میانگین‌های متحرک
    
    Args:
        data (pd.DataFrame): دیتافریم حاوی ستون‌های قیمت
        
    Returns:
        Dict: اطلاعات قدرت روند
    """
    if len(data) < 50:
        return {'trend': 'نامشخص', 'strength': 0, 'direction': 'neutral'}
    
    close = data['close']
    
    # میانگین‌های متحرک کوتاه‌مدت و بلندمدت
    ma_20 = close.rolling(20).mean().iloc[-1]
    ma_50 = close.rolling(50).mean().iloc[-1]
    current_price = close.iloc[-1]
    
    # تعیین روند
    if current_price > ma_20 > ma_50:
        trend = "صعودی"
        direction = "up"
        # محاسبه قدرت روند
        strength = min(100, ((current_price - ma_50) / ma_50) * 1000)
    elif current_price < ma_20 < ma_50:
        trend = "نزولی"
        direction = "down"
        strength = min(100, ((ma_50 - current_price) / ma_50) * 1000)
    else:
        trend = "خنثی"
        direction = "neutral"
        strength = 0
    
    return {
        'trend': trend,
        'strength': round(strength, 2),
        'direction': direction,
        'ma_20': round(ma_20, 2),
        'ma_50': round(ma_50, 2),
        'price_vs_ma20': round(((current_price - ma_20) / ma_20) * 100, 2)
    }

def calculate_ichimoku_cloud(data: pd.DataFrame) -> pd.DataFrame:
    """
    محاسبه ابر ایچیموکو
    """
    df = data.copy()
    
    # خط تبدیل (Tenkan-sen)
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    df['Ichimoku_Tenkan'] = (high_9 + low_9) / 2
    
    # خط پایه (Kijun-sen)
    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    df['Ichimoku_Kijun'] = (high_26 + low_26) / 2
    
    # خط پیشرو (Senkou Span A)
    df['Ichimoku_Senkou_A'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(26)
    
    # خط پیشرو (Senkou Span B)
    high_52 = df['high'].rolling(window=52).max()
    low_52 = df['low'].rolling(window=52).min()
    df['Ichimoku_Senkou_B'] = ((high_52 + low_52) / 2).shift(26)
    
    # خط تاخیر (Chikou Span)
    df['Ichimoku_Chikou'] = df['close'].shift(-26)
    
    return df