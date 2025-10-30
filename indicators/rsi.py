# indicators/rsi.py

import pandas as pd
from ta.momentum import RSIIndicator

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    اندیکاتور RSI را به دیتافریم اضافه می‌کند.

    Args:
        data (pd.DataFrame): دیتافریم حاوی ستون 'close'.
        period (int): دوره زمانی برای محاسبه RSI.

    Returns:
        pd.DataFrame: دیتافریم با ستون جدید 'RSI'.
    """
    if 'close' not in data.columns:
        raise ValueError("دیتافریم ورودی باید ستون 'close' داشته باشد.")

    rsi_indicator = RSIIndicator(close=data['close'], window=period)
    data['RSI'] = rsi_indicator.rsi()
    return data