# utils/plot_chart.py

import pandas as pd
import matplotlib.pyplot as plt
from config import RSI_OVERSOLD, RSI_OVERBOUGHT

def plot_price_and_rsi(data: pd.DataFrame, symbol: str):
    """
    نمودار قیمت و اندیکاتور RSI را رسم می‌کند.

    Args:
        data (pd.DataFrame): دیتافریم حاوی داده‌های قیمت و RSI.
        symbol (str): نماد برای نمایش در عنوان نمودار.
    """
    if data.empty:
        print("داده‌ای برای رسم نمودار وجود ندارد.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f'نمودار قیمت و RSI برای {symbol}', fontsize=16)

    # نمودار قیمت
    ax1.plot(data['open_time'], data['close'], label='Price (Close)', color='blue')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend()

    # نمودار RSI
    ax2.plot(data['open_time'], data['RSI'], label='RSI', color='purple')
    ax2.axhline(RSI_OVERBOUGHT, color='red', linestyle='--', label=f'Overbought ({RSI_OVERBOUGHT})')
    ax2.axhline(RSI_OVERSOLD, color='green', linestyle='--', label=f'Oversold ({RSI_OVERSOLD})')
    ax2.set_ylabel('RSI')
    ax2.set_xlabel('Time')
    ax2.grid(True)
    ax2.legend()
    ax2.set_ylim(0, 100)

    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()