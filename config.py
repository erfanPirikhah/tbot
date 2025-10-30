# API Settings
CRYPTOCOMPARE_API_KEY = "5f197703c0eaed6946a976ec93311f53a8770d2d870f5739b4cad7bad1ac90f1"

CRYPTOCOMPARE_SYMBOL_MAP = {
    "بیت‌کوین (BTC)": "BTC",
    "اتریوم (ETH)": "ETH", 
    "بایننس کوین (BNB)": "BNB",
    "سولانا (SOL)": "SOL",
    "ریپل (XRP)": "XRP",
    "کاردانو (ADA)": "ADA",
    "دوج‌کوین (DOGE)": "DOGE",
    "ترون (TRX)": "TRX",
    "آوالانچ (AVAX)": "AVAX",
    "پالیگان (MATIC)": "MATIC"
}

CRYPTOCOMPARE_INTERVAL_MAP = {
    "۱ ساعت": "1h",
    "۱ روز": "1d",
    "۱ هفته": "1w"
}

# Default Settings
DEFAULT_SYMBOL = "بیت‌کوین (BTC)"
DEFAULT_INTERVAL = "۱ ساعت"

# Indicator Settings
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Improved Strategy Default Parameters
IMPROVED_STRATEGY_PARAMS = {
    'overbought': 70,
    'oversold': 30,
    'rsi_period': 14,
    'trend_ma_short': 20,
    'trend_ma_long': 50,
    'trend_threshold': 0.015,
    'min_conditions': 3,
    'volume_ma_period': 20,
    'volume_spike_threshold': 1.5,
    'risk_per_trade': 0.02,
    'stop_loss_atr_multiplier': 1.5,
    'take_profit_ratio': 2.5,
    'use_trailing_stop': True,
    'trailing_stop_activation': 1.0,
    'trailing_stop_distance': 0.5,
    'divergence_lookback': 14,
    'min_divergence_strength': 0.2,
    'max_trade_duration': 72
}