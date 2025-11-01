# config.py

# API Settings
CRYPTOCOMPARE_API_KEY = "5f197703c0eaed6946a976ec93311f53a8770d2d870f5739b4cad7bad1ac90f1"

# نمادهای CryptoCompare (موجود)
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

# نمادهای MetaTrader5 (جدید)
MT5_SYMBOL_MAP = {
    "طلا (XAUUSD)": "XAUUSD",
    "نقره (XAGUSD)": "XAGUSD",
    "یورو/دلار (EURUSD)": "EURUSD",
    "پوند/دلار (GBPUSD)": "GBPUSD",
    "دلار/ین (USDJPY)": "USDJPY",
    "دلار/فرانک (USDCHF)": "USDCHF",
    "دلار/دلار کانادا (USDCAD)": "USDCAD",
    "دلار استرالیا/دلار (AUDUSD)": "AUDUSD",
    "یورو/ین (EURJPY)": "EURJPY",
    "بیت‌کوین (BTCUSD)": "BTCUSD",
    "اتریوم (ETHUSD)": "ETHUSD",
    "نفت (XTIUSD)": "XTIUSD",
    "شاخص داوجونز (US30)": "US30",
    "ناسداک (NAS100)": "NAS100",
    "S&P 500 (SPX500)": "SPX500"
}

# مپ کلی همه نمادها
ALL_SYMBOL_MAP = {**CRYPTOCOMPARE_SYMBOL_MAP, **MT5_SYMBOL_MAP}

CRYPTOCOMPARE_INTERVAL_MAP = {
    "۱ ساعت": "1h",
    "۱ روز": "1d", 
    "۱ هفته": "1w"
}

# مپ تایم‌فریم‌های MT5
MT5_INTERVAL_MAP = {
    "۱ دقیقه": "M1",
    "۵ دقیقه": "M5", 
    "۱۵ دقیقه": "M15",
    "۳۰ دقیقه": "M30",
    "۱ ساعت": "H1",
    "۴ ساعت": "H4",
    "۱ روز": "D1",
    "۱ هفته": "W1"
}

# مپ تایم‌فریم کلی
ALL_INTERVAL_MAP = {**CRYPTOCOMPARE_INTERVAL_MAP, **MT5_INTERVAL_MAP}

# Default Settings
DEFAULT_SYMBOL = "طلا (XAUUSD)"  # تغییر به طلا به عنوان پیشفرض
DEFAULT_INTERVAL = "۱ ساعت"
DEFAULT_DATA_SOURCE = "MT5"  # MT5 یا CryptoCompare

# تنظیمات MT5
MT5_SETTINGS = {
    'server': None,  # اگر None باشد از پیشفرض متاتریدر استفاده می‌کند
    'login': None,   # شماره حساب
    'password': None,# رمز
    'timeout': 10000,
    'portable': False
}

# Indicator Settings - اضافه کردن مقادیر از دست رفته
RSI_PERIOD = 14
RSI_OVERSOLD = 30    # اضافه شده
RSI_OVERBOUGHT = 70  # اضافه شده

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