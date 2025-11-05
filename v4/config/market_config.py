# config/market_config.py

# نگاشت نمادها
SYMBOL_MAPPING = {
    # CryptoCompare
    "CRYPTOCOMPARE": {
        "بیت‌کوین": "BTC",
        "اتریوم": "ETH", 
        "دوج‌کوین": "DOGE",
        "ریپل": "XRP",
        "کاردانو": "ADA",
        "لایت‌کوین": "LTC",
        "پولکادات": "DOT", 
        "چین‌لینک": "LINK",
        "بایننس‌کوین": "BNB",
        "سولانا": "SOL",
        "BTC": "BTC",
        "ETH": "ETH",
        "USDT": "USDT",
        "BNB": "BNB",
        "ADA": "ADA",
        "SOL": "SOL",
        "DOT": "DOT",
        "DOGE": "DOGE",
        "AVAX": "AVAX",
        "MATIC": "MATIC",
        "LINK": "LINK",
        "UNI": "UNI",
        "LTC": "LTC",
        "ATOM": "ATOM",
        "XRP": "XRP",
    },
    
    # MetaTrader 5
    "MT5": {
        "یورو/دلار": "EURUSD",
        "پوند/دلار": "GBPUSD", 
        "دلار/ین": "USDJPY",
        "دلار/فرانک": "USDCHF",
        "طلای دلار": "XAUUSD",
        "نفت": "USOIL",
        "نقره": "XAGUSD",
        "دلار استرالیا": "AUDUSD",
        "دلار کانادا": "USDCAD",
        "نیوزیلند دلار": "NZDUSD",
        "EURUSD": "EURUSD",
        "GBPUSD": "GBPUSD",
        "USDJPY": "USDJPY", 
        "XAUUSD": "XAUUSD",
        "USOIL": "USOIL"
    }
}

# نگاشت تایم‌فریم‌ها
TIMEFRAME_MAPPING = {
    "CRYPTOCOMPARE": {
        "۱ دقیقه": "1m",
        "۵ دقیقه": "5m",
        "۱۵ دقیقه": "15m", 
        "۳۰ دقیقه": "30m",
        "۱ ساعت": "1h",
        "۴ ساعت": "4h",
        "۱ روز": "1d",
        "۱ هفته": "1w",
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m", 
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
        "1w": "1w",
        "H1": "1h",
        "H4": "4h",
        "D1": "1d",
    },
    
    "MT5": {
        "۱ دقیقه": "M1",
        "۵ دقیقه": "M5",
        "۱۵ دقیقه": "M15",
        "۳۰ دقیقه": "M30",
        "۱ ساعت": "H1", 
        "۴ ساعت": "H4",
        "۱ روز": "D1",
        "۱ هفته": "W1",
        "M1": "M1",
        "M5": "M5",
        "M15": "M15",
        "M30": "M30",
        "H1": "H1",
        "H4": "H4",
        "D1": "D1", 
        "W1": "W1",
    }
}

# تنظیمات پیش‌فرض
DEFAULT_CONFIG = {
    'symbol': "EURUSD",
    'timeframe': "H1", 
    'data_source': "AUTO",
    'initial_capital': 10000,
    'commission': 0.0003,
    'slippage': 0.0001
}