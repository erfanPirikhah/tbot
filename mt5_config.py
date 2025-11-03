# config/mt5_config.py

# تنظیمات اتصال MT5
MT5_CONFIG = {
    "account": None,      # شماره حساب (None برای استفاده از پیشفرض)
    "password": None,     # رمز حساب
    "server": None,       # سرور (None برای استفاده از پیشفرض)
}

# تنظیمات معامله‌گری
TRADING_CONFIG = {
    "symbols": ["XAUUSD", "EURUSD", "GBPUSD"],
    "timeframe": "H1",
    "default_lot_size": 0.1,
    "max_risk_per_trade": 0.02,
    "check_interval_minutes": 60,
    "max_daily_trades": 3,
}

# تنظیمات استراتژی
STRATEGY_CONFIG = {
    "enable_short_trades": True,
    "use_adx_filter": True,
    "use_partial_exits": True,
    "use_break_even": True,
    "min_signal_score": 7.0,
    "avoid_ranging_markets": True,
}