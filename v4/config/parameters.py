# config/parameters.py

# پارامترهای بهینه‌شده برای استراتژی RSI
OPTIMIZED_PARAMS_V4 = {
    # پارامترهای اصلی RSI
    'rsi_period': 11,
    'rsi_oversold': 28,
    'rsi_overbought': 72,
    'rsi_entry_buffer': 2,
    
    # مدیریت ریسک پیشرفته
    'risk_per_trade': 0.008,
    'stop_loss_atr_multiplier': 1.8,
    'take_profit_ratio': 2.2,
    'min_position_size': 800,
    'max_position_size_ratio': 0.25,
    
    # کنترل معاملات
    'max_trades_per_100': 20,
    'min_candles_between': 8,
    'max_trade_duration': 75,
    
    # فیلترهای پیشرفته
    'enable_trend_filter': True,
    'trend_strength_threshold': 0.008,
    'enable_volume_filter': False,
    'enable_volatility_filter': True,
    'enable_short_trades': True,
    
    # ویژگی‌های پیشرفته
    'enable_trailing_stop': True,
    'trailing_activation_percent': 0.4,
    'trailing_stop_atr_multiplier': 1.0,
    'enable_partial_exit': True,
    'partial_exit_ratio': 0.5,
    'partial_exit_threshold': 0.8,
    
    # کنترل ضرر
    'max_consecutive_losses': 3,
    'pause_after_losses': 20,
    'risk_reduction_after_loss': True,
    
    # فیلترهای زمانی
    'enable_time_filter': False,
    'trading_hours': ['00:00-23:59'],
    
    # تاییدیه‌های اضافی
    'require_rsi_confirmation': True,
    'require_price_confirmation': True,
    'confirmation_candles': 2
}

# تنظیمات برای شرایط مختلف بازار
MARKET_CONDITION_PARAMS = {
    "TRENDING": {
        'rsi_oversold': 32,
        'rsi_overbought': 68,
        'risk_per_trade': 0.01,
        'stop_loss_atr_multiplier': 1.5,
        'enable_trend_filter': True
    },
    "RANGING": {
        'rsi_oversold': 25,
        'rsi_overbought': 75,
        'risk_per_trade': 0.006,
        'stop_loss_atr_multiplier': 2.0,
        'enable_trend_filter': False
    },
    "VOLATILE": {
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'risk_per_trade': 0.005,
        'stop_loss_atr_multiplier': 2.2,
        'enable_volatility_filter': True
    }
}