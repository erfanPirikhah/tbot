# config/parameters.py - OPTIMIZED VERSION

# 🔥 پارامترهای بهینه‌شده برای سودآوری بالاتر - V5 با بهبودهای کامل
OPTIMIZED_PARAMS_V5 = {
    # Core RSI - بهبود یافته برای نرخ موفقیت بالاتر
    'rsi_period': 14,  # استاندارد و قابل اعتماد
    'rsi_oversold': 30,  # کاهش یافت تا فرصت‌های بیشتری ایجاد شود
    'rsi_overbought': 70,  # افزایش یافت تا فرصت‌های بیشتری ایجاد شود
    'rsi_entry_buffer': 3,  # کاهش یافت برای ورودهای دقیق‌تر

    # Risk Management - پویا و تطبیقی
    'risk_per_trade': 0.015,  # 1.5% - متعادل
    'stop_loss_atr_multiplier': 2.0,  # فاصله کافی برای تنفس
    'take_profit_ratio': 2.5,  # هدف سود 5% (2.5 * 2% SL)
    'min_position_size': 100,  # قابل دسترس برای حساب‌های کوچک
    'max_position_size_ratio': 0.4,  # افزایش یافت تا در شرایط مساعد بیشتر از سرمایه استفاده شود

    # Trade Control - منطقی
    'max_trades_per_100': 30,  # فرصت‌های بیشتر
    'min_candles_between': 5,  # فاصله منطقی
    'max_trade_duration': 100,  # زمان کافی برای رشد

    # Filters - فعال برای بهبود دقت
    'enable_trend_filter': True,  # 🔥 فعال شد - استفاده از فیلتر روند بهبود یافته
    'trend_strength_threshold': 0.4,  # آستانه مناسب برای تشخیص روند
    'enable_volume_filter': False,
    'enable_volatility_filter': True,  # 🔥 فعال شد - فیلتر نوسان
    'enable_short_trades': True,  # سود از هر دو جهت

    # Advanced - بهبود یافته
    'enable_trailing_stop': True,
    'trailing_activation_percent': 1.0,  # 🔥 فعال می‌شود در 1% سود
    'trailing_stop_atr_multiplier': 1.5,  # حفظ سود بیشتر
    'enable_partial_exit': True,
    'partial_exit_ratio': 0.5,  # 50% خروج جزئی
    'partial_exit_threshold': 1.5,  # 🔥 در 1.5% سود

    # Loss Control - متعادل
    'max_consecutive_losses': 4,  # تحمل بیشتر
    'pause_after_losses': 10,  # وقفه کوتاه‌تر
    'risk_reduction_after_loss': False,  # ثابت ماندن ریسک

    # Confirmations - ساده
    'require_rsi_confirmation': False,
    'require_price_confirmation': False,
    'confirmation_candles': 1,

    # Multi-Timeframe Analysis (MTF) - بهبود یافته
    'enable_mtf': True,
    'mtf_timeframes': ['H4', 'D1', 'H1'],   # افزایش تایم‌فریم‌ها
    'mtf_require_all': False,          # 🔥 تغییر: فقط تعدادی از تایم‌فریم‌ها نیاز به همسو بودن دارند
    'mtf_long_rsi_min': 40.0,          # 🔥 کاهش یافت - انعطاف‌پذیری بیشتر
    'mtf_short_rsi_max': 60.0,         # 🔥 افزایش یافت - انعطاف‌پذیری بیشتر
    'mtf_trend_ema_fast': 21,          # EMA سریع تایم‌فریم بالاتر
    'mtf_trend_ema_slow': 50,          # EMA کند تایم‌فریم بالاتر

    # Advanced Filters (new addition)
    'enable_advanced_filters': True,  # Enable advanced market regime and condition filters
    'advanced_filter_confidence_threshold': 0.7,  # Minimum confidence for market regime
    'market_strength_min_score': 3.0,  # Minimum market strength score (0-10)
    'support_resistance_check': True,  # Check price against support/resistance levels
    'divergence_check': True,  # Check for RSI-price divergence
    'volatility_band_check': True,  # Check price position in volatility bands
}

# Legacy mapping for backward compatibility
OPTIMIZED_PARAMS_V4 = OPTIMIZED_PARAMS_V5.copy()

# 🔥 پارامترهای محافظه‌کارانه (برای حساب‌های بزرگ)
CONSERVATIVE_PARAMS = {
    'rsi_period': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'rsi_entry_buffer': 3,
    
    'risk_per_trade': 0.01,  # 1% فقط
    'stop_loss_atr_multiplier': 2.5,
    'take_profit_ratio': 3.0,
    'min_position_size': 500,
    'max_position_size_ratio': 0.2,
    
    'max_trades_per_100': 15,
    'min_candles_between': 10,
    'max_trade_duration': 120,
    
    'enable_trend_filter': True,
    'trend_strength_threshold': 0.01,
    'enable_volume_filter': False,
    'enable_volatility_filter': True,
    'enable_short_trades': False,  # فقط LONG
    
    'enable_trailing_stop': True,
    'trailing_activation_percent': 2.0,
    'trailing_stop_atr_multiplier': 2.0,
    'enable_partial_exit': True,
    'partial_exit_ratio': 0.3,
    'partial_exit_threshold': 2.5,
    
    'max_consecutive_losses': 3,
    'pause_after_losses': 20,
    'risk_reduction_after_loss': True,
    
    'require_rsi_confirmation': True,
    'require_price_confirmation': True,
    'confirmation_candles': 2,

    # Advanced Filters (new addition)
    'enable_advanced_filters': True,  # Enable advanced market regime and condition filters
    'advanced_filter_confidence_threshold': 0.6,  # Conservative threshold
    'market_strength_min_score': 3.5,  # Slightly higher for conservative approach
    'support_resistance_check': True,  # Check price against support/resistance levels
    'divergence_check': True,  # Check for RSI-price divergence
    'volatility_band_check': True,  # Check price position in volatility bands
}

# 🔥 پارامترهای پرریسک (برای تریدرهای حرفه‌ای)
AGGRESSIVE_PARAMS = {
    'rsi_period': 11,
    'rsi_oversold': 40,  # ورود سریع‌تر
    'rsi_overbought': 60,
    'rsi_entry_buffer': 8,
    
    'risk_per_trade': 0.025,  # 2.5% ریسک
    'stop_loss_atr_multiplier': 1.5,
    'take_profit_ratio': 2.0,
    'min_position_size': 50,
    'max_position_size_ratio': 0.4,
    
    'max_trades_per_100': 50,
    'min_candles_between': 3,
    'max_trade_duration': 80,
    
    'enable_trend_filter': False,
    'trend_strength_threshold': 0.003,
    'enable_volume_filter': False,
    'enable_volatility_filter': False,
    'enable_short_trades': True,
    
    'enable_trailing_stop': True,
    'trailing_activation_percent': 0.5,
    'trailing_stop_atr_multiplier': 1.0,
    'enable_partial_exit': True,
    'partial_exit_ratio': 0.6,
    'partial_exit_threshold': 1.0,
    
    'max_consecutive_losses': 5,
    'pause_after_losses': 5,
    'risk_reduction_after_loss': False,
    
    'require_rsi_confirmation': False,
    'require_price_confirmation': False,
    'confirmation_candles': 1,

    # Advanced Filters (new addition)
    'enable_advanced_filters': True,  # Enable advanced market regime and condition filters
    'advanced_filter_confidence_threshold': 0.65,  # Moderate threshold for aggressive approach
    'market_strength_min_score': 2.5,  # Lower threshold for aggressive approach
    'support_resistance_check': True,  # Check price against support/resistance levels
    'divergence_check': True,  # Check for RSI-price divergence
    'volatility_band_check': True,  # Check price position in volatility bands
}

# 🎯 پارامترهای Ensemble برای M5 و M15
ENSEMBLE_SCALPING_M5 = {
    'strategy_class': 'EnsembleRsiStrategyV4',
    'rsi_period': 14,
    'rsi_oversold': 35,
    'rsi_overbought': 65,
    'rsi_entry_buffer': 6,

    'risk_per_trade': 0.02,
    'stop_loss_atr_multiplier': 1.4,
    'take_profit_ratio': 1.6,
    'min_position_size': 80,
    'max_position_size_ratio': 0.35,

    'max_trades_per_100': 80,
    'min_candles_between': 2,
    'max_trade_duration': 35,

    'enable_trailing_stop': True,
    'trailing_activation_percent': 0.5,
    'trailing_stop_atr_multiplier': 1.0,
    'enable_partial_exit': True,
    'partial_exit_ratio': 0.5,
    'partial_exit_threshold': 0.8,

    'enable_short_trades': True,

    'session_filter_enabled': True,
    'session_hours': [(7, 12), (13, 20)],
    'session_timezone_offset': 0,

    'bb_width_min': 0.001,
    'bb_width_max': 0.06,

    # Volatility adaptation for SL tightening avoidance
    'vol_sl_min_multiplier': 1.5,       # enforce minimum SL width
    'vol_sl_high_multiplier': 2.2,      # widen SL under high volatility
    'bb_width_vol_threshold': 0.015,    # BB width threshold to detect volatile regime

    # Advanced Filters (new addition)
    'enable_advanced_filters': True,  # Enable advanced market regime and condition filters
    'advanced_filter_confidence_threshold': 0.65,  # Moderate threshold for scalping
    'market_strength_min_score': 2.8,  # Appropriate for scalping approach
    'support_resistance_check': True,  # Check price against support/resistance levels
    'divergence_check': True,  # Check for RSI-price divergence
    'volatility_band_check': True,  # Check price position in volatility bands
}

ENSEMBLE_INTRADAY_M15 = {
    'strategy_class': 'EnsembleRsiStrategyV4',
    'rsi_period': 14,
    'rsi_oversold': 35,
    'rsi_overbought': 65,
    'rsi_entry_buffer': 5,

    'risk_per_trade': 0.015,
    'stop_loss_atr_multiplier': 1.6,
    'take_profit_ratio': 1.8,
    'min_position_size': 100,
    'max_position_size_ratio': 0.35,

    'max_trades_per_100': 50,
    'min_candles_between': 3,
    'max_trade_duration': 60,

    'enable_trailing_stop': True,
    'trailing_activation_percent': 0.6,
    'trailing_stop_atr_multiplier': 1.2,
    'enable_partial_exit': True,
    'partial_exit_ratio': 0.5,
    'partial_exit_threshold': 1.0,

    'enable_short_trades': True,

    'session_filter_enabled': True,
    'session_hours': [(7, 12), (13, 20)],
    'session_timezone_offset': 0,

    'bb_width_min': 0.001,
    'bb_width_max': 0.06,

    # Volatility adaptation for SL tightening avoidance
    'vol_sl_min_multiplier': 1.5,
    'vol_sl_high_multiplier': 2.2,
    'bb_width_vol_threshold': 0.015,

    # Advanced Filters (new addition)
    'enable_advanced_filters': True,  # Enable advanced market regime and condition filters
    'advanced_filter_confidence_threshold': 0.65,  # Moderate threshold for intraday
    'market_strength_min_score': 3.0,  # Appropriate for intraday approach
    'support_resistance_check': True,  # Check price against support/resistance levels
    'divergence_check': True,  # Check for RSI-price divergence
    'volatility_band_check': True,  # Check price position in volatility bands
}

# ✅ H1 profile tuned for higher win rate and realistic TP/SL
ENHANCED_INTRADAY_H1 = {
    'rsi_period': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'rsi_entry_buffer': 4,

    'risk_per_trade': 0.01,
    'stop_loss_atr_multiplier': 1.8,
    'take_profit_ratio': 2.0,
    'min_position_size': 200,
    'max_position_size_ratio': 0.25,

    'max_trades_per_100': 25,
    'min_candles_between': 3,
    'max_trade_duration': 80,

    'enable_trend_filter': True,
    'trend_strength_threshold': 0.005,
    'enable_volume_filter': False,
    'enable_volatility_filter': False,
    'enable_short_trades': True,

    'enable_trailing_stop': True,
    'trailing_activation_percent': 1.0,
    'trailing_stop_atr_multiplier': 1.2,
    'enable_partial_exit': True,
    'partial_exit_ratio': 0.5,
    'partial_exit_threshold': 1.0,

    # Disable strict MTF gating for H1 to avoid over-filtering entries
    'enable_mtf': False,

    # Advanced Filters (new addition)
    'enable_advanced_filters': True,  # Enable advanced market regime and condition filters
    'advanced_filter_confidence_threshold': 0.65,  # Moderate threshold for H1
    'market_strength_min_score': 3.0,  # Appropriate for H1 approach
    'support_resistance_check': True,  # Check price against support/resistance levels
    'divergence_check': True,  # Check for RSI-price divergence
    'volatility_band_check': True,  # Check price position in volatility bands
}

# تنظیمات بر اساس شرایط مختلف بازار
MARKET_CONDITION_PARAMS = {
    "TRENDING": {
        'rsi_oversold': 35,
        'rsi_overbought': 65,
        'risk_per_trade': 0.018,
        'stop_loss_atr_multiplier': 1.8,
        'enable_trend_filter': False,
        'enable_short_trades': True,

        # Advanced Filters (new addition)
        'enable_advanced_filters': True,  # Enable advanced market regime and condition filters
        'advanced_filter_confidence_threshold': 0.75,  # Higher threshold for trending markets
        'market_strength_min_score': 3.5,  # Higher threshold for trending markets
        'support_resistance_check': True,  # Check price against support/resistance levels
        'divergence_check': True,  # Check for RSI-price divergence
        'volatility_band_check': True,  # Check price position in volatility bands
    },
    "RANGING": {
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'risk_per_trade': 0.012,
        'stop_loss_atr_multiplier': 2.2,
        'enable_trend_filter': False,
        'enable_short_trades': True,

        # Advanced Filters (new addition)
        'enable_advanced_filters': True,  # Enable advanced market regime and condition filters
        'advanced_filter_confidence_threshold': 0.65,  # Moderate threshold for ranging markets
        'market_strength_min_score': 3.0,  # Moderate threshold for ranging markets
        'support_resistance_check': True,  # Check price against support/resistance levels
        'divergence_check': True,  # Check for RSI-price divergence
        'volatility_band_check': True,  # Check price position in volatility bands
    },
    "VOLATILE": {
        'rsi_oversold': 35,
        'rsi_overbought': 65,
        'risk_per_trade': 0.01,
        'stop_loss_atr_multiplier': 2.5,
        'enable_volatility_filter': False,  # 🔥 حتی در نوسان هم فعال
        'enable_short_trades': True,

        # Advanced Filters (new addition)
        'enable_advanced_filters': True,  # Enable advanced market regime and condition filters
        'advanced_filter_confidence_threshold': 0.6,  # Lower threshold for volatile markets
        'market_strength_min_score': 2.5,  # Lower threshold for volatile markets
        'support_resistance_check': True,  # Check price against support/resistance levels
        'divergence_check': True,  # Check for RSI-price divergence
        'volatility_band_check': True,  # Check price position in volatility bands
    },
    "SCALPING": {
        'rsi_period': 9,
        'rsi_oversold': 40,
        'rsi_overbought': 60,
        'rsi_entry_buffer': 10,
        'risk_per_trade': 0.02,
        'stop_loss_atr_multiplier': 1.2,
        'take_profit_ratio': 1.5,
        'max_trades_per_100': 80,
        'min_candles_between': 2,
        'max_trade_duration': 30,
        'trailing_activation_percent': 0.3,
        'partial_exit_threshold': 0.5,

        # Advanced Filters (new addition)
        'enable_advanced_filters': True,  # Enable advanced market regime and condition filters
        'advanced_filter_confidence_threshold': 0.6,  # Lower threshold for scalping
        'market_strength_min_score': 2.8,  # Appropriate for scalping approach
        'support_resistance_check': True,  # Check price against support/resistance levels
        'divergence_check': True,  # Check for RSI-price divergence
        'volatility_band_check': True,  # Check price position in volatility bands
    }
}

# 🔥 پارامترهای تست (برای آزمایش سریع)
TEST_PARAMS = {
    'rsi_period': 14,
    'rsi_oversold': 35,
    'rsi_overbought': 65,
    'rsi_entry_buffer': 5,
    'risk_per_trade': 0.015,
    'enable_trend_filter': False,
    'enable_volatility_filter': False,
    'enable_short_trades': True,
    'min_candles_between': 3,
    'max_trades_per_100': 40,
    'trailing_activation_percent': 1.0,
    'partial_exit_threshold': 1.5,

    # Advanced Filters (new addition)
    'enable_advanced_filters': True,  # Enable advanced market regime and condition filters
    'advanced_filter_confidence_threshold': 0.65,  # Moderate threshold for testing
    'market_strength_min_score': 3.0,  # Standard threshold for testing
    'support_resistance_check': True,  # Check price against support/resistance levels
    'divergence_check': True,  # Check for RSI-price divergence
    'volatility_band_check': True,  # Check price position in volatility bands
}

# 🎯 انتخاب خودکار بهترین پارامتر
def get_best_params_for_timeframe(timeframe: str) -> dict:
    """
    انتخاب بهترین پارامترها بر اساس تایم‌فریم
    """
    tf = (timeframe or '').upper()
    if tf in ['M1', 'M5']:
        return ENSEMBLE_SCALPING_M5.copy()
    elif tf in ['M15', 'M30']:
        return ENSEMBLE_INTRADAY_M15.copy()
    elif tf == 'H1':
        return ENHANCED_INTRADAY_H1.copy()
    elif tf == 'H4':
        return OPTIMIZED_PARAMS_V5.copy()  # Updated to use V5
    else:  # D1, W1
        return CONSERVATIVE_PARAMS.copy()

# 🎯 انتخاب خودکار بر اساس سرمایه
def get_params_for_capital(capital: float) -> dict:
    """
    انتخاب پارامترها بر اساس میزان سرمایه
    """
    if capital < 1000:
        params = AGGRESSIVE_PARAMS.copy()
        params['min_position_size'] = 50
        return params
    elif capital < 10000:
        return OPTIMIZED_PARAMS_V4.copy()
    else:
        return CONSERVATIVE_PARAMS.copy()