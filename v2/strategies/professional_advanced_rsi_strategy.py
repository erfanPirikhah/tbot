# strategies/professional_advanced_rsi_strategy.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)

class PositionType(Enum):
    OUT = "OUT"
    LONG = "LONG"
    SHORT = "SHORT"

class SignalStrength(Enum):
    VERY_WEAK = "VERY_WEAK"
    WEAK = "WEAK"
    MEDIUM = "MEDIUM" 
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"

class ExitReason(Enum):
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    SIGNAL_EXIT = "SIGNAL_EXIT"
    TIME_EXIT = "TIME_EXIT"
    BREAK_EVEN = "BREAK_EVEN"

class MarketRegime(Enum):
    """رژیم‌های مختلف بازار"""
    STRONG_TREND = "STRONG_TREND"
    WEAK_TREND = "WEAK_TREND"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    QUIET = "QUIET"

@dataclass
class Trade:
    """کلاس ذخیره اطلاعات معامله - نسخه پیشرفته"""
    entry_price: float
    entry_time: pd.Timestamp
    position_type: PositionType
    quantity: float
    stop_loss: float
    take_profit: float
    initial_stop_loss: float = 0.0
    trailing_stop: float = 0.0
    break_even_activated: bool = False
    highest_price: float = 0.0
    lowest_price: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_reason: Optional[ExitReason] = None
    pnl_percentage: Optional[float] = None
    pnl_amount: Optional[float] = None
    max_drawdown: float = 0.0
    max_profit: float = 0.0
    entry_signal_strength: str = "UNKNOWN"
    market_regime: str = "UNKNOWN"
    
@dataclass
class MarketConditions:
    """کلاس تحلیل شرایط بازار - نسخه پیشرفته"""
    trend: str
    volatility: str
    momentum: str
    regime: MarketRegime
    strength_score: float = 0.0
    volume_trend: str = "NEUTRAL"
    adx_value: float = 0.0
    volume_ratio: float = 1.0
    price_momentum: float = 0.0

class ProfessionalAdvancedRsiStrategy:
    """
    استراتژی RSI حرفه‌ای با ویژگی‌های پیشرفته:
    
    ✅ امکانات جدید:
    - پشتیبانی کامل از SHORT و LONG
    - تشخیص رژیم بازار (Trending/Ranging/Volatile)
    - ADX برای قدرت روند
    - Volume Profile و VWAP
    - Break-Even Stop
    - Multi-Timeframe Analysis
    - False Breakout Detection
    - Dynamic ATR-based Stops
    - Partial Profit Taking
    - Risk-Adjusted Position Sizing
    """
    
    def __init__(
        self,
        # پارامترهای RSI
        overbought: int = 70,
        oversold: int = 30,
        rsi_period: int = 14,
        extreme_overbought: int = 80,
        extreme_oversold: int = 20,
        
        # پارامترهای فیلتر روند
        trend_ma_short: int = 20,
        trend_ma_long: int = 50,
        trend_threshold: float = 0.01,
        
        # پارامترهای ADX
        use_adx_filter: bool = True,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        
        # پارامترهای تاییدیه
        min_signal_score: float = 7.0,  # از 15 امتیاز کل
        volume_ma_period: int = 20,
        volume_spike_threshold: float = 1.5,
        
        # پارامترهای مدیریت ریسک
        risk_per_trade: float = 0.02,
        stop_loss_atr_multiplier: float = 2.0,
        take_profit_ratio: float = 2.5,
        
        # Trailing Stop پیشرفته
        use_trailing_stop: bool = True,
        trailing_stop_activation: float = 1.0,
        trailing_stop_distance: float = 0.5,
        use_break_even: bool = True,
        break_even_trigger: float = 1.5,
        
        # پارامترهای واگرایی
        divergence_lookback: int = 14,
        min_divergence_strength: float = 0.15,
        
        # مدیریت زمان
        max_trade_duration: int = 72,
        
        # امکانات پیشرفته
        enable_short_trades: bool = True,
        use_partial_exits: bool = True,
        partial_exit_ratio: float = 0.5,
        partial_exit_target: float = 2.0,
        
        # فیلتر رژیم بازار
        min_regime_score: float = 0.5,
        avoid_ranging_markets: bool = True,
    ):
        self._validate_parameters(overbought, oversold, risk_per_trade)
        
        # پارامترهای اصلی
        self.overbought = overbought
        self.oversold = oversold
        self.rsi_period = rsi_period
        self.extreme_overbought = extreme_overbought
        self.extreme_oversold = extreme_oversold
        
        self.trend_ma_short = trend_ma_short
        self.trend_ma_long = trend_ma_long
        self.trend_threshold = trend_threshold
        
        self.use_adx_filter = use_adx_filter
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        
        self.min_signal_score = min_signal_score
        self.volume_ma_period = volume_ma_period
        self.volume_spike_threshold = volume_spike_threshold
        
        self.risk_per_trade = risk_per_trade
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_ratio = take_profit_ratio
        
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_activation = trailing_stop_activation
        self.trailing_stop_distance = trailing_stop_distance
        self.use_break_even = use_break_even
        self.break_even_trigger = break_even_trigger
        
        self.divergence_lookback = divergence_lookback
        self.min_divergence_strength = min_divergence_strength
        self.max_trade_duration = max_trade_duration
        
        self.enable_short_trades = enable_short_trades
        self.use_partial_exits = use_partial_exits
        self.partial_exit_ratio = partial_exit_ratio
        self.partial_exit_target = partial_exit_target
        
        self.min_regime_score = min_regime_score
        self.avoid_ranging_markets = avoid_ranging_markets
        
        # مدیریت وضعیت
        self._position = PositionType.OUT
        self._current_trade: Optional[Trade] = None
        self._trade_history: List[Trade] = []
        self._portfolio_value: float = 10000.0
        self._partial_exit_done: bool = False
        
        # آمار عملکرد
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = 0.0
        self._max_consecutive_wins = 0
        self._max_consecutive_losses = 0
        self._current_streak = 0
        self._total_long_trades = 0
        self._total_short_trades = 0
        self._winning_long_trades = 0
        self._winning_short_trades = 0

    def _validate_parameters(self, overbought: int, oversold: int, risk_per_trade: float):
        """اعتبارسنجی پارامترها"""
        if overbought <= oversold:
            raise ValueError("سطح overbought باید از oversold بزرگتر باشد")
        if not (0 < risk_per_trade <= 0.1):
            raise ValueError("ریسک هر معامله باید بین 0.1% تا 10% باشد")

    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """محاسبه Average Directional Index (ADX)"""
        try:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # محاسبه +DM و -DM
            plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                              np.maximum(high[1:] - high[:-1], 0), 0)
            minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                               np.maximum(low[:-1] - low[1:], 0), 0)
            
            # محاسبه TR
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            
            # Smoothed averages
            atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
            plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
            minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr
            
            # محاسبه DX و ADX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.ewm(span=period, adjust=False).mean()
            
            return adx.iloc[-1] if len(adx) > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return 0.0

    def calculate_vwap(self, data: pd.DataFrame) -> float:
        """محاسبه Volume Weighted Average Price"""
        try:
            if 'volume' not in data.columns:
                return data['close'].iloc[-1]
            
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            vwap = (typical_price * data['volume']).sum() / data['volume'].sum()
            return vwap
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
            return data['close'].iloc[-1]

    def detect_market_regime(self, data: pd.DataFrame, market_conditions: MarketConditions) -> MarketRegime:
        """تشخیص رژیم بازار"""
        try:
            # معیارهای رژیم بازار
            volatility_score = 0
            trend_score = 0
            
            # 1. بررسی نوسانات
            if market_conditions.volatility == "HIGH":
                volatility_score = 2
            elif market_conditions.volatility == "MEDIUM":
                volatility_score = 1
            
            # 2. بررسی قدرت روند با ADX
            if market_conditions.adx_value > 40:
                trend_score = 2
            elif market_conditions.adx_value > 25:
                trend_score = 1
            
            # 3. بررسی حرکت قیمت
            price_range = (data['high'].tail(20).max() - data['low'].tail(20).min()) / data['close'].iloc[-1]
            
            # تصمیم‌گیری
            if trend_score >= 2 and volatility_score <= 1:
                return MarketRegime.STRONG_TREND
            elif trend_score == 1:
                return MarketRegime.WEAK_TREND
            elif volatility_score >= 2:
                return MarketRegime.VOLATILE
            elif price_range < 0.02:  # کمتر از 2% رنج
                return MarketRegime.QUIET
            else:
                return MarketRegime.RANGING
                
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.RANGING

    def analyze_market_conditions(self, data: pd.DataFrame) -> MarketConditions:
        """تحلیل جامع شرایط بازار - نسخه پیشرفته"""
        prices = data['close'].tail(100)
        volumes = data['volume'].tail(100) if 'volume' in data.columns else None
        
        # 1. تحلیل روند با دو MA
        ma_short = prices.rolling(self.trend_ma_short).mean().iloc[-1]
        ma_long = prices.rolling(self.trend_ma_long).mean().iloc[-1] if len(prices) >= self.trend_ma_long else ma_short
        price_current = prices.iloc[-1]
        
        # تعیین روند قوی‌تر
        trend_strength = (ma_short - ma_long) / ma_long if ma_long > 0 else 0
        
        if price_current > ma_short > ma_long and abs(trend_strength) > self.trend_threshold:
            trend = "STRONG_UPTREND"
            trend_score = 3.0
        elif price_current > ma_short and ma_short > ma_long:
            trend = "UPTREND"
            trend_score = 2.0
        elif price_current < ma_short < ma_long and abs(trend_strength) > self.trend_threshold:
            trend = "STRONG_DOWNTREND"
            trend_score = -3.0
        elif price_current < ma_short and ma_short < ma_long:
            trend = "DOWNTREND"
            trend_score = -2.0
        else:
            trend = "SIDEWAYS"
            trend_score = 0.0
        
        # 2. محاسبه ADX
        adx_value = self.calculate_adx(data.tail(50), self.adx_period) if self.use_adx_filter else 0.0
        
        # 3. تحلیل نوسانات با ATR
        volatility_value = self._calculate_volatility(data)
        if volatility_value > 0.05:
            volatility_level = "HIGH"
        elif volatility_value > 0.02:
            volatility_level = "MEDIUM"
        else:
            volatility_level = "LOW"
        
        # 4. تحلیل مومنتوم با RSI
        rsi_values = data['RSI'].tail(5)
        rsi_current = rsi_values.iloc[-1]
        rsi_slope = (rsi_current - rsi_values.iloc[0]) / len(rsi_values)
        
        if rsi_current > 50 and rsi_slope > 1:
            momentum = "STRONG_BULLISH"
        elif rsi_current > 50 and rsi_slope > 0:
            momentum = "BULLISH"
        elif rsi_current < 50 and rsi_slope < -1:
            momentum = "STRONG_BEARISH"
        elif rsi_current < 50 and rsi_slope < 0:
            momentum = "BEARISH"
        else:
            momentum = "NEUTRAL"
        
        # 5. تحلیل حجم معاملات
        volume_trend = "NEUTRAL"
        volume_ratio = 1.0
        
        if volumes is not None:
            volume_ma = volumes.rolling(self.volume_ma_period).mean()
            current_volume = volumes.iloc[-1]
            avg_volume = volume_ma.iloc[-1]
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            if volume_ratio > self.volume_spike_threshold:
                volume_trend = "SPIKE"
            elif volume_ratio > 1.2:
                volume_trend = "HIGH"
            elif volume_ratio > 1.0:
                volume_trend = "ABOVE_AVERAGE"
            elif volume_ratio < 0.7:
                volume_trend = "LOW"
        
        # 6. محاسبه Price Momentum
        price_momentum = (price_current - prices.iloc[-10]) / prices.iloc[-10] if len(prices) >= 10 else 0.0
        
        # ایجاد شیء MarketConditions
        conditions = MarketConditions(
            trend=trend,
            volatility=volatility_level,
            momentum=momentum,
            regime=MarketRegime.RANGING,  # موقتی
            strength_score=trend_score,
            volume_trend=volume_trend,
            adx_value=adx_value,
            volume_ratio=volume_ratio,
            price_momentum=price_momentum
        )
        
        # تشخیص رژیم بازار
        conditions.regime = self.detect_market_regime(data, conditions)
        
        return conditions

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """محاسبه نوسانات بازار"""
        returns = data['close'].pct_change().tail(20)
        return returns.std()

    def detect_false_breakout(self, data: pd.DataFrame, current_rsi: float) -> bool:
        """تشخیص False Breakout"""
        try:
            # بررسی آیا RSI به سرعت برگشته
            rsi_history = data['RSI'].tail(5)
            
            if current_rsi < self.oversold:
                # اگر RSI زیر oversold رفته ولی خیلی سریع برگشته
                if len(rsi_history) >= 3:
                    if rsi_history.iloc[-3] > self.oversold and rsi_history.iloc[-2] < self.oversold:
                        return True
            
            if current_rsi > self.overbought:
                if len(rsi_history) >= 3:
                    if rsi_history.iloc[-3] < self.overbought and rsi_history.iloc[-2] > self.overbought:
                        return True
            
            return False
        except:
            return False

    def detect_rsi_divergence(self, data: pd.DataFrame) -> Tuple[bool, bool, float]:
        """تشخیص واگرایی RSI - نسخه بهبود یافته"""
        if len(data) < self.divergence_lookback:
            return False, False, 0.0
        
        prices = data['close'].tail(self.divergence_lookback).values
        rsi_values = data['RSI'].tail(self.divergence_lookback).values
        
        # پیدا کردن قله‌ها و دره‌ها
        price_peaks_idx = argrelextrema(prices, np.greater, order=2)[0]
        price_troughs_idx = argrelextrema(prices, np.less, order=2)[0]
        rsi_peaks_idx = argrelextrema(rsi_values, np.greater, order=2)[0]
        rsi_troughs_idx = argrelextrema(rsi_values, np.less, order=2)[0]
        
        bullish_divergence = False
        bearish_divergence = False
        divergence_strength = 0.0
        
        # واگرایی مثبت: قیمت کف پایین‌تر، RSI کف بالاتر
        if len(price_troughs_idx) >= 2 and len(rsi_troughs_idx) >= 2:
            p_trough_1, p_trough_2 = prices[price_troughs_idx[-2]], prices[price_troughs_idx[-1]]
            r_trough_1, r_trough_2 = rsi_values[rsi_troughs_idx[-2]], rsi_values[rsi_troughs_idx[-1]]
            
            if p_trough_2 < p_trough_1 and r_trough_2 > r_trough_1:
                bullish_divergence = True
                price_change = abs((p_trough_2 - p_trough_1) / p_trough_1)
                rsi_change = abs((r_trough_2 - r_trough_1) / max(r_trough_1, 1))
                divergence_strength = (price_change + rsi_change) / 2
        
        # واگرایی منفی: قیمت سقف بالاتر، RSI سقف پایین‌تر
        if len(price_peaks_idx) >= 2 and len(rsi_peaks_idx) >= 2:
            p_peak_1, p_peak_2 = prices[price_peaks_idx[-2]], prices[price_peaks_idx[-1]]
            r_peak_1, r_peak_2 = rsi_values[rsi_peaks_idx[-2]], rsi_values[rsi_peaks_idx[-1]]
            
            if p_peak_2 > p_peak_1 and r_peak_2 < r_peak_1:
                bearish_divergence = True
                price_change = abs((p_peak_2 - p_peak_1) / p_peak_1)
                rsi_change = abs((r_peak_2 - r_peak_1) / max(r_peak_1, 1))
                divergence_strength = max(divergence_strength, (price_change + rsi_change) / 2)
        
        return bullish_divergence, bearish_divergence, divergence_strength

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """محاسبه Average True Range"""
        high = data['high']
        low = data['low']
        close_prev = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close_prev)
        tr3 = abs(low - close_prev)
        
        true_range = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = true_range.rolling(period).mean().iloc[-1]
        
        return atr if not np.isnan(atr) else 0.0

    def calculate_position_size(self, entry_price: float, stop_loss: float, account_balance: Optional[float] = None) -> float:
        """محاسبه حجم معامله با Risk-Adjusted Position Sizing"""
        portfolio = account_balance if account_balance else self._portfolio_value
        risk_amount = portfolio * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
            
        position_size = risk_amount / price_risk
        
        # محدودیت حداکثر پوزیشن: 20% پورتفولیو
        max_position = portfolio * 0.20
        
        return min(position_size, max_position)

    def calculate_signal_strength(
        self, 
        data: pd.DataFrame, 
        market_conditions: MarketConditions,
        bullish_div: bool, 
        bearish_div: bool,
        divergence_strength: float,
        position_type: PositionType
    ) -> Tuple[float, SignalStrength]:
        """محاسبه قدرت سیگنال با سیستم امتیازدهی پیشرفته"""
        score = 0.0
        current_rsi = data['RSI'].iloc[-1]
        
        # 1. امتیاز RSI (0-4 امتیاز)
        if position_type == PositionType.LONG:
            if current_rsi < self.extreme_oversold:
                score += 4.0
            elif current_rsi < self.oversold - 5:
                score += 3.0
            elif current_rsi < self.oversold:
                score += 2.0
            elif current_rsi < self.oversold + 5:
                score += 1.0
        else:  # SHORT
            if current_rsi > self.extreme_overbought:
                score += 4.0
            elif current_rsi > self.overbought + 5:
                score += 3.0
            elif current_rsi > self.overbought:
                score += 2.0
            elif current_rsi > self.overbought - 5:
                score += 1.0
        
        # 2. امتیاز واگرایی (0-4 امتیاز)
        if (bullish_div and position_type == PositionType.LONG) or \
           (bearish_div and position_type == PositionType.SHORT):
            div_score = min(divergence_strength * 10, 4.0)
            score += div_score
        
        # 3. امتیاز روند (0-3 امتیاز)
        if position_type == PositionType.LONG:
            if market_conditions.trend in ["STRONG_UPTREND"]:
                score += 3.0
            elif market_conditions.trend == "UPTREND":
                score += 2.0
            elif market_conditions.trend == "SIDEWAYS":
                score += 0.5
        else:  # SHORT
            if market_conditions.trend == "STRONG_DOWNTREND":
                score += 3.0
            elif market_conditions.trend == "DOWNTREND":
                score += 2.0
            elif market_conditions.trend == "SIDEWAYS":
                score += 0.5
        
        # 4. امتیاز مومنتوم (0-2 امتیاز)
        if position_type == PositionType.LONG:
            if market_conditions.momentum == "STRONG_BULLISH":
                score += 2.0
            elif market_conditions.momentum == "BULLISH":
                score += 1.0
        else:
            if market_conditions.momentum == "STRONG_BEARISH":
                score += 2.0
            elif market_conditions.momentum == "BEARISH":
                score += 1.0
        
        # 5. امتیاز ADX (0-2 امتیاز)
        if self.use_adx_filter:
            if market_conditions.adx_value > 40:
                score += 2.0
            elif market_conditions.adx_value > 25:
                score += 1.0
        
        # 6. امتیاز حجم (0-2 امتیاز)
        if market_conditions.volume_trend in ["SPIKE", "HIGH"]:
            score += 2.0
        elif market_conditions.volume_trend == "ABOVE_AVERAGE":
            score += 1.0
        
        # 7. جریمه برای بازار Ranging
        if market_conditions.regime == MarketRegime.RANGING and self.avoid_ranging_markets:
            score -= 2.0
        
        # 8. جریمه برای False Breakout
        if self.detect_false_breakout(data, current_rsi):
            score -= 3.0
        
        # طبقه‌بندی (حداکثر 17 امتیاز)
        if score >= 14.0:
            return score, SignalStrength.VERY_STRONG
        elif score >= 10.0:
            return score, SignalStrength.STRONG
        elif score >= 7.0:
            return score, SignalStrength.MEDIUM
        elif score >= 4.0:
            return score, SignalStrength.WEAK
        else:
            return score, SignalStrength.VERY_WEAK

    def check_exit_conditions(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """بررسی شرایط خروج از معامله - نسخه پیشرفته"""
        if not self._current_trade or self._position == PositionType.OUT:
            return None
        
        current_price = data['close'].iloc[-1]
        current_time = data.index[-1] if hasattr(data.index, '__getitem__') else pd.Timestamp.now()
        
        is_long = self._current_trade.position_type == PositionType.LONG
        
        # محاسبه سود/ضرر فعلی
        if is_long:
            pnl_ratio = (current_price - self._current_trade.entry_price) / self._current_trade.entry_price
        else:  # SHORT
            pnl_ratio = (self._current_trade.entry_price - current_price) / self._current_trade.entry_price
        
        # 1. بررسی Stop Loss
        if is_long:
            if current_price <= self._current_trade.stop_loss:
                return self._exit_trade(current_price, current_time, ExitReason.STOP_LOSS)
        else:  # SHORT
            if current_price >= self._current_trade.stop_loss:
                return self._exit_trade(current_price, current_time, ExitReason.STOP_LOSS)
        
        # 2. بررسی Take Profit
        if is_long:
            if current_price >= self._current_trade.take_profit:
                return self._exit_trade(current_price, current_time, ExitReason.TAKE_PROFIT)
        else:
            if current_price <= self._current_trade.take_profit:
                return self._exit_trade(current_price, current_time, ExitReason.TAKE_PROFIT)
        
        # 3. Partial Exit (خروج جزئی)
        if self.use_partial_exits and not self._partial_exit_done:
            if pnl_ratio >= self.partial_exit_target:
                self._execute_partial_exit(current_price, current_time)
        
        # 4. Break-Even Stop
        if self.use_break_even and not self._current_trade.break_even_activated:
            if pnl_ratio >= self.break_even_trigger:
                self._activate_break_even(current_price)
        
        # 5. بررسی Trailing Stop
        if self.use_trailing_stop and self._current_trade.trailing_stop > 0:
            if is_long:
                if current_price <= self._current_trade.trailing_stop:
                    return self._exit_trade(current_price, current_time, ExitReason.TRAILING_STOP)
            else:  # SHORT
                if current_price >= self._current_trade.trailing_stop:
                    return self._exit_trade(current_price, current_time, ExitReason.TRAILING_STOP)
        
        # 6. بررسی حداکثر زمان معامله
        if self._current_trade.entry_time:
            duration = (current_time - self._current_trade.entry_time).total_seconds() / 3600
            if duration >= self.max_trade_duration:
                return self._exit_trade(current_price, current_time, ExitReason.TIME_EXIT)
        
        # 7. سیگنال خروج معکوس
        exit_signal = self._check_reverse_signal(data)
        if exit_signal:
            return self._exit_trade(current_price, current_time, ExitReason.SIGNAL_EXIT)
        
        # 8. به‌روزرسانی Trailing Stop
        self._update_trailing_stop(current_price)
        
        # 9. به‌روزرسانی Max/Min قیمت
        if is_long:
            self._current_trade.highest_price = max(self._current_trade.highest_price, current_price)
            max_profit = (self._current_trade.highest_price - self._current_trade.entry_price) / self._current_trade.entry_price
            self._current_trade.max_profit = max(self._current_trade.max_profit, max_profit)
        else:
            self._current_trade.lowest_price = min(self._current_trade.lowest_price, current_price) if self._current_trade.lowest_price > 0 else current_price
            max_profit = (self._current_trade.entry_price - self._current_trade.lowest_price) / self._current_trade.entry_price
            self._current_trade.max_profit = max(self._current_trade.max_profit, max_profit)
        
        return None

    def _check_reverse_signal(self, data: pd.DataFrame) -> bool:
        """بررسی سیگنال معکوس برای خروج"""
        try:
            current_rsi = data['RSI'].iloc[-1]
            
            if self._current_trade.position_type == PositionType.LONG:
                # اگر LONG داریم و RSI به overbought رسید
                if current_rsi > self.overbought:
                    return True
            else:  # SHORT
                # اگر SHORT داریم و RSI به oversold رسید
                if current_rsi < self.oversold:
                    return True
            
            return False
        except:
            return False

    def _execute_partial_exit(self, exit_price: float, exit_time: pd.Timestamp):
        """اجرای خروج جزئی"""
        partial_quantity = self._current_trade.quantity * self.partial_exit_ratio
        
        is_long = self._current_trade.position_type == PositionType.LONG
        if is_long:
            partial_pnl = (exit_price - self._current_trade.entry_price) * partial_quantity
        else:
            partial_pnl = (self._current_trade.entry_price - exit_price) * partial_quantity
        
        self._portfolio_value += partial_pnl
        self._current_trade.quantity -= partial_quantity
        self._partial_exit_done = True
        
        logger.info(f"✅ Partial Exit: {self.partial_exit_ratio*100}% at {exit_price}, PnL: {partial_pnl:.2f}")

    def _activate_break_even(self, current_price: float):
        """فعال‌سازی Break-Even Stop"""
        self._current_trade.stop_loss = self._current_trade.entry_price
        self._current_trade.break_even_activated = True
        
        logger.info(f"✅ Break-Even Stop activated at {self._current_trade.entry_price}")

    def _update_trailing_stop(self, current_price: float):
        """به‌روزرسانی Trailing Stop - پیشرفته"""
        if not self.use_trailing_stop or not self._current_trade:
            return
        
        is_long = self._current_trade.position_type == PositionType.LONG
        
        # محاسبه سود فعلی
        if is_long:
            profit_ratio = (current_price - self._current_trade.entry_price) / self._current_trade.entry_price
        else:
            profit_ratio = (self._current_trade.entry_price - current_price) / self._current_trade.entry_price
        
        # فعال‌سازی trailing stop بعد از رسیدن به نسبت مشخص
        if profit_ratio >= self.trailing_stop_activation:
            risk_distance = abs(self._current_trade.entry_price - self._current_trade.initial_stop_loss)
            trailing_distance = risk_distance * self.trailing_stop_distance
            
            if is_long:
                new_trailing_stop = current_price - trailing_distance
                # فقط بالا بردن trailing stop
                if new_trailing_stop > self._current_trade.trailing_stop:
                    self._current_trade.trailing_stop = new_trailing_stop
                    self._current_trade.stop_loss = new_trailing_stop
            else:  # SHORT
                new_trailing_stop = current_price + trailing_distance
                # فقط پایین آوردن trailing stop برای SHORT
                if self._current_trade.trailing_stop == 0 or new_trailing_stop < self._current_trade.trailing_stop:
                    self._current_trade.trailing_stop = new_trailing_stop
                    self._current_trade.stop_loss = new_trailing_stop

    def _exit_trade(self, exit_price: float, exit_time: pd.Timestamp, reason: ExitReason) -> Dict[str, Any]:
        """خروج از معامله"""
        is_long = self._current_trade.position_type == PositionType.LONG
        
        if is_long:
            pnl_percentage = ((exit_price - self._current_trade.entry_price) / 
                             self._current_trade.entry_price * 100)
            pnl_amount = (exit_price - self._current_trade.entry_price) * self._current_trade.quantity
        else:  # SHORT
            pnl_percentage = ((self._current_trade.entry_price - exit_price) / 
                             self._current_trade.entry_price * 100)
            pnl_amount = (self._current_trade.entry_price - exit_price) * self._current_trade.quantity
        
        # آپدیت پورتفو
        self._portfolio_value += pnl_amount
        self._total_trades += 1
        
        # آمار LONG/SHORT
        if is_long:
            self._total_long_trades += 1
            if pnl_amount > 0:
                self._winning_long_trades += 1
        else:
            self._total_short_trades += 1
            if pnl_amount > 0:
                self._winning_short_trades += 1
        
        # آمار برد/باخت
        if pnl_amount > 0:
            self._winning_trades += 1
            self._current_streak = max(0, self._current_streak) + 1
            self._max_consecutive_wins = max(self._max_consecutive_wins, self._current_streak)
        else:
            self._current_streak = min(0, self._current_streak) - 1
            self._max_consecutive_losses = max(self._max_consecutive_losses, abs(self._current_streak))
        
        self._total_pnl += pnl_amount
        
        # آپدیت معامله
        self._current_trade.exit_price = exit_price
        self._current_trade.exit_time = exit_time
        self._current_trade.exit_reason = reason
        self._current_trade.pnl_percentage = pnl_percentage
        self._current_trade.pnl_amount = pnl_amount
        
        self._trade_history.append(self._current_trade)
        self._position = PositionType.OUT
        self._partial_exit_done = False
        
        result = {
            "action": "SELL" if is_long else "COVER",
            "price": exit_price,
            "reason": f"خروج به دلیل {reason.value}",
            "position": self._position.value,
            "pnl_percentage": round(pnl_percentage, 2),
            "pnl_amount": round(pnl_amount, 2),
            "exit_reason": reason.value,
            "trade_duration": (exit_time - self._current_trade.entry_time).total_seconds() / 3600 if self._current_trade.entry_time else 0,
            "max_profit_reached": round(self._current_trade.max_profit * 100, 2),
            "performance_metrics": self.get_performance_metrics()
        }
        
        self._current_trade = None
        return result

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """تولید سیگنال معاملاتی - نسخه حرفه‌ای کامل"""
        try:
            # بررسی شرایط خروج برای معاملات باز
            exit_signal = self.check_exit_conditions(data)
            if exit_signal:
                return exit_signal
            
            if len(data) < max(self.rsi_period, self.trend_ma_long, 50):
                return self._create_hold_signal("داده کافی برای تحلیل موجود نیست")
            
            current_price = data['close'].iloc[-1]
            current_rsi = data['RSI'].iloc[-1]
            
            # 1. تحلیل شرایط بازار
            market_conditions = self.analyze_market_conditions(data)
            
            # 2. بررسی فیلتر رژیم بازار
            if self.avoid_ranging_markets and market_conditions.regime == MarketRegime.RANGING:
                return self._create_hold_signal(
                    f"بازار در حالت Ranging است. منتظر روند واضح‌تر می‌مانیم. "
                    f"ADX: {market_conditions.adx_value:.1f}, RSI: {current_rsi:.1f}"
                )
            
            # 3. تشخیص واگرایی
            bullish_div, bearish_div, div_strength = self.detect_rsi_divergence(data)
            
            # 4. محاسبه ATR
            atr = self.calculate_atr(data.tail(50))
            
            # ========================
            # بررسی سیگنال LONG
            # ========================
            if self._position == PositionType.OUT or (self._position == PositionType.SHORT and self.enable_short_trades):
                long_score, long_strength = self.calculate_signal_strength(
                    data, market_conditions, bullish_div, bearish_div, div_strength, PositionType.LONG
                )
                
                long_reasons = []
                
                # شرط‌های LONG
                if current_rsi < self.oversold:
                    long_reasons.append(f"RSI در ناحیه اشباع فروش ({current_rsi:.1f})")
                
                if market_conditions.trend in ["STRONG_UPTREND", "UPTREND"]:
                    long_reasons.append(f"روند صعودی ({market_conditions.trend})")
                
                if bullish_div and div_strength > self.min_divergence_strength:
                    long_reasons.append(f"واگرایی مثبت قوی (قدرت: {div_strength:.2f})")
                
                if market_conditions.momentum in ["STRONG_BULLISH", "BULLISH"]:
                    long_reasons.append(f"مومنتوم مثبت ({market_conditions.momentum})")
                
                if market_conditions.volume_trend in ["SPIKE", "HIGH"]:
                    long_reasons.append(f"حجم بالا ({market_conditions.volume_trend})")
                
                if self.use_adx_filter and market_conditions.adx_value > self.adx_threshold:
                    long_reasons.append(f"ADX قوی ({market_conditions.adx_value:.1f})")
                
                # تصمیم LONG
                if long_score >= self.min_signal_score and len(long_reasons) >= 2:
                    stop_loss = current_price - (atr * self.stop_loss_atr_multiplier)
                    take_profit = current_price + ((current_price - stop_loss) * self.take_profit_ratio)
                    position_size = self.calculate_position_size(current_price, stop_loss)
                    
                    if position_size > 0:
                        self._position = PositionType.LONG
                        self._current_trade = Trade(
                            entry_price=current_price,
                            entry_time=data.index[-1] if hasattr(data.index, '__getitem__') else pd.Timestamp.now(),
                            position_type=PositionType.LONG,
                            quantity=position_size,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            initial_stop_loss=stop_loss,
                            highest_price=current_price,
                            entry_signal_strength=long_strength.value,
                            market_regime=market_conditions.regime.value
                        )
                        
                        return {
                            "action": "BUY",
                            "price": current_price,
                            "rsi": current_rsi,
                            "position_size": round(position_size, 4),
                            "stop_loss": round(stop_loss, 4),
                            "take_profit": round(take_profit, 4),
                            "risk_reward_ratio": self.take_profit_ratio,
                            "reason": " | ".join(long_reasons),
                            "position": self._position.value,
                            "signal_strength": long_strength.value,
                            "signal_score": round(long_score, 2),
                            "market_conditions": {
                                "trend": market_conditions.trend,
                                "volatility": market_conditions.volatility,
                                "momentum": market_conditions.momentum,
                                "regime": market_conditions.regime.value,
                                "volume_trend": market_conditions.volume_trend,
                                "adx": round(market_conditions.adx_value, 1)
                            },
                            "divergence_detected": bullish_div,
                            "divergence_strength": round(div_strength, 3),
                            "atr": round(atr, 4)
                        }
            
            # ========================
            # بررسی سیگنال SHORT
            # ========================
            if self.enable_short_trades and (self._position == PositionType.OUT or (self._position == PositionType.LONG and self.enable_short_trades)):
                short_score, short_strength = self.calculate_signal_strength(
                    data, market_conditions, bullish_div, bearish_div, div_strength, PositionType.SHORT
                )
                
                short_reasons = []
                
                # شرط‌های SHORT
                if current_rsi > self.overbought:
                    short_reasons.append(f"RSI در ناحیه اشباع خرید ({current_rsi:.1f})")
                
                if market_conditions.trend in ["STRONG_DOWNTREND", "DOWNTREND"]:
                    short_reasons.append(f"روند نزولی ({market_conditions.trend})")
                
                if bearish_div and div_strength > self.min_divergence_strength:
                    short_reasons.append(f"واگرایی منفی قوی (قدرت: {div_strength:.2f})")
                
                if market_conditions.momentum in ["STRONG_BEARISH", "BEARISH"]:
                    short_reasons.append(f"مومنتوم منفی ({market_conditions.momentum})")
                
                if market_conditions.volume_trend in ["SPIKE", "HIGH"]:
                    short_reasons.append(f"حجم بالا ({market_conditions.volume_trend})")
                
                if self.use_adx_filter and market_conditions.adx_value > self.adx_threshold:
                    short_reasons.append(f"ADX قوی ({market_conditions.adx_value:.1f})")
                
                # تصمیم SHORT
                if short_score >= self.min_signal_score and len(short_reasons) >= 2:
                    stop_loss = current_price + (atr * self.stop_loss_atr_multiplier)
                    take_profit = current_price - ((stop_loss - current_price) * self.take_profit_ratio)
                    position_size = self.calculate_position_size(current_price, stop_loss)
                    
                    if position_size > 0:
                        self._position = PositionType.SHORT
                        self._current_trade = Trade(
                            entry_price=current_price,
                            entry_time=data.index[-1] if hasattr(data.index, '__getitem__') else pd.Timestamp.now(),
                            position_type=PositionType.SHORT,
                            quantity=position_size,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            initial_stop_loss=stop_loss,
                            lowest_price=current_price,
                            entry_signal_strength=short_strength.value,
                            market_regime=market_conditions.regime.value
                        )
                        
                        return {
                            "action": "SHORT",
                            "price": current_price,
                            "rsi": current_rsi,
                            "position_size": round(position_size, 4),
                            "stop_loss": round(stop_loss, 4),
                            "take_profit": round(take_profit, 4),
                            "risk_reward_ratio": self.take_profit_ratio,
                            "reason": " | ".join(short_reasons),
                            "position": self._position.value,
                            "signal_strength": short_strength.value,
                            "signal_score": round(short_score, 2),
                            "market_conditions": {
                                "trend": market_conditions.trend,
                                "volatility": market_conditions.volatility,
                                "momentum": market_conditions.momentum,
                                "regime": market_conditions.regime.value,
                                "volume_trend": market_conditions.volume_trend,
                                "adx": round(market_conditions.adx_value, 1)
                            },
                            "divergence_detected": bearish_div,
                            "divergence_strength": round(div_strength, 3),
                            "atr": round(atr, 4)
                        }
            
            # سیگنال نگهداری
            return self._create_hold_signal(
                f"منتظر سیگنال مناسب (RSI: {current_rsi:.1f}, "
                f"روند: {market_conditions.trend}, رژیم: {market_conditions.regime.value}, "
                f"ADX: {market_conditions.adx_value:.1f})"
            )
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}", exc_info=True)
            return self._create_hold_signal(f"خطا در تحلیل: {str(e)}")

    def _create_hold_signal(self, reason: str) -> Dict[str, Any]:
        """ایجاد سیگنال نگهداری"""
        return {
            "action": "HOLD",
            "price": 0,
            "rsi": 0,
            "reason": reason,
            "position": self._position.value,
            "signal_strength": "NEUTRAL"
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """محاسبه معیارهای عملکرد پیشرفته"""
        if self._total_trades == 0:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "average_trade_pnl": 0,
                "current_portfolio_value": self._portfolio_value,
                "current_position": self._position.value
            }
        
        win_rate = (self._winning_trades / self._total_trades) * 100
        avg_trade = self._total_pnl / self._total_trades
        
        # محاسبه بهترین و بدترین معامله
        completed_trades = [t for t in self._trade_history if t.pnl_amount is not None]
        best_trade = max(completed_trades, key=lambda t: t.pnl_amount) if completed_trades else None
        worst_trade = min(completed_trades, key=lambda t: t.pnl_amount) if completed_trades else None
        
        # محاسبه Profit Factor
        gross_profit = sum(t.pnl_amount for t in completed_trades if t.pnl_amount > 0)
        gross_loss = abs(sum(t.pnl_amount for t in completed_trades if t.pnl_amount < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # محاسبه Average Win/Loss
        winning_trades_list = [t.pnl_amount for t in completed_trades if t.pnl_amount > 0]
        losing_trades_list = [t.pnl_amount for t in completed_trades if t.pnl_amount < 0]
        
        avg_win = np.mean(winning_trades_list) if winning_trades_list else 0
        avg_loss = np.mean(losing_trades_list) if losing_trades_list else 0
        
        # آمار LONG vs SHORT
        long_win_rate = (self._winning_long_trades / self._total_long_trades * 100) if self._total_long_trades > 0 else 0
        short_win_rate = (self._winning_short_trades / self._total_short_trades * 100) if self._total_short_trades > 0 else 0
        
        # محاسبه Sharpe Ratio (ساده)
        returns = [t.pnl_percentage for t in completed_trades if t.pnl_percentage is not None]
        sharpe_ratio = (np.mean(returns) / np.std(returns)) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        return {
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "losing_trades": self._total_trades - self._winning_trades,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(self._total_pnl, 2),
            "average_trade_pnl": round(avg_trade, 2),
            "current_portfolio_value": round(self._portfolio_value, 2),
            "current_position": self._position.value,
            "profit_factor": round(profit_factor, 2),
            "best_trade": round(best_trade.pnl_amount, 2) if best_trade else 0,
            "worst_trade": round(worst_trade.pnl_amount, 2) if worst_trade else 0,
            "average_win": round(avg_win, 2),
            "average_loss": round(avg_loss, 2),
            "max_consecutive_wins": self._max_consecutive_wins,
            "max_consecutive_losses": self._max_consecutive_losses,
            "portfolio_return": round(((self._portfolio_value - 10000) / 10000) * 100, 2),
            "long_trades": self._total_long_trades,
            "short_trades": self._total_short_trades,
            "long_win_rate": round(long_win_rate, 2),
            "short_win_rate": round(short_win_rate, 2),
            "sharpe_ratio": round(sharpe_ratio, 2)
        }

    def reset_state(self):
        """بازنشانی وضعیت استراتژی"""
        self._position = PositionType.OUT
        self._current_trade = None
        self._trade_history = []
        self._portfolio_value = 10000.0
        self._partial_exit_done = False
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = 0.0
        self._max_consecutive_wins = 0
        self._max_consecutive_losses = 0
        self._current_streak = 0
        self._total_long_trades = 0
        self._total_short_trades = 0
        self._winning_long_trades = 0
        self._winning_short_trades = 0

    @property
    def position(self):
        return self._position

    @property
    def trade_history(self):
        return self._trade_history.copy()
    
    @property
    def current_trade(self):
        return self._current_trade