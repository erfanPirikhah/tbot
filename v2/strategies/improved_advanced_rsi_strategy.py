# strategies/improved_advanced_rsi_strategy.py

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

@dataclass
class Trade:
    """کلاس برای ذخیره اطلاعات معامله"""
    entry_price: float
    entry_time: pd.Timestamp
    position_type: PositionType
    quantity: float
    stop_loss: float
    take_profit: float
    initial_stop_loss: float = 0.0
    trailing_stop: float = 0.0
    highest_price: float = 0.0  # برای trailing stop
    lowest_price: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_reason: Optional[ExitReason] = None
    pnl_percentage: Optional[float] = None
    pnl_amount: Optional[float] = None
    max_drawdown: float = 0.0
    max_profit: float = 0.0

@dataclass
class MarketConditions:
    """کلاس برای تحلیل شرایط بازار"""
    trend: str
    volatility: str
    momentum: str
    strength_score: float = 0.0
    volume_trend: str = "NEUTRAL"

class ImprovedAdvancedRsiStrategy:
    """
    استراتژی بهبود یافته RSI با ویژگی‌های:
    - تشخیص واگرایی دقیق‌تر
    - Trailing Stop Loss
    - مدیریت معاملات باز
    - فیلترهای حجم معاملات
    - سیستم امتیازدهی هوشمند
    """
    
    def __init__(
        self,
        overbought: int = 70,
        oversold: int = 30,
        rsi_period: int = 14,
        # پارامترهای فیلتر روند
        trend_ma_short: int = 20,
        trend_ma_long: int = 50,
        trend_threshold: float = 0.015,  # 1.5%
        # پارامترهای تاییدیه
        min_conditions: int = 3,  # تعداد کمتر از 4
        volume_ma_period: int = 20,
        volume_spike_threshold: float = 1.5,
        # پارامترهای مدیریت ریسک
        risk_per_trade: float = 0.02,
        stop_loss_atr_multiplier: float = 1.5,
        take_profit_ratio: float = 2.5,  # افزایش یافته
        # Trailing Stop
        use_trailing_stop: bool = True,
        trailing_stop_activation: float = 1.0,  # فعال شدن بعد از 1:1
        trailing_stop_distance: float = 0.5,  # 0.5 ATR
        # پارامترهای تشخیص واگرایی
        divergence_lookback: int = 14,
        min_divergence_strength: float = 0.2,  # کاهش یافته
        # مدیریت زمان
        max_trade_duration: int = 72,  # ساعت
    ):
        self._validate_parameters(
            overbought, oversold, risk_per_trade, 
            stop_loss_atr_multiplier, take_profit_ratio
        )
        
        # پارامترهای استراتژی
        self.overbought = overbought
        self.oversold = oversold
        self.rsi_period = rsi_period
        self.trend_ma_short = trend_ma_short
        self.trend_ma_long = trend_ma_long
        self.trend_threshold = trend_threshold
        self.min_conditions = min_conditions
        self.volume_ma_period = volume_ma_period
        self.volume_spike_threshold = volume_spike_threshold
        self.risk_per_trade = risk_per_trade
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_ratio = take_profit_ratio
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_activation = trailing_stop_activation
        self.trailing_stop_distance = trailing_stop_distance
        self.divergence_lookback = divergence_lookback
        self.min_divergence_strength = min_divergence_strength
        self.max_trade_duration = max_trade_duration
        
        # مدیریت وضعیت
        self._position = PositionType.OUT
        self._current_trade: Optional[Trade] = None
        self._trade_history: List[Trade] = []
        self._portfolio_value: float = 10000.0
        
        # آمار عملکرد
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = 0.0
        self._max_consecutive_wins = 0
        self._max_consecutive_losses = 0
        self._current_streak = 0

    def _validate_parameters(self, overbought: int, oversold: int, 
                           risk_per_trade: float, stop_loss_multiplier: float,
                           take_profit_ratio: float):
        """اعتبارسنجی پارامترهای ورودی"""
        if overbought <= oversold:
            raise ValueError("سطح overbought باید از oversold بزرگتر باشد")
        if not (0 < risk_per_trade <= 0.1):
            raise ValueError("ریسک هر معامله باید بین 0.1% تا 10% باشد")
        if stop_loss_multiplier <= 0:
            raise ValueError("ضریب stop loss باید بزرگتر از صفر باشد")
        if take_profit_ratio <= 0:
            raise ValueError("نسبت take profit باید بزرگتر از صفر باشد")

    def analyze_market_conditions(self, data: pd.DataFrame) -> MarketConditions:
        """تحلیل جامع شرایط بازار با Volume"""
        prices = data['close'].tail(100)
        volumes = data['volume'].tail(100) if 'volume' in data.columns else None
        
        # تحلیل روند با دو MA
        ma_short = prices.rolling(self.trend_ma_short).mean().iloc[-1]
        ma_long = prices.rolling(self.trend_ma_long).mean().iloc[-1] if len(prices) >= self.trend_ma_long else ma_short
        
        price_current = prices.iloc[-1]
        
        # تعیین روند قوی‌تر
        if price_current > ma_short > ma_long and ma_short > ma_long * (1 + self.trend_threshold):
            trend = "STRONG_UPTREND"
            trend_score = 3.0
        elif price_current > ma_short and ma_short > ma_long:
            trend = "UPTREND"
            trend_score = 2.0
        elif price_current < ma_short < ma_long and ma_short < ma_long * (1 - self.trend_threshold):
            trend = "STRONG_DOWNTREND"
            trend_score = -3.0
        elif price_current < ma_short and ma_short < ma_long:
            trend = "DOWNTREND"
            trend_score = -2.0
        else:
            trend = "SIDEWAYS"
            trend_score = 0.0
        
        # تحلیل نوسانات با ATR
        volatility_value = self._calculate_volatility(data)
        if volatility_value > 0.05:
            volatility_level = "HIGH"
        elif volatility_value > 0.02:
            volatility_level = "MEDIUM"
        else:
            volatility_level = "LOW"
        
        # تحلیل مومنتوم با RSI
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
        
        # تحلیل حجم معاملات
        volume_trend = "NEUTRAL"
        if volumes is not None:
            volume_ma = volumes.rolling(self.volume_ma_period).mean()
            current_volume = volumes.iloc[-1]
            avg_volume = volume_ma.iloc[-1]
            
            if current_volume > avg_volume * self.volume_spike_threshold:
                volume_trend = "HIGH"
            elif current_volume > avg_volume:
                volume_trend = "ABOVE_AVERAGE"
            elif current_volume < avg_volume * 0.7:
                volume_trend = "LOW"
        
        return MarketConditions(
            trend=trend,
            volatility=volatility_level,
            momentum=momentum,
            strength_score=trend_score,
            volume_trend=volume_trend
        )

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """محاسبه نوسانات بازار"""
        returns = data['close'].pct_change().tail(20)
        return returns.std()

    def detect_rsi_divergence(self, data: pd.DataFrame) -> Tuple[bool, bool, float]:
        """
        تشخیص واگرایی RSI با الگوریتم بهبود یافته
        """
        if len(data) < self.divergence_lookback:
            return False, False, 0.0
        
        prices = data['close'].tail(self.divergence_lookback).values
        rsi_values = data['RSI'].tail(self.divergence_lookback).values
        
        # پیدا کردن قله‌ها و دره‌ها با scipy
        price_peaks_idx = argrelextrema(prices, np.greater, order=2)[0]
        price_troughs_idx = argrelextrema(prices, np.less, order=2)[0]
        rsi_peaks_idx = argrelextrema(rsi_values, np.greater, order=2)[0]
        rsi_troughs_idx = argrelextrema(rsi_values, np.less, order=2)[0]
        
        bullish_divergence = False
        bearish_divergence = False
        divergence_strength = 0.0
        
        # واگرایی مثبت: قیمت کف پایین‌تر، RSI کف بالاتر
        if len(price_troughs_idx) >= 2 and len(rsi_troughs_idx) >= 2:
            # مقایسه 2 کف آخر
            p_trough_1, p_trough_2 = prices[price_troughs_idx[-2]], prices[price_troughs_idx[-1]]
            r_trough_1, r_trough_2 = rsi_values[rsi_troughs_idx[-2]], rsi_values[rsi_troughs_idx[-1]]
            
            if p_trough_2 < p_trough_1 and r_trough_2 > r_trough_1:
                bullish_divergence = True
                # قدرت واگرایی = میانگین تغییرات نسبی
                price_change = abs((p_trough_2 - p_trough_1) / p_trough_1)
                rsi_change = abs((r_trough_2 - r_trough_1) / r_trough_1)
                divergence_strength = (price_change + rsi_change) / 2
        
        # واگرایی منفی: قیمت سقف بالاتر، RSI سقف پایین‌تر
        if len(price_peaks_idx) >= 2 and len(rsi_peaks_idx) >= 2:
            p_peak_1, p_peak_2 = prices[price_peaks_idx[-2]], prices[price_peaks_idx[-1]]
            r_peak_1, r_peak_2 = rsi_values[rsi_peaks_idx[-2]], rsi_values[rsi_peaks_idx[-1]]
            
            if p_peak_2 > p_peak_1 and r_peak_2 < r_peak_1:
                bearish_divergence = True
                price_change = abs((p_peak_2 - p_peak_1) / p_peak_1)
                rsi_change = abs((r_peak_2 - r_peak_1) / r_peak_1)
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

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """محاسبه حجم معامله بر اساس مدیریت ریسک"""
        risk_amount = self._portfolio_value * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
            
        position_size = risk_amount / price_risk
        max_position = self._portfolio_value * 0.15  # افزایش به 15%
        
        return min(position_size, max_position)

    def calculate_signal_strength(
        self, 
        data: pd.DataFrame, 
        market_conditions: MarketConditions,
        bullish_div: bool, 
        bearish_div: bool,
        divergence_strength: float
    ) -> SignalStrength:
        """محاسبه قدرت سیگنال با سیستم امتیازدهی"""
        score = 0.0
        current_rsi = data['RSI'].iloc[-1]
        
        # امتیاز RSI (0-4 امتیاز)
        if current_rsi < 20 or current_rsi > 80:
            score += 4.0
        elif current_rsi < 25 or current_rsi > 75:
            score += 3.0
        elif current_rsi < 30 or current_rsi > 70:
            score += 2.0
        elif current_rsi < 35 or current_rsi > 65:
            score += 1.0
        
        # امتیاز واگرایی (0-4 امتیاز)
        if bullish_div or bearish_div:
            div_score = min(divergence_strength * 10, 4.0)
            score += div_score
        
        # امتیاز روند (0-3 امتیاز)
        if "STRONG" in market_conditions.trend:
            score += 3.0
        elif market_conditions.trend in ["UPTREND", "DOWNTREND"]:
            score += 2.0
        elif market_conditions.trend == "SIDEWAYS":
            score += 0.5
        
        # امتیاز مومنتوم (0-2 امتیاز)
        if "STRONG" in market_conditions.momentum:
            score += 2.0
        elif market_conditions.momentum in ["BULLISH", "BEARISH"]:
            score += 1.0
        
        # امتیاز حجم (0-2 امتیاز)
        if market_conditions.volume_trend == "HIGH":
            score += 2.0
        elif market_conditions.volume_trend == "ABOVE_AVERAGE":
            score += 1.0
        
        # طبقه‌بندی (حداکثر 15 امتیاز)
        if score >= 12.0:
            return SignalStrength.VERY_STRONG
        elif score >= 9.0:
            return SignalStrength.STRONG
        elif score >= 6.0:
            return SignalStrength.MEDIUM
        else:
            return SignalStrength.WEAK

    def check_exit_conditions(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """بررسی شرایط خروج از معامله"""
        if not self._current_trade or self._position == PositionType.OUT:
            return None
        
        current_price = data['close'].iloc[-1]
        current_time = data.index[-1] if hasattr(data.index, '__getitem__') else pd.Timestamp.now()
        
        # بررسی Stop Loss
        if current_price <= self._current_trade.stop_loss:
            return self._exit_trade(current_price, current_time, ExitReason.STOP_LOSS)
        
        # بررسی Take Profit
        if current_price >= self._current_trade.take_profit:
            return self._exit_trade(current_price, current_time, ExitReason.TAKE_PROFIT)
        
        # بررسی Trailing Stop
        if self.use_trailing_stop and self._current_trade.trailing_stop > 0:
            if current_price <= self._current_trade.trailing_stop:
                return self._exit_trade(current_price, current_time, ExitReason.TRAILING_STOP)
        
        # بررسی حداکثر زمان معامله
        if self._current_trade.entry_time:
            duration = (current_time - self._current_trade.entry_time).total_seconds() / 3600
            if duration >= self.max_trade_duration:
                return self._exit_trade(current_price, current_time, ExitReason.TIME_EXIT)
        
        # به‌روزرسانی Trailing Stop
        self._update_trailing_stop(current_price)
        
        return None

    def _update_trailing_stop(self, current_price: float):
        """به‌روزرسانی Trailing Stop"""
        if not self.use_trailing_stop or not self._current_trade:
            return
        
        # محاسبه سود فعلی
        profit_ratio = (current_price - self._current_trade.entry_price) / self._current_trade.entry_price
        
        # فعال‌سازی trailing stop بعد از رسیدن به نسبت مشخص
        if profit_ratio >= self.trailing_stop_activation:
            risk_distance = self._current_trade.entry_price - self._current_trade.initial_stop_loss
            trailing_distance = risk_distance * self.trailing_stop_distance
            
            new_trailing_stop = current_price - trailing_distance
            
            # فقط بالا بردن trailing stop
            if new_trailing_stop > self._current_trade.trailing_stop:
                self._current_trade.trailing_stop = new_trailing_stop
                self._current_trade.stop_loss = new_trailing_stop

    def _exit_trade(self, exit_price: float, exit_time: pd.Timestamp, reason: ExitReason) -> Dict[str, Any]:
        """خروج از معامله"""
        pnl_percentage = ((exit_price - self._current_trade.entry_price) / 
                         self._current_trade.entry_price * 100)
        pnl_amount = (exit_price - self._current_trade.entry_price) * self._current_trade.quantity
        
        # آپدیت پورتفو
        self._portfolio_value += pnl_amount
        self._total_trades += 1
        
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
        
        result = {
            "action": "SELL",
            "price": exit_price,
            "reason": f"خروج به دلیل {reason.value}",
            "position": self._position.value,
            "pnl_percentage": pnl_percentage,
            "pnl_amount": pnl_amount,
            "exit_reason": reason.value,
            "trade_duration": (exit_time - self._current_trade.entry_time).total_seconds() / 3600 if self._current_trade.entry_time else 0,
            "performance_metrics": self.get_performance_metrics()
        }
        
        self._current_trade = None
        return result

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """تولید سیگنال معاملاتی بهبود یافته"""
        try:
            # بررسی شرایط خروج برای معاملات باز
            exit_signal = self.check_exit_conditions(data)
            if exit_signal:
                return exit_signal
            
            if len(data) < max(self.rsi_period, self.trend_ma_long, 20):
                return self._create_hold_signal("داده کافی برای تحلیل موجود نیست")
            
            current_price = data['close'].iloc[-1]
            current_rsi = data['RSI'].iloc[-1]
            
            # 1. تحلیل شرایط بازار
            market_conditions = self.analyze_market_conditions(data)
            
            # 2. تشخیص واگرایی
            bullish_div, bearish_div, div_strength = self.detect_rsi_divergence(data)
            
            # 3. محاسبه قدرت سیگنال
            signal_strength = self.calculate_signal_strength(
                data, market_conditions, bullish_div, bearish_div, div_strength
            )
            
            # 4. محاسبه ATR
            atr = self.calculate_atr(data.tail(20))
            
            # 5. شرایط خرید با سیستم امتیازدهی
            buy_score = 0
            buy_reasons = []
            
            # شرط RSI (حیاتی)
            if current_rsi < self.oversold:
                buy_score += 2
                buy_reasons.append(f"RSI در ناحیه اشباع فروش ({current_rsi:.1f})")
            elif current_rsi < self.oversold + 5:
                buy_score += 1
                buy_reasons.append(f"RSI نزدیک به اشباع فروش ({current_rsi:.1f})")
            
            # شرط روند (مهم)
            if market_conditions.trend in ["STRONG_UPTREND", "UPTREND"]:
                buy_score += 2
                buy_reasons.append(f"روند صعودی ({market_conditions.trend})")
            elif market_conditions.trend == "SIDEWAYS":
                buy_score += 1
                buy_reasons.append("روند خنثی")
            
            # شرط واگرایی (مهم)
            if bullish_div and div_strength > self.min_divergence_strength:
                buy_score += 2
                buy_reasons.append(f"واگرایی مثبت قوی (قدرت: {div_strength:.2f})")
            elif bullish_div:
                buy_score += 1
                buy_reasons.append("واگرایی مثبت ضعیف")
            
            # شرط مومنتوم
            if market_conditions.momentum in ["STRONG_BULLISH", "BULLISH"]:
                buy_score += 1
                buy_reasons.append(f"مومنتوم مثبت ({market_conditions.momentum})")
            
            # شرط حجم
            if market_conditions.volume_trend in ["HIGH", "ABOVE_AVERAGE"]:
                buy_score += 1
                buy_reasons.append(f"حجم معاملات بالا ({market_conditions.volume_trend})")
            
            # شرط وضعیت معاملاتی (حیاتی)
            if self._position == PositionType.OUT:
                buy_score += 1
            else:
                buy_score = 0  # اگر پوزیشن باز داریم، نمی‌توانیم خرید کنیم
            
            # شرط قدرت سیگنال
            if signal_strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]:
                buy_score += 1
                buy_reasons.append(f"قدرت سیگنال بالا ({signal_strength.value})")
            
            # تصمیم خرید: حداقل امتیاز کمتر و انعطاف‌پذیرتر
            min_buy_score = 5  # کاهش از 4 شرط به امتیاز 5
            
            if buy_score >= min_buy_score:
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
                        lowest_price=current_price
                    )
                    
                    return {
                        "action": "BUY",
                        "price": current_price,
                        "rsi": current_rsi,
                        "position_size": position_size,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "risk_reward_ratio": self.take_profit_ratio,
                        "reason": " | ".join(buy_reasons),
                        "position": self._position.value,
                        "signal_strength": signal_strength.value,
                        "buy_score": buy_score,
                        "market_conditions": {
                            "trend": market_conditions.trend,
                            "volatility": market_conditions.volatility,
                            "momentum": market_conditions.momentum,
                            "volume_trend": market_conditions.volume_trend
                        },
                        "divergence_detected": bullish_div,
                        "divergence_strength": div_strength
                    }
            
            # سیگنال نگهداری
            return self._create_hold_signal(
                f"منتظر سیگنال مناسب (امتیاز: {buy_score}/{min_buy_score}). "
                f"شرایط بازار: {market_conditions.trend}, RSI: {current_rsi:.1f}, "
                f"قدرت: {signal_strength.value}"
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
            "portfolio_return": round(((self._portfolio_value - 10000) / 10000) * 100, 2)
        }

    def reset_state(self):
        """بازنشانی وضعیت استراتژی"""
        self._position = PositionType.OUT
        self._current_trade = None
        self._trade_history = []
        self._portfolio_value = 10000.0
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = 0.0
        self._max_consecutive_wins = 0
        self._max_consecutive_losses = 0
        self._current_streak = 0

    @property
    def position(self):
        return self._position

    @property
    def trade_history(self):
        return self._trade_history.copy()
    
    @property
    def current_trade(self):
        return self._current_trade