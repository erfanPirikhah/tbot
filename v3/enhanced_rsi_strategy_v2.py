# strategies/enhanced_rsi_strategy_v2_improved.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PositionType(Enum):
    OUT = "OUT"
    LONG = "LONG"
    SHORT = "SHORT"

class ExitReason(Enum):
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    SIGNAL_EXIT = "SIGNAL_EXIT"
    TIME_EXIT = "TIME_EXIT"
    PARTIAL_EXIT = "PARTIAL_EXIT"

@dataclass
class TradeEntry:
    price: float
    quantity: float
    time: pd.Timestamp

@dataclass
class Trade:
    entries: List[TradeEntry] = field(default_factory=list)
    position_type: PositionType = PositionType.OUT
    stop_loss: float = 0.0
    take_profit: float = 0.0
    initial_stop_loss: float = 0.0
    trailing_stop: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_reason: Optional[ExitReason] = None
    pnl_percentage: Optional[float] = None
    pnl_amount: Optional[float] = None
    partial_exits: List[Dict] = field(default_factory=list)
    
    @property
    def entry_price(self) -> float:
        """محاسبه میانگین وزنی قیمت ورود"""
        if not self.entries:
            return 0.0
        total_cost = sum(entry.price * entry.quantity for entry in self.entries)
        total_quantity = sum(entry.quantity for entry in self.entries)
        return total_cost / total_quantity if total_quantity > 0 else 0.0
    
    @property
    def quantity(self) -> float:
        """مجموع حجم معامله"""
        return sum(entry.quantity for entry in self.entries)
    
    @property
    def entry_time(self) -> pd.Timestamp:
        """زمان اولین ورود"""
        return self.entries[0].time if self.entries else pd.Timestamp.now()

class EnhancedRsiStrategyV2Improved:
    """
    استراتژی RSI نسخه ۲ بهبود یافته با رفع نقاط ضعف عملیاتی
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        rsi_base_oversold: int = 35,  # 🔧 تغییر از 30 به 35 برای سیگنال بیشتر
        rsi_base_overbought: int = 65,  # 🔧 تغییر از 70 به 65 برای سیگنال بیشتر
        risk_per_trade: float = 0.015,
        base_stop_atr_multiplier: float = 2.5,
        base_take_profit_ratio: float = 2.5,
        max_trade_duration: int = 150,  # 🔧 افزایش از 72 به 150 برای تایم‌فریم‌های بالاتر
        enable_short_trades: bool = True,
        use_trend_filter: bool = True,
        use_divergence: bool = True,
        use_partial_exit: bool = True,
        max_trades_per_100: int = 15,
        min_candles_between: int = 8,
        enable_trailing_stop: bool = True,
        trailing_atr_multiplier: float = 1.5,
        enable_adaptive_rsi: bool = True,
        adaptive_rsi_sensitivity: float = 0.5,
        enable_analytical_logging: bool = True,
        enable_pyramiding: bool = False,
        pyramid_profit_threshold: float = 1.0,
        pyramid_max_entries: int = 3,
        pyramid_risk_reduction: float = 0.5,
        # 🔧 پارامترهای جدید برای بهبود
        min_position_size: float = 0.001,  # 🔧 حداقل سایز پوزیشن
        ma_period_trend: int = 20,  # 🔧 دوره کوتاه‌تر برای MA
        divergence_lookback: int = 10,  # 🔧 دوره lookback برای واگرایی
    ):
        # پارامترهای اصلی
        self.rsi_period = rsi_period
        self.rsi_base_oversold = rsi_base_oversold
        self.rsi_base_overbought = rsi_base_overbought
        self.risk_per_trade = risk_per_trade
        self.base_stop_atr_multiplier = base_stop_atr_multiplier
        self.base_take_profit_ratio = base_take_profit_ratio
        self.max_trade_duration = max_trade_duration
        self.enable_short_trades = enable_short_trades
        self.use_trend_filter = use_trend_filter
        self.use_divergence = use_divergence
        self.use_partial_exit = use_partial_exit
        self.max_trades_per_100 = max_trades_per_100
        self.min_candles_between = min_candles_between
        self.enable_trailing_stop = enable_trailing_stop
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.enable_adaptive_rsi = enable_adaptive_rsi
        self.adaptive_rsi_sensitivity = adaptive_rsi_sensitivity
        self.enable_analytical_logging = enable_analytical_logging
        
        # پارامترهای نسخه ۲
        self.enable_pyramiding = enable_pyramiding
        self.pyramid_profit_threshold = pyramid_profit_threshold
        self.pyramid_max_entries = pyramid_max_entries
        self.pyramid_risk_reduction = pyramid_risk_reduction
        
        # 🔧 پارامترهای جدید بهبود
        self.min_position_size = min_position_size
        self.ma_period_trend = ma_period_trend
        self.divergence_lookback = divergence_lookback
        
        # State
        self._position = PositionType.OUT
        self._current_trade: Optional[Trade] = None
        self._trade_history: List[Trade] = []
        self._portfolio_value: float = 10000.0
        self._last_trade_index: int = -100
        
        # Statistics
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = 0.0
        self._gross_profit = 0.0
        self._gross_loss = 0.0
        self._daily_returns: List[float] = []
        self._last_portfolio_value: float = 10000.0
        
        # Drawdown Tracking
        self._max_portfolio_value: float = 10000.0
        self._max_drawdown: float = 0.0
        self._current_drawdown: float = 0.0
        
        # Equity Curve Logging
        self._equity_curve: List[Tuple[pd.Timestamp, float]] = []

    def _create_hold_signal(self, reason: str) -> Dict[str, Any]:
        """ایجاد سیگنال HOLD"""
        return {
            "action": "HOLD",
            "reason": reason,
            "position": self._position.value
        }

    def check_exit_conditions(self, data: pd.DataFrame, current_index: int) -> Optional[Dict[str, Any]]:
        """بررسی شرایط خروج از معامله - نسخه بهبود یافته"""
        if self._position == PositionType.OUT or self._current_trade is None:
            return None

        current_price = data['close'].iloc[-1]
        current_time = data.index[-1]

        # 🔧 بهبود: بررسی Trailing Stop فعال
        if self.enable_trailing_stop and self._current_trade.trailing_stop > 0:
            if self._position == PositionType.LONG:
                # به‌روزرسانی trailing stop برای long
                new_trailing_stop = current_price - (self.calculate_atr(data) * self.trailing_atr_multiplier)
                self._current_trade.trailing_stop = max(new_trailing_stop, self._current_trade.trailing_stop)
                if current_price <= self._current_trade.trailing_stop:
                    return self._create_exit_signal("TRAILING_STOP", current_price, current_time)
            else:  # SHORT
                new_trailing_stop = current_price + (self.calculate_atr(data) * self.trailing_atr_multiplier)
                self._current_trade.trailing_stop = min(new_trailing_stop, self._current_trade.trailing_stop)
                if current_price >= self._current_trade.trailing_stop:
                    return self._create_exit_signal("TRAILING_STOP", current_price, current_time)

        # بررسی Stop Loss
        if self._current_trade.stop_loss > 0:
            if (self._position == PositionType.LONG and current_price <= self._current_trade.stop_loss):
                return self._create_exit_signal("STOP_LOSS", current_price, current_time)
            elif (self._position == PositionType.SHORT and current_price >= self._current_trade.stop_loss):
                return self._create_exit_signal("STOP_LOSS", current_price, current_time)

        # بررسی Take Profit
        if self._current_trade.take_profit > 0:
            if (self._position == PositionType.LONG and current_price >= self._current_trade.take_profit):
                return self._create_exit_signal("TAKE_PROFIT", current_price, current_time)
            elif (self._position == PositionType.SHORT and current_price <= self._current_trade.take_profit):
                return self._create_exit_signal("TAKE_PROFIT", current_price, current_time)

        # بررسی Time Exit (با دوره طولانی‌تر)
        if self.max_trade_duration > 0:
            trade_duration = current_index - self._last_trade_index
            if trade_duration >= self.max_trade_duration:
                return self._create_exit_signal("TIME_EXIT", current_price, current_time)

        return None

    def _create_exit_signal(self, exit_reason: str, price: float, time: pd.Timestamp) -> Dict[str, Any]:
        """ایجاد سیگنال خروج"""
        if self._current_trade is None:
            return self._create_hold_signal("No trade to exit")

        # محاسبه سود/زیان
        entry_price = self._current_trade.entry_price
        if self._position == PositionType.LONG:
            pnl_pct = ((price - entry_price) / entry_price) * 100
            pnl_amount = self._current_trade.quantity * (price - entry_price)
        else:
            pnl_pct = ((entry_price - price) / entry_price) * 100
            pnl_amount = self._current_trade.quantity * (entry_price - price)

        # به‌روزرسانی پورتفو
        self._portfolio_value += pnl_amount
        self._total_pnl += pnl_amount
        self._total_trades += 1

        if pnl_amount > 0:
            self._winning_trades += 1
            self._gross_profit += pnl_amount
        else:
            self._gross_loss += abs(pnl_amount)

        # ثبت معامله
        self._current_trade.exit_price = price
        self._current_trade.exit_time = time
        self._current_trade.pnl_percentage = pnl_pct
        self._current_trade.pnl_amount = pnl_amount

        self._trade_history.append(self._current_trade)

        # ریست پوزیشن
        old_position = self._position
        self._position = PositionType.OUT
        self._current_trade = None

        return {
            "action": "EXIT",
            "price": price,
            "exit_reason": exit_reason,
            "pnl_percentage": round(pnl_pct, 2),
            "pnl_amount": round(pnl_amount, 2),
            "position": "OUT",
            "previous_position": old_position.value
        }

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """محاسبه ATR"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else 0.0

    def calculate_dynamic_stop(self, data: pd.DataFrame, position_type: PositionType) -> Tuple[float, float]:
        """محاسبه Stop Loss و Take Profit داینامیک"""
        atr = self.calculate_atr(data)
        current_price = data['close'].iloc[-1]

        if atr == 0:  # 🔧 جلوگیری از خطا
            atr = current_price * 0.01  # 1% fallback

        if position_type == PositionType.LONG:
            stop_loss = current_price - (atr * self.base_stop_atr_multiplier)
            take_profit = current_price + (atr * self.base_stop_atr_multiplier * self.base_take_profit_ratio)
        else:
            stop_loss = current_price + (atr * self.base_stop_atr_multiplier)
            take_profit = current_price - (atr * self.base_stop_atr_multiplier * self.base_take_profit_ratio)

        return stop_loss, take_profit

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """محاسبه سایز پوزیشن - نسخه بهبود یافته"""
        risk_amount = self._portfolio_value * self.risk_per_trade
        
        if self._position == PositionType.LONG:
            price_risk = entry_price - stop_loss
        else:
            price_risk = stop_loss - entry_price

        if price_risk <= 0:
            # 🔧 اگر price_risk صفر یا منفی باشد، از حداقل سایز استفاده کن
            return self.min_position_size

        position_size = risk_amount / price_risk
        
        # 🔧 اضافه کردن حداقل سایز پوزیشن
        position_size = max(position_size, self.min_position_size)
        
        return position_size

    def _calculate_divergence(self, data: pd.DataFrame) -> Tuple[bool, bool]:
        """
        🔧 محاسبه واگرایی - نسخه عملیاتی
        بازگشت: (واگرایی_مثبت, واگرایی_منفی)
        """
        if len(data) < self.divergence_lookback + 5:
            return False, False

        try:
            # محاسبه highs و lows
            highs = data['high'].tail(self.divergence_lookback)
            lows = data['low'].tail(self.divergence_lookback)
            rsi_values = data['RSI'].tail(self.divergence_lookback)

            # پیدا کردن سقف‌ها و کف‌ها
            price_peaks = []
            price_troughs = []
            rsi_peaks = []
            rsi_troughs = []

            for i in range(1, len(highs)-1):
                # سقف قیمت
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                    price_peaks.append((i, highs.iloc[i]))
                    rsi_peaks.append((i, rsi_values.iloc[i]))
                
                # کف قیمت
                if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                    price_troughs.append((i, lows.iloc[i]))
                    rsi_troughs.append((i, rsi_values.iloc[i]))

            # بررسی واگرایی منفی (Bearish Divergence)
            bearish_divergence = False
            if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
                # قیمت سقف بالاتر ولی RSI سقف پایین‌تر
                last_price_peak = price_peaks[-1][1]
                prev_price_peak = price_peaks[-2][1]
                last_rsi_peak = rsi_peaks[-1][1]
                prev_rsi_peak = rsi_peaks[-2][1]

                if (last_price_peak > prev_price_peak) and (last_rsi_peak < prev_rsi_peak):
                    bearish_divergence = True

            # بررسی واگرایی مثبت (Bullish Divergence)
            bullish_divergence = False
            if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
                # قیمت کف پایین‌تر ولی RSI کف بالاتر
                last_price_trough = price_troughs[-1][1]
                prev_price_trough = price_troughs[-2][1]
                last_rsi_trough = rsi_troughs[-1][1]
                prev_rsi_trough = rsi_troughs[-2][1]

                if (last_price_trough < prev_price_trough) and (last_rsi_trough > prev_rsi_trough):
                    bullish_divergence = True

            return bullish_divergence, bearish_divergence

        except Exception as e:
            logger.error(f"خطا در محاسبه واگرایی: {e}")
            return False, False

    def check_entry_conditions(self, data: pd.DataFrame, position_type: PositionType) -> Tuple[bool, List[str]]:
        """بررسی شرایط ورود - نسخه بهبود یافته"""
        conditions = []
        current_rsi = data['RSI'].iloc[-1]

        # 🔧 بهبود: اضافه کردن واگرایی به داده‌ها
        bullish_div, bearish_div = self._calculate_divergence(data)

        if position_type == PositionType.LONG:
            # 🔧 بهبود: آستانه نرم‌تر برای RSI
            if current_rsi <= self.rsi_base_oversold:
                conditions.append(f"RSI oversold ({current_rsi:.1f} <= {self.rsi_base_oversold})")
            else:
                return False, []

            # 🔧 بهبود: فیلتر MA با دوره کوتاه‌تر
            if self.use_trend_filter:
                ma = data['close'].rolling(self.ma_period_trend).mean().iloc[-1]
                if not pd.isna(ma) and data['close'].iloc[-1] > ma:
                    conditions.append(f"روند صعودی (Price > MA{self.ma_period_trend})")
                else:
                    return False, []

            # 🔧 بهبود: استفاده عملیاتی از واگرایی
            if self.use_divergence and bullish_div:
                conditions.append("واگرایی مثبت")

        else:  # SHORT
            if current_rsi >= self.rsi_base_overbought:
                conditions.append(f"RSI overbought ({current_rsi:.1f} >= {self.rsi_base_overbought})")
            else:
                return False, []

            if self.use_trend_filter:
                ma = data['close'].rolling(self.ma_period_trend).mean().iloc[-1]
                if not pd.isna(ma) and data['close'].iloc[-1] < ma:
                    conditions.append(f"روند نزولی (Price < MA{self.ma_period_trend})")
                else:
                    return False, []

            if self.use_divergence and bearish_div:
                conditions.append("واگرایی منفی")

        return len(conditions) > 0, conditions

    def _update_drawdown_and_equity(self, current_time: pd.Timestamp):
        """به‌روزرسانی Drawdown و Equity Curve"""
        if self._portfolio_value > self._max_portfolio_value:
            self._max_portfolio_value = self._portfolio_value
        
        if self._max_portfolio_value > 0:
            self._current_drawdown = ((self._max_portfolio_value - self._portfolio_value) / self._max_portfolio_value) * 100
            self._max_drawdown = max(self._max_drawdown, self._current_drawdown)
        
        self._equity_curve.append((current_time, self._portfolio_value))

    def _calculate_signal_strength(
        self, 
        conditions: List[str], 
        position_type: PositionType,
        current_rsi: float
    ) -> Tuple[str, float]:
        """محاسبه قدرت سیگنال"""
        score = 1.0  # امتیاز پایه
        
        # امتیاز برای RSI عمیق
        if position_type == PositionType.LONG and current_rsi < 25:
            score += 0.3
        elif position_type == PositionType.SHORT and current_rsi > 75:
            score += 0.3
        
        # امتیاز برای واگرایی
        if self.use_divergence:
            for condition in conditions:
                if "واگرایی" in condition:
                    score += 0.5
                    break
        
        # 🔧 بهبود: امتیاز برای شرایط خاص
        for condition in conditions:
            if "روند صعودی" in condition or "روند نزولی" in condition:
                score += 0.2
                break
        
        # تبدیل امتیاز به متن
        if score >= 2.0:
            return "VERY_STRONG", score
        elif score >= 1.5:
            return "STRONG", score
        elif score >= 1.0:
            return "NORMAL", score
        else:
            return "WEAK", score

    def generate_signal(
        self, 
        data: pd.DataFrame, 
        current_index: int = 0
    ) -> Dict[str, Any]:
        """تولید سیگنال با تمام بهبودهای نسخه ۲"""
        try:
            current_time = data.index[-1]
            
            # به‌روزرسانی drawdown و equity curve
            self._update_drawdown_and_equity(current_time)
            
            # بررسی خروج
            exit_signal = self.check_exit_conditions(data, current_index)
            if exit_signal:
                return exit_signal
            
            if len(data) < 50:
                return self._create_hold_signal("داده کافی نیست")
            
            # محدودیت معاملات
            recent_trades = len([t for t in self._trade_history[-100:]])
            if recent_trades >= self.max_trades_per_100:
                return self._create_hold_signal(f"حد معاملات ({recent_trades}/{self.max_trades_per_100})")
            
            # فاصله معاملات
            if current_index - self._last_trade_index < self.min_candles_between:
                return self._create_hold_signal("فاصله کم از آخرین معامله")
            
            current_price = data['close'].iloc[-1]
            current_rsi = data['RSI'].iloc[-1]
            
            # بررسی LONG
            if self._position == PositionType.OUT:
                has_signal, conditions = self.check_entry_conditions(data, PositionType.LONG)
                
                if has_signal:
                    stop_loss, take_profit = self.calculate_dynamic_stop(data, PositionType.LONG)
                    position_size = self.calculate_position_size(current_price, stop_loss)
                    
                    if position_size > 0:
                        self._position = PositionType.LONG
                        self._current_trade = Trade(
                            entries=[TradeEntry(current_price, position_size, current_time)],
                            position_type=PositionType.LONG,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            initial_stop_loss=stop_loss,
                            trailing_stop=stop_loss  # 🔧 مقدار اولیه برای trailing stop
                        )
                        
                        self._last_trade_index = current_index
                        
                        signal_strength, score = self._calculate_signal_strength(conditions, PositionType.LONG, current_rsi)
                        
                        if self.enable_analytical_logging:
                            logger.info(f"🚀 LONG Entry: Price={current_price:.4f}, Size={position_size:.4f}, "
                                       f"SL={stop_loss:.4f}, TP={take_profit:.4f}, Strength={signal_strength} ({score:.2f})")
                        
                        return {
                            "action": "BUY",
                            "price": current_price,
                            "rsi": current_rsi,
                            "position_size": round(position_size, 4),
                            "stop_loss": round(stop_loss, 4),
                            "take_profit": round(take_profit, 4),
                            "risk_reward_ratio": round(self.base_take_profit_ratio, 2),
                            "reason": "\n".join(conditions),
                            "position": self._position.value,
                            "signal_strength": signal_strength,
                            "signal_score": round(score, 2)
                        }
            
            # بررسی SHORT
            if self._position == PositionType.OUT and self.enable_short_trades:
                has_signal, conditions = self.check_entry_conditions(data, PositionType.SHORT)
                
                if has_signal:
                    stop_loss, take_profit = self.calculate_dynamic_stop(data, PositionType.SHORT)
                    position_size = self.calculate_position_size(current_price, stop_loss)
                    
                    if position_size > 0:
                        self._position = PositionType.SHORT
                        self._current_trade = Trade(
                            entries=[TradeEntry(current_price, position_size, current_time)],
                            position_type=PositionType.SHORT,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            initial_stop_loss=stop_loss,
                            trailing_stop=stop_loss  # 🔧 مقدار اولیه برای trailing stop
                        )
                        
                        self._last_trade_index = current_index
                        
                        signal_strength, score = self._calculate_signal_strength(conditions, PositionType.SHORT, current_rsi)
                        
                        if self.enable_analytical_logging:
                            logger.info(f"🚀 SHORT Entry: Price={current_price:.4f}, Size={position_size:.4f}, "
                                       f"SL={stop_loss:.4f}, TP={take_profit:.4f}, Strength={signal_strength} ({score:.2f})")
                        
                        return {
                            "action": "SELL",
                            "price": current_price,
                            "rsi": current_rsi,
                            "position_size": round(position_size, 4),
                            "stop_loss": round(stop_loss, 4),
                            "take_profit": round(take_profit, 4),
                            "risk_reward_ratio": round(self.base_take_profit_ratio, 2),
                            "reason": "\n".join(conditions),
                            "position": self._position.value,
                            "signal_strength": signal_strength,
                            "signal_score": round(score, 2)
                        }
            
            return self._create_hold_signal(f"منتظر شرایط (RSI: {current_rsi:.1f})")
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return self._create_hold_signal(f"خطا: {str(e)}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """محاسبه معیارهای عملکرد"""
        if self._total_trades == 0:
            return {"total_trades": 0, "winning_trades": 0, "win_rate": 0}
        
        win_rate = (self._winning_trades / self._total_trades) * 100
        avg_trade = self._total_pnl / self._total_trades
        
        # محاسبه Profit Factor
        profit_factor = self._gross_profit / max(self._gross_loss, 1) if self._gross_loss > 0 else float('inf')
        
        # محاسبه Sharpe Ratio
        if len(self._daily_returns) > 1:
            daily_returns_array = np.array(self._daily_returns)
            mean_daily_return = np.mean(daily_returns_array)
            std_daily_return = np.std(daily_returns_array)
            sharpe_ratio = np.sqrt(252) * mean_daily_return / max(std_daily_return, 0.0001)
        else:
            sharpe_ratio = 0
        
        # محاسبه Recovery Factor
        recovery_factor = self._total_pnl / max(self._max_drawdown * self._max_portfolio_value / 100, 1)
        
        return {
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "losing_trades": self._total_trades - self._winning_trades,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(self._total_pnl, 2),
            "average_trade_pnl": round(avg_trade, 2),
            "current_portfolio_value": round(self._portfolio_value, 2),
            "current_position": self._position.value,
            "portfolio_return": round(((self._portfolio_value - 10000) / 10000) * 100, 2),
            "profit_factor": round(profit_factor, 2),
            "sharpe_ratio": round(sharpe_ratio, 3),
            "gross_profit": round(self._gross_profit, 2),
            "gross_loss": round(self._gross_loss, 2),
            "max_drawdown": round(self._max_drawdown, 2),
            "current_drawdown": round(self._current_drawdown, 2),
            "recovery_factor": round(recovery_factor, 2),
            "max_portfolio_value": round(self._max_portfolio_value, 2),
        }

    def get_equity_curve(self) -> List[Tuple[pd.Timestamp, float]]:
        """دریافت منحنی سرمایه"""
        return self._equity_curve.copy()

    def reset_state(self):
        """ریست کردن state استراتژی"""
        self._position = PositionType.OUT
        self._current_trade = None
        self._trade_history = []
        self._portfolio_value = 10000.0
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = 0.0
        self._gross_profit = 0.0
        self._gross_loss = 0.0
        self._last_trade_index = -100
        self._daily_returns = []
        self._last_portfolio_value = 10000.0
        self._max_portfolio_value = 10000.0
        self._max_drawdown = 0.0
        self._current_drawdown = 0.0
        self._equity_curve = []