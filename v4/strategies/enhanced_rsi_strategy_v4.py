# strategies/enhanced_rsi_strategy_v4.py - FIXED VERSION

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
    VOLATILITY_EXIT = "VOLATILITY_EXIT"

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
    highest_profit: float = 0.0  # 🔥 NEW: Track highest profit
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_reason: Optional[ExitReason] = None
    pnl_percentage: Optional[float] = None
    pnl_amount: Optional[float] = None
    partial_exits: List[Dict] = field(default_factory=list)
    partial_exit_done: bool = False
    entry_conditions: List[str] = field(default_factory=list)
    
    @property
    def entry_price(self) -> float:
        if not self.entries:
            return 0.0
        total_cost = sum(entry.price * entry.quantity for entry in self.entries)
        total_quantity = sum(entry.quantity for entry in self.entries)
        return total_cost / total_quantity if total_quantity > 0 else 0.0
    
    @property
    def quantity(self) -> float:
        return sum(entry.quantity for entry in self.entries)

class EnhancedRsiStrategyV4:
    """
    🔥 OPTIMIZED VERSION - بهینه‌شده برای سودآوری بالاتر
    """
    
    def __init__(
        self,
        # Core RSI - بهینه‌شده
        rsi_period: int = 14,  # 🔥 Changed: از 11 به 14 (استاندارد)
        rsi_oversold: int = 35,  # 🔥 Changed: از 28 به 35 (تعادل بهتر)
        rsi_overbought: int = 65,  # 🔥 Changed: از 72 به 65
        rsi_entry_buffer: int = 5,  # 🔥 Changed: از 2 به 5 (انعطاف بیشتر)
        
        # Risk Management - بهبود یافته
        risk_per_trade: float = 0.015,  # 🔥 Changed: از 0.008 به 0.015
        stop_loss_atr_multiplier: float = 2.0,  # 🔥 Changed: از 1.8 به 2.0
        take_profit_ratio: float = 2.5,  # 🔥 Changed: از 2.2 به 2.5
        min_position_size: float = 100,  # 🔥 Changed: از 800 به 100
        max_position_size_ratio: float = 0.3,  # 🔥 Changed: از 0.25 به 0.3
        
        # Trade Control - منطقی‌تر
        max_trades_per_100: int = 30,  # 🔥 Changed: از 20 به 30
        min_candles_between: int = 5,  # 🔥 Changed: از 8 به 5
        max_trade_duration: int = 100,  # 🔥 Changed: از 75 به 100
        
        # Filters - انعطاف‌پذیرتر
        enable_trend_filter: bool = False,  # 🔥 Changed: غیرفعال شد
        trend_strength_threshold: float = 0.005,  # 🔥 Changed: کاهش یافت
        enable_volume_filter: bool = False,
        enable_volatility_filter: bool = False,  # 🔥 Changed: غیرفعال شد
        enable_short_trades: bool = True,
        
        # Advanced - بهینه‌شده
        enable_trailing_stop: bool = True,
        trailing_activation_percent: float = 1.0,  # 🔥 Changed: از 0.4 به 1.0%
        trailing_stop_atr_multiplier: float = 1.5,  # 🔥 Changed: از 1.0 به 1.5
        enable_partial_exit: bool = True,
        partial_exit_ratio: float = 0.5,
        partial_exit_threshold: float = 1.5,  # 🔥 Changed: از 0.8 به 1.5%
        
        # Loss Control - متعادل‌تر
        max_consecutive_losses: int = 4,  # 🔥 Changed: از 3 به 4
        pause_after_losses: int = 10,  # 🔥 Changed: از 20 به 10
        risk_reduction_after_loss: bool = False,  # 🔥 Changed: غیرفعال
        
        # Confirmations - ساده‌تر
        require_rsi_confirmation: bool = False,  # 🔥 Changed: غیرفعال
        require_price_confirmation: bool = False,  # 🔥 Changed: غیرفعال
        confirmation_candles: int = 1,  # 🔥 Changed: از 2 به 1

        # Multi-Timeframe Analysis (MTF)
        enable_mtf: bool = True,
        mtf_timeframes: Optional[List[str]] = None,   # e.g., ['H4','D1']
        mtf_require_all: bool = True,                 # require all HTFs to align, else any-one
        mtf_long_rsi_min: float = 50.0,               # min RSI on HTFs for LONG
        mtf_short_rsi_max: float = 50.0,              # max RSI on HTFs for SHORT
        mtf_trend_ema_fast: int = 21,                 # fast EMA used for HTF trend
        mtf_trend_ema_slow: int = 50                  # slow EMA used for HTF trend
    ):
        # Initialize all parameters
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.rsi_entry_buffer = rsi_entry_buffer
        
        self.risk_per_trade = risk_per_trade
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_ratio = take_profit_ratio
        self.min_position_size = min_position_size
        self.max_position_size_ratio = max_position_size_ratio
        
        self.max_trades_per_100 = max_trades_per_100
        self.min_candles_between = min_candles_between
        self.max_trade_duration = max_trade_duration
        
        self.enable_trend_filter = enable_trend_filter
        self.trend_strength_threshold = trend_strength_threshold
        self.enable_volume_filter = enable_volume_filter
        self.enable_volatility_filter = enable_volatility_filter
        self.enable_short_trades = enable_short_trades
        
        self.enable_trailing_stop = enable_trailing_stop
        self.trailing_activation_percent = trailing_activation_percent
        self.trailing_stop_atr_multiplier = trailing_stop_atr_multiplier
        self.enable_partial_exit = enable_partial_exit
        self.partial_exit_ratio = partial_exit_ratio
        self.partial_exit_threshold = partial_exit_threshold
        
        self.max_consecutive_losses = max_consecutive_losses
        self.pause_after_losses = pause_after_losses
        self.risk_reduction_after_loss = risk_reduction_after_loss
        
        self.require_rsi_confirmation = require_rsi_confirmation
        self.require_price_confirmation = require_price_confirmation
        self.confirmation_candles = confirmation_candles

        # MTF configuration
        self.enable_mtf = enable_mtf
        self.mtf_timeframes = mtf_timeframes or ['H4', 'D1']
        self.mtf_require_all = mtf_require_all
        self.mtf_long_rsi_min = mtf_long_rsi_min
        self.mtf_short_rsi_max = mtf_short_rsi_max
        self.mtf_trend_ema_fast = mtf_trend_ema_fast
        self.mtf_trend_ema_slow = mtf_trend_ema_slow
        
        # State variables
        self._position = PositionType.OUT
        self._current_trade = None
        self._trade_history = []
        self._portfolio_value = 10000.0
        self._last_trade_index = -100
        self._consecutive_losses = 0
        self._pause_until_index = -1
        self._original_risk = risk_per_trade
        
        # Statistics
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = 0.0
        self._gross_profit = 0.0
        self._gross_loss = 0.0
        
        # Logs
        self._signal_log = []
        self._market_condition = "UNKNOWN"
        
        logger.info(f"🔥 OPTIMIZED RSI Strategy V4 - RSI({rsi_period}), Risk: {risk_per_trade*100}%")

    def _calculate_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """محاسبه RSI با مدیریت خطا"""
        try:
            if len(data) < self.rsi_period + 1:
                return data
                
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.ewm(alpha=1/self.rsi_period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/self.rsi_period, adjust=False).mean()
            
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))
            data['RSI'] = data['RSI'].fillna(method='bfill').fillna(50)
            
            return data
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            data['RSI'] = 50
            return data

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """محاسبه ATR"""
        try:
            if len(data) < period + 1:
                return data['close'].iloc[-1] * 0.015  # 🔥 Changed: از 0.01 به 0.015
            
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            if pd.isna(atr) or atr <= 0:
                atr = data['close'].iloc[-1] * 0.015
                
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return data['close'].iloc[-1] * 0.015

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """🔥 IMPROVED: محاسبه Position Size بهبود یافته"""
        try:
            # 🔥 کاهش ریسک فقط اگر فعال باشد
            current_risk = self.risk_per_trade
            if self.risk_reduction_after_loss and self._consecutive_losses > 0:
                risk_reduction = max(0.7, 1.0 - (self._consecutive_losses * 0.05))
                current_risk = self._original_risk * risk_reduction
            
            risk_amount = self._portfolio_value * current_risk
            
            # محاسبه فاصله قیمتی
            if stop_loss < entry_price:  # LONG
                price_risk = entry_price - stop_loss
            else:  # SHORT
                price_risk = stop_loss - entry_price
            
            # 🔥 اعتبارسنجی منطقی‌تر
            if price_risk <= 0 or price_risk > entry_price * 0.1:  # Changed: از 0.05 به 0.1
                logger.warning(f"Invalid price risk: {price_risk:.6f}")
                return 0
            
            position_size = risk_amount / price_risk
            
            # 🔥 محدودیت‌های واقع‌بینانه‌تر
            max_position = self._portfolio_value * self.max_position_size_ratio
            position_size = min(position_size, max_position)
            
            if position_size < self.min_position_size:
                return 0
                
            logger.info(f"Position Size: {position_size:.0f} (Risk: ${risk_amount:.2f})")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0

    def check_entry_conditions(self, data: pd.DataFrame, position_type: PositionType) -> Tuple[bool, List[str]]:
        """Enhanced entry conditions with optional Trend/Price/RSI confirmations + MTF"""
        conditions: List[str] = []
        try:
            # Ensure RSI present for base timeframe
            if 'RSI' not in data.columns:
                data = self._calculate_rsi(data)

            current_rsi = float(data['RSI'].iloc[-1]) if 'RSI' in data.columns else 50.0

            # Base RSI gate
            if position_type == PositionType.LONG:
                if not (current_rsi <= (self.rsi_oversold + self.rsi_entry_buffer)):
                    return False, [f"RSI not suitable for LONG ({current_rsi:.1f})"]
                conditions.append(f"RSI in BUY zone ({current_rsi:.1f})")
            elif position_type == PositionType.SHORT:
                if not (current_rsi >= (self.rsi_overbought - self.rsi_entry_buffer)):
                    return False, [f"RSI not suitable for SHORT ({current_rsi:.1f})"]
                conditions.append(f"RSI in SELL zone ({current_rsi:.1f})")

            # Optional RSI slope confirmation
            if self.require_rsi_confirmation and len(data['RSI']) >= 2:
                prev_rsi = float(data['RSI'].iloc[-2])
                if position_type == PositionType.LONG and not (current_rsi >= prev_rsi):
                    return False, [f"RSI slope not rising ({prev_rsi:.1f}->{current_rsi:.1f})"]
                if position_type == PositionType.SHORT and not (current_rsi <= prev_rsi):
                    return False, [f"RSI slope not falling ({prev_rsi:.1f}->{current_rsi:.1f})"]
                conditions.append("RSI slope confirmed")

            # Optional Trend filter using EMA alignment and ADX if available
            if self.enable_trend_filter:
                ema_ok = True
                try:
                    ema21 = float(data['EMA_21'].iloc[-1]) if 'EMA_21' in data.columns else None
                    ema50 = float(data['EMA_50'].iloc[-1]) if 'EMA_50' in data.columns else None
                    if ema21 is not None and ema50 is not None:
                        if position_type == PositionType.LONG:
                            ema_ok = ema21 >= ema50
                        else:
                            ema_ok = ema21 <= ema50
                except Exception:
                    ema_ok = True

                adx_ok = True
                try:
                    if 'ADX' in data.columns:
                        adx_val = float(data['ADX'].iloc[-1])
                        adx_ok = adx_val >= 12.0
                except Exception:
                    adx_ok = True

                if not (ema_ok and adx_ok):
                    reason_parts = []
                    if not ema_ok:
                        reason_parts.append("EMA21-EMA50 misaligned")
                    if not adx_ok:
                        reason_parts.append("ADX<12")
                    return False, [f"Trend filter: {' & '.join(reason_parts)}"]
                conditions.append("Trend filter passed")

            # Optional price confirmation using EMA9 and candle direction
            if self.require_price_confirmation:
                price_ok = True
                try:
                    close = float(data['close'].iloc[-1])
                    open_ = float(data['open'].iloc[-1]) if 'open' in data.columns else close
                    ema9 = float(data['EMA_9'].iloc[-1]) if 'EMA_9' in data.columns else None
                    if ema9 is not None:
                        if position_type == PositionType.LONG:
                            price_ok = (close > ema9) and (close > open_)
                        else:
                            price_ok = (close < ema9) and (close < open_)
                except Exception:
                    price_ok = True
                if not price_ok:
                    return False, [f"Price confirmation failed"]
                conditions.append("Price confirmation passed")

            # Multi-Timeframe alignment gate (uses columns if available)
            if self.enable_mtf:
                mtf_ok, mtf_msgs = self._check_mtf_alignment(data, position_type)
                if not mtf_ok:
                    return False, [f"MTF filter: {' | '.join(mtf_msgs)}"]
                # annotate reasons for transparency
                for msg in mtf_msgs:
                    conditions.append(f"MTF: {msg}")

            # Distance from last trade
            candles_since_last = len(data) - 1 - self._last_trade_index
            if candles_since_last < self.min_candles_between:
                return False, [f"Too close to last trade ({candles_since_last} candles)"]
            conditions.append(f"Gap from last trade: {candles_since_last} candles")

            # Pause after consecutive losses
            if len(data) - 1 <= self._pause_until_index:
                return False, [f"Paused after {self._consecutive_losses} losses"]

            return True, conditions

        except Exception as e:
            logger.error(f"Error checking entry conditions: {e}")
            return False, [f"Error: {e}"]

    def _check_mtf_alignment(self, data: pd.DataFrame, position_type: PositionType) -> Tuple[bool, List[str]]:
        """
        بررسی هم‌جهتی تایم‌فریم‌های بالاتر.
        - از ستون‌های از پیش محاسبه‌شده استفاده می‌کند:
          RSI_{TF}, EMA_21_{TF}, EMA_50_{TF}, TrendDir_{TF} (1=UP, -1=DOWN, 0=FLAT)
        - اگر هیچ داده‌ای وجود نداشته باشد، عبور می‌دهد (اثر ندارد).
        """
        messages: List[str] = []
        available = 0
        passed = 0

        # Which HTFs to inspect
        timeframes = self.mtf_timeframes or []

        for tf in timeframes:
            rsi_col = f'RSI_{tf}'
            ema_fast_col = f'EMA_21_{tf}'
            ema_slow_col = f'EMA_50_{tf}'
            trend_col = f'TrendDir_{tf}'

            has_rsi = rsi_col in data.columns and not pd.isna(data[rsi_col].iloc[-1])
            has_ema = (
                ema_fast_col in data.columns and ema_slow_col in data.columns and
                not pd.isna(data[ema_fast_col].iloc[-1]) and not pd.isna(data[ema_slow_col].iloc[-1])
            )
            has_trend = trend_col in data.columns and not pd.isna(data[trend_col].iloc[-1])

            # Skip this TF if no usable signals
            if not (has_rsi or has_ema or has_trend):
                continue

            available += 1
            tf_ok = True
            tf_msgs: List[str] = []

            if position_type == PositionType.LONG:
                if has_rsi and float(data[rsi_col].iloc[-1]) < self.mtf_long_rsi_min:
                    tf_ok = False
                    tf_msgs.append(f"{tf}: RSI {float(data[rsi_col].iloc[-1]):.1f} < {self.mtf_long_rsi_min}")
                if has_ema and not (float(data[ema_fast_col].iloc[-1]) >= float(data[ema_slow_col].iloc[-1])):
                    tf_ok = False
                    tf_msgs.append(f"{tf}: EMA{self.mtf_trend_ema_fast} < EMA{self.mtf_trend_ema_slow}")
                if has_trend and int(data[trend_col].iloc[-1]) < 0:
                    tf_ok = False
                    tf_msgs.append(f"{tf}: Downtrend")
            else:  # SHORT
                if has_rsi and float(data[rsi_col].iloc[-1]) > self.mtf_short_rsi_max:
                    tf_ok = False
                    tf_msgs.append(f"{tf}: RSI {float(data[rsi_col].iloc[-1]):.1f} > {self.mtf_short_rsi_max}")
                if has_ema and not (float(data[ema_fast_col].iloc[-1]) <= float(data[ema_slow_col].iloc[-1])):
                    tf_ok = False
                    tf_msgs.append(f"{tf}: EMA{self.mtf_trend_ema_fast} > EMA{self.mtf_trend_ema_slow}")
                if has_trend and int(data[trend_col].iloc[-1]) > 0:
                    tf_ok = False
                    tf_msgs.append(f"{tf}: Uptrend")

            if tf_ok:
                passed += 1
                messages.append(f"{tf} aligned")
            else:
                messages.append(" | ".join(tf_msgs) if tf_msgs else f"{tf} misaligned")

        # No MTF data -> do not block entries
        if available == 0:
            return True, ["No MTF data - skipped"]

        if self.mtf_require_all:
            return (passed == available), messages
        else:
            return (passed > 0), messages

    def check_exit_conditions(self, data: pd.DataFrame, current_index: int) -> Optional[Dict[str, Any]]:
        """🔥 IMPROVED: شرایط خروج بهبود یافته"""
        if self._position == PositionType.OUT or self._current_trade is None:
            return None

        try:
            current_price = data['close'].iloc[-1]
            current_time = data.index[-1]
            entry_price = self._current_trade.entry_price
            
            # محاسبه سود/زیان
            if self._position == PositionType.LONG:
                profit_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                profit_pct = ((entry_price - current_price) / entry_price) * 100
            
            # 🔥 Track highest profit
            if profit_pct > self._current_trade.highest_profit:
                self._current_trade.highest_profit = profit_pct
            
            # 🔥 خروج جزئی بهبود یافته
            if (self.enable_partial_exit and 
                profit_pct >= self.partial_exit_threshold and 
                not self._current_trade.partial_exit_done):
                
                self._current_trade.partial_exit_done = True
                partial_quantity = self._current_trade.quantity * self.partial_exit_ratio
                partial_pnl = (profit_pct / 100) * partial_quantity * entry_price
                self._portfolio_value += partial_pnl
                
                # Update trailing stop after partial exit
                if self.enable_trailing_stop:
                    atr = self.calculate_atr(data)
                    if self._position == PositionType.LONG:
                        self._current_trade.trailing_stop = current_price - (atr * self.trailing_stop_atr_multiplier)
                    else:
                        self._current_trade.trailing_stop = current_price + (atr * self.trailing_stop_atr_multiplier)
                
                logger.info(f"✅ Partial exit: {self.partial_exit_ratio*100}% at profit {profit_pct:.2f}%")
                
                return {
                    "action": "PARTIAL_EXIT",
                    "price": current_price,
                    "quantity": partial_quantity,
                    "pnl_percentage": profit_pct,
                    "pnl_amount": partial_pnl,
                    "reason": "PARTIAL_TAKE_PROFIT"
                }
            
            # Stop Loss
            if self._position == PositionType.LONG and current_price <= self._current_trade.stop_loss:
                return self._create_exit_signal("STOP_LOSS", current_price, current_time)
            elif self._position == PositionType.SHORT and current_price >= self._current_trade.stop_loss:
                return self._create_exit_signal("STOP_LOSS", current_price, current_time)
            
            # Take Profit
            if self._position == PositionType.LONG and current_price >= self._current_trade.take_profit:
                return self._create_exit_signal("TAKE_PROFIT", current_price, current_time)
            elif self._position == PositionType.SHORT and current_price <= self._current_trade.take_profit:
                return self._create_exit_signal("TAKE_PROFIT", current_price, current_time)
            
            # 🔥 Trailing Stop بهبود یافته
            if self.enable_trailing_stop and profit_pct >= self.trailing_activation_percent:
                atr = self.calculate_atr(data)
                trailing_atr = atr * self.trailing_stop_atr_multiplier
                
                if self._position == PositionType.LONG:
                    new_trailing = current_price - trailing_atr
                    if new_trailing > self._current_trade.trailing_stop:
                        self._current_trade.trailing_stop = new_trailing
                        logger.debug(f"Trailing stop updated: {new_trailing:.4f}")
                    
                    if current_price <= self._current_trade.trailing_stop:
                        return self._create_exit_signal("TRAILING_STOP", current_price, current_time)
                
                else:  # SHORT
                    new_trailing = current_price + trailing_atr
                    if new_trailing < self._current_trade.trailing_stop or self._current_trade.trailing_stop == self._current_trade.stop_loss:
                        self._current_trade.trailing_stop = new_trailing
                        logger.debug(f"Trailing stop updated: {new_trailing:.4f}")
                    
                    if current_price >= self._current_trade.trailing_stop:
                        return self._create_exit_signal("TRAILING_STOP", current_price, current_time)
            
            # Time Exit
            trade_duration = current_index - self._last_trade_index
            if trade_duration >= self.max_trade_duration:
                return self._create_exit_signal("TIME_EXIT", current_price, current_time)
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return None

    def _create_exit_signal(self, exit_reason: str, price: float, time: pd.Timestamp) -> Dict[str, Any]:
        """ایجاد سیگنال خروج"""
        try:
            if self._current_trade is None:
                return {"action": "HOLD", "reason": "No active trade"}
            
            entry_price = self._current_trade.entry_price
            quantity = self._current_trade.quantity
            
            if self._position == PositionType.LONG:
                pnl_amount = (price - entry_price) * quantity
            else:
                pnl_amount = (entry_price - price) * quantity
            
            pnl_percentage = (pnl_amount / (entry_price * quantity)) * 100
            
            # Update portfolio
            self._portfolio_value += pnl_amount
            self._total_pnl += pnl_amount
            self._total_trades += 1
            
            # Track wins/losses
            if pnl_amount < 0:
                self._consecutive_losses += 1
                self._gross_loss += abs(pnl_amount)
                if self._consecutive_losses >= self.max_consecutive_losses:
                    self._pause_until_index = len(self._trade_history) + self.pause_after_losses
                    logger.warning(f"⚠️ {self._consecutive_losses} consecutive losses - pausing for {self.pause_after_losses} candles")
            else:
                self._consecutive_losses = 0
                self._winning_trades += 1
                self._gross_profit += pnl_amount
            
            # Record trade
            self._current_trade.exit_price = price
            self._current_trade.exit_time = time
            self._current_trade.pnl_percentage = pnl_percentage
            self._current_trade.pnl_amount = pnl_amount
            self._current_trade.exit_reason = ExitReason(exit_reason)
            
            old_position = self._position
            self._trade_history.append(self._current_trade)
            self._position = PositionType.OUT
            self._current_trade = None
            
            logger.info(f"🔚 EXIT at {price:.4f}, PnL: {pnl_percentage:.2f}%, Reason: {exit_reason}")
            
            return {
                "action": "EXIT",
                "price": float(price),
                "exit_reason": exit_reason,
                "pnl_percentage": round(pnl_percentage, 2),
                "pnl_amount": round(pnl_amount, 2),
                "position": "OUT",
                "previous_position": old_position.value
            }
            
        except Exception as e:
            logger.error(f"Error creating exit signal: {e}")
            return {"action": "HOLD", "reason": f"Exit error: {e}"}

    def calculate_stop_take_profit(self, data: pd.DataFrame, position_type: PositionType, entry_price: float) -> Tuple[float, float]:
        """محاسبه Stop Loss و Take Profit"""
        try:
            atr = self.calculate_atr(data)
            
            sl_multiplier = self.stop_loss_atr_multiplier
            tp_multiplier = self.stop_loss_atr_multiplier * self.take_profit_ratio
            
            if position_type == PositionType.LONG:
                stop_loss = entry_price - (atr * sl_multiplier)
                take_profit = entry_price + (atr * tp_multiplier)
            else:  # SHORT
                stop_loss = entry_price + (atr * sl_multiplier)
                take_profit = entry_price - (atr * tp_multiplier)
            
            # Validation
            if position_type == PositionType.LONG:
                if stop_loss >= entry_price:
                    stop_loss = entry_price * 0.985
                if take_profit <= entry_price:
                    take_profit = entry_price * 1.04
            else:
                if stop_loss <= entry_price:
                    stop_loss = entry_price * 1.015
                if take_profit >= entry_price:
                    take_profit = entry_price * 0.96
            
            logger.info(f"SL: {stop_loss:.4f}, TP: {take_profit:.4f}")
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating SL/TP: {e}")
            if position_type == PositionType.LONG:
                return entry_price * 0.985, entry_price * 1.04
            else:
                return entry_price * 1.015, entry_price * 0.96

    def generate_signal(self, data: pd.DataFrame, current_index: int = 0) -> Dict[str, Any]:
        """تولید سیگنال + ثبت در لاگ سیگنال"""
        try:
            # محاسبه RSI در صورت نیاز برای لاگ
            if 'RSI' not in data.columns:
                try:
                    data = self._calculate_rsi(data)
                except Exception:
                    pass

            current_time = data.index[-1]
            current_price = data['close'].iloc[-1]
            current_rsi = float(data['RSI'].iloc[-1]) if 'RSI' in data.columns else None

            # ابتدا بررسی شرایط خروج (EXIT / PARTIAL_EXIT / TRAILING_STOP ...)
            exit_signal = self.check_exit_conditions(data, current_index)
            if exit_signal:
                # ثبت لاگ سیگنال خروج
                self._signal_log.append({
                    "time": current_time,
                    "action": exit_signal.get("action", "EXIT"),
                    "price": float(current_price),
                    "position": self._position.value if hasattr(self, "_position") else "OUT",
                    "reason": exit_signal.get("reason") or exit_signal.get("exit_reason", ""),
                    "pnl_percentage": exit_signal.get("pnl_percentage"),
                    "rsi": current_rsi
                })
                return exit_signal

            # بررسی‌های پایه
            if len(data) < 50:
                hold = {"action": "HOLD", "reason": "Insufficient data"}
                self._signal_log.append({
                    "time": current_time,
                    "action": "HOLD",
                    "price": float(current_price),
                    "position": self._position.value,
                    "reason": hold["reason"],
                    "rsi": current_rsi
                })
                return hold

            if current_index <= self._pause_until_index:
                hold = {"action": "HOLD", "reason": "Paused after losses"}
                self._signal_log.append({
                    "time": current_time,
                    "action": "HOLD",
                    "price": float(current_price),
                    "position": self._position.value,
                    "reason": hold["reason"],
                    "rsi": current_rsi
                })
                return hold

            # تلاش برای ورود LONG
            if self._position == PositionType.OUT:
                long_ok, long_conditions = self.check_entry_conditions(data, PositionType.LONG)
                if long_ok:
                    stop_loss, take_profit = self.calculate_stop_take_profit(data, PositionType.LONG, current_price)
                    position_size = self.calculate_position_size(current_price, stop_loss)

                    if position_size > 0:
                        self._position = PositionType.LONG
                        self._current_trade = Trade(
                            entries=[TradeEntry(current_price, position_size, data.index[-1])],
                            position_type=PositionType.LONG,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            initial_stop_loss=stop_loss,
                            trailing_stop=stop_loss,
                            entry_conditions=long_conditions
                        )
                        self._last_trade_index = current_index

                        logger.info(f"🟢 BUY at {current_price:.4f}, Size: {position_size:.0f}")

                        signal = {
                            "action": "BUY",
                            "price": float(current_price),
                            "position_size": float(position_size),
                            "stop_loss": float(stop_loss),
                            "take_profit": float(take_profit),
                            "reason": " | ".join(long_conditions),
                            "position": "LONG"
                        }

                        # ثبت لاگ سیگنال ورود
                        self._signal_log.append({
                            "time": current_time,
                            "action": "BUY",
                            "price": float(current_price),
                            "position": "LONG",
                            "stop_loss": float(stop_loss),
                            "take_profit": float(take_profit),
                            "rsi": current_rsi,
                            "reason": " | ".join(long_conditions)
                        })

                        return signal

            # تلاش برای ورود SHORT
            if self._position == PositionType.OUT and self.enable_short_trades:
                short_ok, short_conditions = self.check_entry_conditions(data, PositionType.SHORT)
                if short_ok:
                    stop_loss, take_profit = self.calculate_stop_take_profit(data, PositionType.SHORT, current_price)
                    position_size = self.calculate_position_size(current_price, stop_loss)

                    if position_size > 0:
                        self._position = PositionType.SHORT
                        self._current_trade = Trade(
                            entries=[TradeEntry(current_price, position_size, data.index[-1])],
                            position_type=PositionType.SHORT,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            initial_stop_loss=stop_loss,
                            trailing_stop=stop_loss,
                            entry_conditions=short_conditions
                        )
                        self._last_trade_index = current_index

                        logger.info(f"🔴 SELL at {current_price:.4f}, Size: {position_size:.0f}")

                        signal = {
                            "action": "SELL",
                            "price": float(current_price),
                            "position_size": float(position_size),
                            "stop_loss": float(stop_loss),
                            "take_profit": float(take_profit),
                            "reason": " | ".join(short_conditions),
                            "position": "SHORT"
                        }

                        # ثبت لاگ سیگنال فروش
                        self._signal_log.append({
                            "time": current_time,
                            "action": "SELL",
                            "price": float(current_price),
                            "position": "SHORT",
                            "stop_loss": float(stop_loss),
                            "take_profit": float(take_profit),
                            "rsi": current_rsi,
                            "reason": " | ".join(short_conditions)
                        })

                        return signal

            # پیش‌فرض: HOLD
            hold = {"action": "HOLD", "reason": "Waiting for conditions"}
            self._signal_log.append({
                "time": current_time,
                "action": "HOLD",
                "price": float(current_price),
                "position": self._position.value,
                "reason": hold["reason"],
                "rsi": current_rsi
            })
            return hold

        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return {"action": "HOLD", "reason": f"Error: {e}"}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """دریافت معیارهای عملکرد"""
        if self._total_trades == 0:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "current_portfolio_value": self._portfolio_value,
                "consecutive_losses": self._consecutive_losses
            }
        
        win_rate = (self._winning_trades / self._total_trades) * 100
        profit_factor = self._gross_profit / max(self._gross_loss, 1)
        
        if self._total_trades > 1:
            trade_returns = [t.pnl_percentage for t in self._trade_history if t.pnl_percentage is not None]
            if trade_returns:
                avg_return = np.mean(trade_returns)
                volatility = np.std(trade_returns)
                sharpe_ratio = avg_return / max(volatility, 0.001)
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        return {
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "losing_trades": self._total_trades - self._winning_trades,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(self._total_pnl, 2),
            "gross_profit": round(self._gross_profit, 2),
            "gross_loss": round(self._gross_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "current_portfolio_value": round(self._portfolio_value, 2),
            "portfolio_return": round(((self._portfolio_value - 10000) / 10000) * 100, 2),
            "consecutive_losses": self._consecutive_losses,
            "current_position": self._position.value,
            "sharpe_ratio": round(sharpe_ratio, 2),
            "avg_trade_return": round(self._total_pnl / max(self._total_trades, 1), 2)
        }

    def get_trade_history(self) -> List[Trade]:
        """دریافت تاریخچه معاملات"""
        return self._trade_history.copy()

    def get_signal_log(self) -> List[Dict[str, Any]]:
        """دریافت لاگ سیگنال‌ها"""
        return self._signal_log.copy()
    
    def reset_state(self):
        """ریست استراتژی"""
        self._position = PositionType.OUT
        self._current_trade = None
        self._trade_history = []
        self._portfolio_value = 10000.0
        self._last_trade_index = -100
        self._consecutive_losses = 0
        self._pause_until_index = -1
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = 0.0
        self._gross_profit = 0.0
        self._gross_loss = 0.0
        self._signal_log = []
        self._original_risk = self.risk_per_trade
        
        logger.info("✅ Strategy V4 reset completed")