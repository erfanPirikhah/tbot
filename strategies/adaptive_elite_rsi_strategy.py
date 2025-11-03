# strategies/adaptive_rsi_strategy_realistic.py
"""
استراتژی RSI اصلاح شده - نسخه نهایی
🎯 هدف: اجرای معاملات با منطق معکوس

منطق جدید:
- LONG: RSI پایین (اشباع فروش) بدون در نظر گرفتن روند
- SHORT: RSI بالا (اشباع خرید) بدون در نظر گرفتن روند
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

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

@dataclass
class Trade:
    entry_price: float
    entry_time: pd.Timestamp
    position_type: PositionType
    quantity: float
    stop_loss: float
    take_profit: float
    initial_stop_loss: float = 0.0
    trailing_stop: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_reason: Optional[ExitReason] = None
    pnl_percentage: Optional[float] = None
    pnl_amount: Optional[float] = None

class ProfessionalAdvancedRsiStrategy:
    """
    استراتژی RSI با منطق معکوس ساده
    
    قوانین جدید:
    LONG: RSI < 35 (فقط اشباع فروش)
    SHORT: RSI > 65 (فقط اشباع خرید)
    
    حذف شرط روند برای افزایش تعداد معاملات
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        rsi_base_oversold: int = 35,  # 🔧 کاهش به 35
        rsi_base_overbought: int = 65,  # 🔧 افزایش به 65
        risk_per_trade: float = 0.02,
        base_stop_atr_multiplier: float = 2.0,
        base_take_profit_ratio: float = 2.0,  # 🔧 کاهش به 2.0
        max_trade_duration: int = 48,
        enable_short_trades: bool = True,
        use_dynamic_trailing: bool = True,
        max_trades_per_100: int = 20,  # 🔧 کاهش برای جلوگیری از معاملات زیاد
        min_candles_between: int = 5,  # 🔧 افزایش فاصله
    ):
        self.rsi_period = rsi_period
        self.rsi_base_oversold = rsi_base_oversold
        self.rsi_base_overbought = rsi_base_overbought
        self.risk_per_trade = risk_per_trade
        self.base_stop_atr_multiplier = base_stop_atr_multiplier
        self.base_take_profit_ratio = base_take_profit_ratio
        self.max_trade_duration = max_trade_duration
        self.enable_short_trades = enable_short_trades
        self.use_dynamic_trailing = use_dynamic_trailing
        self.max_trades_per_100 = max_trades_per_100
        self.min_candles_between = min_candles_between
        
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

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        try:
            high = data['high']
            low = data['low']
            close_prev = data['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close_prev)
            tr3 = abs(low - close_prev)
            
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr = tr.rolling(period).mean().iloc[-1]
            
            return float(atr) if not np.isnan(atr) else 0.0
        except:
            return 0.0

    def check_simple_conditions(
        self, 
        data: pd.DataFrame, 
        position_type: PositionType
    ) -> Tuple[bool, List[str]]:
        """
        🎯 شرایط ساده شده - فقط RSI
        
        LONG: RSI < 35 (فقط اشباع فروش)
        SHORT: RSI > 65 (فقط اشباع خرید)
        """
        conditions = []
        current_rsi = data['RSI'].iloc[-1]
        
        # ============================
        # شرایط LONG (فقط RSI)
        # ============================
        if position_type == PositionType.LONG:
            if current_rsi < self.rsi_base_oversold:  # 35
                conditions.append(f"✅ RSI اشباع فروش ({current_rsi:.1f})")
                return True, conditions
            else:
                return False, [f"❌ RSI مناسب نیست ({current_rsi:.1f} >= {self.rsi_base_oversold})"]
        
        # ============================
        # شرایط SHORT (فقط RSI)
        # ============================
        else:
            if current_rsi > self.rsi_base_overbought:  # 65
                conditions.append(f"✅ RSI اشباع خرید ({current_rsi:.1f})")
                return True, conditions
            else:
                return False, [f"❌ RSI مناسب نیست ({current_rsi:.1f} <= {self.rsi_base_overbought})"]

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        risk_amount = self._portfolio_value * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        position_size = risk_amount / price_risk
        max_position = self._portfolio_value * 0.2
        
        return min(position_size, max_position)

    def check_exit_conditions(self, data: pd.DataFrame, current_index: int) -> Optional[Dict[str, Any]]:
        if not self._current_trade or self._position == PositionType.OUT:
            return None
        
        current_price = data['close'].iloc[-1]
        current_rsi = data['RSI'].iloc[-1]
        current_time = data.index[-1] if hasattr(data.index, '__getitem__') else pd.Timestamp.now()
        
        is_long = self._current_trade.position_type == PositionType.LONG
        
        # Stop Loss
        if is_long:
            if current_price <= self._current_trade.stop_loss:
                return self._exit_trade(current_price, current_time, ExitReason.STOP_LOSS)
        else:
            if current_price >= self._current_trade.stop_loss:
                return self._exit_trade(current_price, current_time, ExitReason.STOP_LOSS)
        
        # Take Profit
        if is_long:
            if current_price >= self._current_trade.take_profit:
                return self._exit_trade(current_price, current_time, ExitReason.TAKE_PROFIT)
        else:
            if current_price <= self._current_trade.take_profit:
                return self._exit_trade(current_price, current_time, ExitReason.TAKE_PROFIT)
        
        # Smart Exit با RSI
        if is_long and current_rsi > 70:  # خروج LONG در اشباع خرید
            return self._exit_trade(current_price, current_time, ExitReason.SIGNAL_EXIT)
        elif not is_long and current_rsi < 30:  # خروج SHORT در اشباع فروش
            return self._exit_trade(current_price, current_time, ExitReason.SIGNAL_EXIT)
        
        # Time Exit
        if self._current_trade.entry_time:
            duration = (current_time - self._current_trade.entry_time).total_seconds() / 3600
            if duration >= self.max_trade_duration:
                return self._exit_trade(current_price, current_time, ExitReason.TIME_EXIT)
        
        return None

    def _exit_trade(self, exit_price: float, exit_time: pd.Timestamp, reason: ExitReason) -> Dict[str, Any]:
        is_long = self._current_trade.position_type == PositionType.LONG
        
        if is_long:
            pnl_percentage = ((exit_price - self._current_trade.entry_price) / 
                             self._current_trade.entry_price * 100)
            pnl_amount = (exit_price - self._current_trade.entry_price) * self._current_trade.quantity
        else:
            pnl_percentage = ((self._current_trade.entry_price - exit_price) / 
                             self._current_trade.entry_price * 100)
            pnl_amount = (self._current_trade.entry_price - exit_price) * self._current_trade.quantity
        
        self._portfolio_value += pnl_amount
        self._total_trades += 1
        
        if pnl_amount > 0:
            self._winning_trades += 1
        
        self._total_pnl += pnl_amount
        
        self._current_trade.exit_price = exit_price
        self._current_trade.exit_time = exit_time
        self._current_trade.exit_reason = reason
        self._current_trade.pnl_percentage = pnl_percentage
        self._current_trade.pnl_amount = pnl_amount
        
        self._trade_history.append(self._current_trade)
        self._position = PositionType.OUT
        
        result = {
            "action": "SELL" if is_long else "COVER",
            "price": exit_price,
            "reason": f"خروج: {reason.value}",
            "position": self._position.value,
            "pnl_percentage": round(pnl_percentage, 2),
            "pnl_amount": round(pnl_amount, 2),
            "exit_reason": reason.value
        }
        
        self._current_trade = None
        return result

    def generate_signal(self, data: pd.DataFrame, current_index: int = 0) -> Dict[str, Any]:
        """تولید سیگنال با منطق ساده"""
        try:
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
            
            atr = self.calculate_atr(data.tail(50))
            if atr == 0:
                return self._create_hold_signal("ATR نامعتبر")
            
            # بررسی LONG
            if self._position == PositionType.OUT:
                has_signal, conditions = self.check_simple_conditions(data, PositionType.LONG)
                
                if has_signal:
                    stop_loss = current_price - (atr * self.base_stop_atr_multiplier)
                    take_profit = current_price + ((current_price - stop_loss) * self.base_take_profit_ratio)
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
                            initial_stop_loss=stop_loss
                        )
                        
                        self._last_trade_index = current_index
                        
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
                            "signal_strength": "STRONG"
                        }
            
            # بررسی SHORT
            if self.enable_short_trades and self._position == PositionType.OUT:
                has_signal, conditions = self.check_simple_conditions(data, PositionType.SHORT)
                
                if has_signal:
                    stop_loss = current_price + (atr * self.base_stop_atr_multiplier)
                    take_profit = current_price - ((stop_loss - current_price) * self.base_take_profit_ratio)
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
                            initial_stop_loss=stop_loss
                        )
                        
                        self._last_trade_index = current_index
                        
                        return {
                            "action": "SHORT",
                            "price": current_price,
                            "rsi": current_rsi,
                            "position_size": round(position_size, 4),
                            "stop_loss": round(stop_loss, 4),
                            "take_profit": round(take_profit, 4),
                            "risk_reward_ratio": round(self.base_take_profit_ratio, 2),
                            "reason": "\n".join(conditions),
                            "position": self._position.value,
                            "signal_strength": "STRONG"
                        }
            
            return self._create_hold_signal(f"منتظر شرایط (RSI: {current_rsi:.1f})")
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}", exc_info=True)
            return self._create_hold_signal(f"خطا: {str(e)}")

    def _create_hold_signal(self, reason: str) -> Dict[str, Any]:
        return {
            "action": "HOLD",
            "price": 0,
            "rsi": 0,
            "reason": reason,
            "position": self._position.value,
            "signal_strength": "NEUTRAL"
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        if self._total_trades == 0:
            return {"total_trades": 0, "winning_trades": 0, "win_rate": 0}
        
        win_rate = (self._winning_trades / self._total_trades) * 100
        avg_trade = self._total_pnl / self._total_trades
        
        return {
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "losing_trades": self._total_trades - self._winning_trades,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(self._total_pnl, 2),
            "average_trade_pnl": round(avg_trade, 2),
            "current_portfolio_value": round(self._portfolio_value, 2),
            "current_position": self._position.value,
            "portfolio_return": round(((self._portfolio_value - 10000) / 10000) * 100, 2)
        }

    def reset_state(self):
        self._position = PositionType.OUT
        self._current_trade = None
        self._trade_history = []
        self._portfolio_value = 10000.0
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = 0.0
        self._last_trade_index = -100

    @property
    def position(self):
        return self._position

    @property
    def trade_history(self):
        return self._trade_history.copy()
    
    @property
    def current_trade(self):
        return self._current_trade