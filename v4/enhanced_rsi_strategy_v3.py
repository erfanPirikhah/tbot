# enhanced_rsi_strategy_v3_fixed.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import traceback

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
    partial_exit_done: bool = False
    
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

class EnhancedRsiStrategyV3:
    """
    استراتژی RSI نسخه ۳ - کاملاً بازنویسی و بهبود یافته
    """
    
    def __init__(
        self,
        # پارامترهای اصلی RSI
        rsi_period: int = 14,
        rsi_oversold: int = 35,
        rsi_overbought: int = 65,
        rsi_entry_buffer: int = 3,
        
        # مدیریت ریسک
        risk_per_trade: float = 0.015,
        stop_loss_atr_multiplier: float = 1.2,
        take_profit_ratio: float = 2.0,
        min_position_size: float = 500,
        
        # کنترل معاملات
        max_trades_per_100: int = 25,
        min_candles_between: int = 3,
        max_trade_duration: int = 50,
        
        # فیلترها
        enable_trend_filter: bool = True,
        enable_volume_filter: bool = False,
        enable_short_trades: bool = True,
        
        # ویژگی‌های پیشرفته
        enable_trailing_stop: bool = True,
        trailing_activation_percent: float = 0.5,
        enable_partial_exit: bool = True,
        partial_exit_ratio: float = 0.4,
        
        # کنترل ضرر
        max_consecutive_losses: int = 4,
        pause_after_losses: int = 10
    ):
        # پارامترهای اصلی
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.rsi_entry_buffer = rsi_entry_buffer
        
        # مدیریت ریسک
        self.risk_per_trade = risk_per_trade
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_ratio = take_profit_ratio
        self.min_position_size = min_position_size
        
        # کنترل معاملات
        self.max_trades_per_100 = max_trades_per_100
        self.min_candles_between = min_candles_between
        self.max_trade_duration = max_trade_duration
        
        # فیلترها
        self.enable_trend_filter = enable_trend_filter
        self.enable_volume_filter = enable_volume_filter
        self.enable_short_trades = enable_short_trades
        
        # ویژگی‌های پیشرفته
        self.enable_trailing_stop = enable_trailing_stop
        self.trailing_activation_percent = trailing_activation_percent
        self.enable_partial_exit = enable_partial_exit
        self.partial_exit_ratio = partial_exit_ratio
        
        # کنترل ضرر
        self.max_consecutive_losses = max_consecutive_losses
        self.pause_after_losses = pause_after_losses
        
        # State variables
        self._position = PositionType.OUT
        self._current_trade = None
        self._trade_history = []
        self._portfolio_value = 10000.0
        self._last_trade_index = -100
        self._consecutive_losses = 0
        self._pause_until_index = -1
        
        # آمارها
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = 0.0
        self._gross_profit = 0.0
        self._gross_loss = 0.0
        
        # لاگ‌های حرفه‌ای
        self._signal_log = []
        
        logger.info(">>> Enhanced RSI Strategy V3 Initialized")
        logger.info(f"    Parameters: RSI({rsi_period}), OS: {rsi_oversold}, OB: {rsi_overbought}")
        logger.info(f"    Risk: {risk_per_trade*100}%, SL: {stop_loss_atr_multiplier}ATR, TP: {take_profit_ratio}:1")

    def _log_signal(self, signal_type: str, details: Dict[str, Any]):
        """لاگ حرفه‌ای برای تمام سیگنال‌ها"""
        log_entry = {
            'timestamp': datetime.now(),
            'type': signal_type,
            'position': self._position.value,
            'portfolio_value': self._portfolio_value,
            'details': details
        }
        self._signal_log.append(log_entry)
        
        # تبدیل مقادیر numpy به float برای نمایش بهتر
        formatted_details = {}
        for key, value in details.items():
            if hasattr(value, 'dtype'):  # اگر numpy type است
                formatted_details[key] = float(value)
            else:
                formatted_details[key] = value
        
        logger.info(f"SIGNAL {signal_type}: {formatted_details}")

    def _calculate_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """محاسبه RSI با مدیریت خطا"""
        try:
            if len(data) < self.rsi_period + 1:
                logger.warning(f"    داده کافی برای محاسبه RSI نیست: {len(data)} کندل")
                return data
                
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.ewm(alpha=1/self.rsi_period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/self.rsi_period, adjust=False).mean()
            
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # پر کردن مقادیر NaN
            data['RSI'] = data['RSI'].fillna(method='bfill').fillna(50)
            
            logger.debug(f"    RSI calculated: {data['RSI'].iloc[-1]:.2f}")
            return data
            
        except Exception as e:
            logger.error(f"    خطا در محاسبه RSI: {e}")
            logger.error(traceback.format_exc())
            data['RSI'] = 50  # مقدار پیش‌فرض
            return data

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """محاسبه ATR با مدیریت خطاهای پیشرفته"""
        try:
            if len(data) < period + 1:
                logger.warning(f"    داده کافی برای ATR نیست: {len(data)} کندل")
                return data['close'].iloc[-1] * 0.01  # مقدار پیش‌فرض
            
            high = data['high']
            low = data['low']
            close = data['close']
            
            # محاسبه True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            if pd.isna(atr) or atr <= 0:
                atr = data['close'].iloc[-1] * 0.01
                
            logger.debug(f"    ATR calculated: {atr:.6f}")
            return atr
            
        except Exception as e:
            logger.error(f"    خطا در محاسبه ATR: {e}")
            logger.error(traceback.format_exc())
            return data['close'].iloc[-1] * 0.01

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """محاسبه سایز پوزیشن با لاگ کامل"""
        try:
            risk_amount = self._portfolio_value * self.risk_per_trade
            
            if stop_loss < entry_price:  # LONG
                price_risk = entry_price - stop_loss
            else:  # SHORT
                price_risk = stop_loss - entry_price
            
            if price_risk <= 0:
                logger.warning(f"    قیمت ریسک نامعتبر: {price_risk}")
                price_risk = entry_price * 0.005  # 0.5% پیش‌فرض
            
            position_size = risk_amount / price_risk
            
            # محدودیت‌های سایز
            max_position = self._portfolio_value * 0.3  # حداکثر 30% سرمایه
            position_size = min(position_size, max_position)
            
            if position_size < self.min_position_size:
                logger.warning(f"    سایز پوزیشن بسیار کوچک: {position_size:.0f}")
                return 0
                
            logger.info(f"    Position Size: {position_size:.0f} (Risk: ${risk_amount:.2f})")
            return position_size
            
        except Exception as e:
            logger.error(f"    خطا در محاسبه سایز پوزیشن: {e}")
            return 0

    def _check_trend_filter(self, data: pd.DataFrame) -> Tuple[bool, str]:
        if not self.enable_trend_filter:
            return True, "فیلتر روند غیرفعال"
            
        try:
            if len(data) < 50:
                return True, "داده ناکافی برای فیلتر روند"
            
            # میانگین‌های متحرک چندگانه
            ema_8 = data['close'].ewm(span=8).mean().iloc[-1]
            ema_21 = data['close'].ewm(span=21).mean().iloc[-1]
            ema_50 = data['close'].ewm(span=50).mean().iloc[-1]
            
            current_price = data['close'].iloc[-1]
            
            # قدرت روند
            trend_strength = abs(ema_8 - ema_21) / ema_21
            
            # فیلتر ترکیبی
            if ema_8 > ema_21 > ema_50 and current_price > ema_8:
                return True, f"روند صعودی قوی (قدرت: {trend_strength:.3f})"
            elif ema_8 < ema_21 < ema_50 and current_price < ema_8:
                return True, f"روند نزولی قوی (قدرت: {trend_strength:.3f})"
            elif trend_strength < 0.005:  # بازار رنج
                return False, f"بازار رنج - عدم ورود (قدرت: {trend_strength:.3f})"
            else:
                return False, f"روند ضعیف یا نامشخص (قدرت: {trend_strength:.3f})"
                
        except Exception as e:
            logger.error(f"خطا در فیلتر روند: {e}")
            return True, f"خطا در فیلتر روند: {e}"

    def check_entry_conditions(self, data: pd.DataFrame, position_type: PositionType) -> Tuple[bool, List[str]]:
        """شرایط ورود ساده‌شده و کارآمد"""
        conditions = []
        
        try:
            # محاسبه RSI اگر وجود ندارد
            if 'RSI' not in data.columns:
                data = self._calculate_rsi(data)
            
            current_rsi = data['RSI'].iloc[-1]
            current_price = data['close'].iloc[-1]
            
            # بررسی RSI
            if position_type == PositionType.LONG:
                rsi_condition = current_rsi <= (self.rsi_oversold + self.rsi_entry_buffer)
                if not rsi_condition:
                    return False, [f"RSI برای LONG مناسب نیست ({current_rsi:.1f} > {self.rsi_oversold + self.rsi_entry_buffer})"]
                conditions.append(f"RSI در ناحیه خرید ({current_rsi:.1f})")
                
            elif position_type == PositionType.SHORT:
                rsi_condition = current_rsi >= (self.rsi_overbought - self.rsi_entry_buffer)
                if not rsi_condition:
                    return False, [f"RSI برای SHORT مناسب نیست ({current_rsi:.1f} < {self.rsi_overbought - self.rsi_entry_buffer})"]
                conditions.append(f"RSI در ناحیه فروش ({current_rsi:.1f})")
            
            # فیلتر روند
            trend_ok, trend_reason = self._check_trend_filter(data)
            if not trend_ok:
                return False, [f"فیلتر روند: {trend_reason}"]
            conditions.append(f"روند: {trend_reason}")
            
            # فیلتر حجم (اختیاری)
            if self.enable_volume_filter and 'volume' in data.columns:
                if len(data) > 20:
                    avg_volume = data['volume'].rolling(20).mean().iloc[-1]
                    current_volume = data['volume'].iloc[-1]
                    if current_volume > avg_volume * 0.7:
                        conditions.append("حجم معاملات مناسب")
                    else:
                        conditions.append("حجم پایین (هشدار)")
            
            # بررسی فاصله از آخرین معامله
            candles_since_last = len(data) - 1 - self._last_trade_index
            if candles_since_last < self.min_candles_between:
                return False, [f"فاصله کم از آخرین معامله ({candles_since_last} کندل)"]
            
            conditions.append(f"فاصله از آخرین معامله: {candles_since_last} کندل")
            
            # بررسی pause بعد از ضرر
            if len(data) - 1 <= self._pause_until_index:
                return False, [f"در حالت استراحت بعد از {self._consecutive_losses} ضرر متوالی"]
            
            return True, conditions
            
        except Exception as e:
            logger.error(f"    خطا در بررسی شرایط ورود: {e}")
            logger.error(traceback.format_exc())
            return False, [f"خطا در بررسی شرایط: {e}"]

    def calculate_stop_take_profit(self, data: pd.DataFrame, position_type: PositionType, entry_price: float) -> Tuple[float, float]:
        """محاسبه استاپ و تیک پروفیت"""
        try:
            atr = self.calculate_atr(data)
            
            if position_type == PositionType.LONG:
                stop_loss = entry_price - (atr * self.stop_loss_atr_multiplier)
                take_profit = entry_price + (atr * self.stop_loss_atr_multiplier * self.take_profit_ratio)
            else:  # SHORT
                stop_loss = entry_price + (atr * self.stop_loss_atr_multiplier)
                take_profit = entry_price - (atr * self.stop_loss_atr_multiplier * self.take_profit_ratio)
            
            # اطمینان از معقول بودن مقادیر
            if position_type == PositionType.LONG:
                if stop_loss >= entry_price:
                    stop_loss = entry_price * 0.995
                if take_profit <= entry_price:
                    take_profit = entry_price * 1.01
            else:
                if stop_loss <= entry_price:
                    stop_loss = entry_price * 1.005
                if take_profit >= entry_price:
                    take_profit = entry_price * 0.99
            
            logger.info(f"    SL: {stop_loss:.4f}, TP: {take_profit:.4f} (ATR: {atr:.5f})")
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"    خطا در محاسبه استاپ/تیک: {e}")
            # مقادیر پیش‌فرض
            if position_type == PositionType.LONG:
                return entry_price * 0.99, entry_price * 1.02
            else:
                return entry_price * 1.01, entry_price * 0.98

    def check_exit_conditions(self, data: pd.DataFrame, current_index: int) -> Optional[Dict[str, Any]]:
        """بررسی شرایط خروج"""
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
            
            # خروج جزئی
            if (self.enable_partial_exit and 
                profit_pct >= 0.8 and 
                not getattr(self._current_trade, 'partial_exit_done', False)):
                
                self._current_trade.partial_exit_done = True
                partial_pnl = (profit_pct / 100) * self._current_trade.quantity * entry_price * self.partial_exit_ratio
                self._portfolio_value += partial_pnl
                
                logger.info(f"    خروج جزئی: {self.partial_exit_ratio*100}% در سود {profit_pct:.2f}%")
                
                return {
                    "action": "PARTIAL_EXIT",
                    "price": current_price,
                    "pnl_percentage": profit_pct,
                    "pnl_amount": partial_pnl,
                    "reason": "PARTIAL_TAKE_PROFIT"
                }
            
            # استاپ لاس
            if self._position == PositionType.LONG and current_price <= self._current_trade.stop_loss:
                return self._create_exit_signal("STOP_LOSS", current_price, current_time)
            elif self._position == PositionType.SHORT and current_price >= self._current_trade.stop_loss:
                return self._create_exit_signal("STOP_LOSS", current_price, current_time)
            
            # تیک پروفیت
            if self._position == PositionType.LONG and current_price >= self._current_trade.take_profit:
                return self._create_exit_signal("TAKE_PROFIT", current_price, current_time)
            elif self._position == PositionType.SHORT and current_price <= self._current_trade.take_profit:
                return self._create_exit_signal("TAKE_PROFIT", current_price, current_time)
            
            # تریلینگ استاپ
            if self.enable_trailing_stop and abs(profit_pct) >= self.trailing_activation_percent:
                atr = self.calculate_atr(data)
                
                if self._position == PositionType.LONG:
                    new_trailing = current_price - (atr * 1.0)
                    if new_trailing > self._current_trade.trailing_stop:
                        self._current_trade.trailing_stop = new_trailing
                        logger.debug(f"    تریلینگ استاپ به‌روزرسانی شد: {new_trailing:.4f}")
                    
                    if current_price <= self._current_trade.trailing_stop:
                        return self._create_exit_signal("TRAILING_STOP", current_price, current_time)
                
                else:  # SHORT
                    new_trailing = current_price + (atr * 1.0)
                    if new_trailing < self._current_trade.trailing_stop:
                        self._current_trade.trailing_stop = new_trailing
                        logger.debug(f"    تریلینگ استاپ به‌روزرسانی شد: {new_trailing:.4f}")
                    
                    if current_price >= self._current_trade.trailing_stop:
                        return self._create_exit_signal("TRAILING_STOP", current_price, current_time)
            
            # زمان خروج
            trade_duration = current_index - self._last_trade_index
            if trade_duration >= self.max_trade_duration:
                return self._create_exit_signal("TIME_EXIT", current_price, current_time)
            
            return None
            
        except Exception as e:
            logger.error(f"    خطا در بررسی شرایط خروج: {e}")
            logger.error(traceback.format_exc())
            return None

    def _create_exit_signal(self, exit_reason: str, price: float, time: pd.Timestamp) -> Dict[str, Any]:
        """ایجاد سیگنال خروج"""
        try:
            if self._current_trade is None:
                return {"action": "HOLD", "reason": "No active trade to exit"}
            
            entry_price = self._current_trade.entry_price
            quantity = self._current_trade.quantity
            
            if self._position == PositionType.LONG:
                pnl_amount = (price - entry_price) * quantity
            else:
                pnl_amount = (entry_price - price) * quantity
            
            pnl_percentage = (pnl_amount / (entry_price * quantity)) * 100
            
            # به‌روزرسانی پورتفو
            self._portfolio_value += pnl_amount
            self._total_pnl += pnl_amount
            self._total_trades += 1
            
            # مدیریت ضررهای متوالی
            if pnl_amount < 0:
                self._consecutive_losses += 1
                if self._consecutive_losses >= self.max_consecutive_losses:
                    self._pause_until_index = len(self._trade_history) + self.pause_after_losses
                    logger.warning(f"    {self._consecutive_losses} ضرر متوالی - استراحت برای {self.pause_after_losses} کندل")
            else:
                self._consecutive_losses = 0
                self._winning_trades += 1
            
            if pnl_amount > 0:
                self._gross_profit += pnl_amount
            else:
                self._gross_loss += abs(pnl_amount)
            
            # ثبت معامله
            self._current_trade.exit_price = price
            self._current_trade.exit_time = time
            self._current_trade.pnl_percentage = pnl_percentage
            self._current_trade.pnl_amount = pnl_amount
            
            old_position = self._position
            self._trade_history.append(self._current_trade)
            self._position = PositionType.OUT
            self._current_trade = None
            
            log_details = {
                "price": float(price),
                "exit_reason": exit_reason,
                "pnl_percentage": round(pnl_percentage, 2),
                "pnl_amount": round(pnl_amount, 2),
                "position": old_position.value
            }
            self._log_signal("EXIT", log_details)
            
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
            logger.error(f"    خطا در ایجاد سیگنال خروج: {e}")
            return {"action": "HOLD", "reason": f"Exit error: {e}"}

    def generate_signal(self, data: pd.DataFrame, current_index: int = 0) -> Dict[str, Any]:
        """تولید سیگنال اصلی"""
        try:
            # بررسی خروج اولویت دارد
            exit_signal = self.check_exit_conditions(data, current_index)
            if exit_signal:
                return exit_signal
            
            # بررسی شرایط عمومی
            if len(data) < 30:
                return {"action": "HOLD", "reason": "داده کافی نیست"}
            
            if current_index <= self._pause_until_index:
                return {"action": "HOLD", "reason": f"استراحت بعد از {self._consecutive_losses} ضرر متوالی"}
            
            # بررسی محدودیت معاملات
            recent_trades = len([t for t in self._trade_history[-100:] if t.exit_time])
            if recent_trades >= self.max_trades_per_100:
                return {"action": "HOLD", "reason": f"حد معاملات ({recent_trades}/{self.max_trades_per_100})"}
            
            current_price = data['close'].iloc[-1]
            
            # بررسی LONG
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
                            trailing_stop=stop_loss
                        )
                        self._last_trade_index = current_index
                        
                        signal_details = {
                            "price": float(current_price),
                            "position_size": float(position_size),
                            "stop_loss": float(stop_loss),
                            "take_profit": float(take_profit),
                            "conditions": long_conditions
                        }
                        self._log_signal("LONG_ENTRY", signal_details)
                        
                        return {
                            "action": "BUY",
                            "price": float(current_price),
                            "position_size": float(position_size),
                            "stop_loss": float(stop_loss),
                            "take_profit": float(take_profit),
                            "reason": " | ".join(long_conditions),
                            "position": "LONG"
                        }
            
            # بررسی SHORT
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
                            trailing_stop=stop_loss
                        )
                        self._last_trade_index = current_index
                        
                        signal_details = {
                            "price": float(current_price),
                            "position_size": float(position_size),
                            "stop_loss": float(stop_loss),
                            "take_profit": float(take_profit),
                            "conditions": short_conditions
                        }
                        self._log_signal("SHORT_ENTRY", signal_details)
                        
                        return {
                            "action": "SELL",
                            "price": float(current_price),
                            "position_size": float(position_size),
                            "stop_loss": float(stop_loss),
                            "take_profit": float(take_profit),
                            "reason": " | ".join(short_conditions),
                            "position": "SHORT"
                        }
            
            return {"action": "HOLD", "reason": "منتظر شرایط مناسب"}
            
        except Exception as e:
            logger.error(f"    خطا در تولید سیگنال: {e}")
            logger.error(traceback.format_exc())
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
            "current_position": self._position.value
        }

    def get_signal_log(self) -> List[Dict]:
        """دریافت لاگ کامل سیگنال‌ها"""
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
        
        logger.info("    استراتژی ریست شد")