# strategies/advanced_rsi_strategy.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# تنظیمات لاگینگ
logger = logging.getLogger(__name__)

class PositionType(Enum):
    OUT = "OUT"
    LONG = "LONG"
    SHORT = "SHORT"

class SignalStrength(Enum):
    WEAK = "WEAK"
    MEDIUM = "MEDIUM" 
    STRONG = "STRONG"

@dataclass
class Trade:
    """کلاس برای ذخیره اطلاعات معامله"""
    entry_price: float
    entry_time: pd.Timestamp
    position_type: PositionType
    quantity: float
    stop_loss: float
    take_profit: float
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    pnl_percentage: Optional[float] = None
    pnl_amount: Optional[float] = None

@dataclass
class MarketConditions:
    """کلاس برای تحلیل شرایط بازار"""
    trend: str  # UPTREND, DOWNTREND, SIDEWAYS
    volatility: str  # HIGH, MEDIUM, LOW
    momentum: str  # BULLISH, BEARISH, NEUTRAL

class AdvancedRsiStrategy:
    """
    استراتژی پیشرفته RSI با ویژگی‌های:
    - فیلتر روند
    - تاییدیه چندگانه
    - مدیریت ریسک پیشرفته
    - تشخیص واگرایی
    - تحلیل نوسانات
    """
    
    def __init__(
        self,
        overbought: int = 70,
        oversold: int = 30,
        rsi_period: int = 14,
        # پارامترهای فیلتر روند
        trend_ma_period: int = 20,
        trend_threshold: float = 0.02,  # 2%
        # پارامترهای تاییدیه
        confirmation_period: int = 3,
        volume_threshold: float = 1.2,
        # پارامترهای مدیریت ریسک
        risk_per_trade: float = 0.02,  # 2%
        stop_loss_atr_multiplier: float = 1.5,
        take_profit_ratio: float = 2.0,  # Risk/Reward = 1:2
        # پارامترهای تشخیص واگرایی
        divergence_lookback: int = 10,
        min_divergence_strength: float = 0.3
    ):
        # اعتبارسنجی پارامترها
        self._validate_parameters(
            overbought, oversold, risk_per_trade, 
            stop_loss_atr_multiplier, take_profit_ratio
        )
        
        # پارامترهای استراتژی
        self.overbought = overbought
        self.oversold = oversold
        self.rsi_period = rsi_period
        self.trend_ma_period = trend_ma_period
        self.trend_threshold = trend_threshold
        self.confirmation_period = confirmation_period
        self.volume_threshold = volume_threshold
        self.risk_per_trade = risk_per_trade
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_ratio = take_profit_ratio
        self.divergence_lookback = divergence_lookback
        self.min_divergence_strength = min_divergence_strength
        
        # مدیریت وضعیت
        self._position = PositionType.OUT
        self._current_trade: Optional[Trade] = None
        self._trade_history: List[Trade] = []
        self._portfolio_value: float = 10000.0  # مقدار اولیه
        
        # داده‌های تاریخی برای تحلیل
        self._price_data: List[float] = []
        self._rsi_data: List[float] = []
        self._volume_data: List[float] = []
        self._timestamps: List[pd.Timestamp] = []
        
        # آمار عملکرد
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = 0.0

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
        """
        تحلیل جامع شرایط بازار
        """
        prices = data['close'].tail(50)  # 50 دوره اخیر
        
        # تحلیل روند
        ma_20 = prices.rolling(20).mean().iloc[-1]
        ma_50 = prices.rolling(50).mean().iloc[-1] if len(prices) >= 50 else ma_20
        
        if ma_20 > ma_50 * (1 + self.trend_threshold):
            trend = "UPTREND"
        elif ma_20 < ma_50 * (1 - self.trend_threshold):
            trend = "DOWNTREND"
        else:
            trend = "SIDEWAYS"
        
        # تحلیل نوسانات
        volatility = prices.pct_change().std()
        if volatility > 0.05:  # 5%
            volatility_level = "HIGH"
        elif volatility > 0.02:  # 2%
            volatility_level = "MEDIUM"
        else:
            volatility_level = "LOW"
        
        # تحلیل مومنتوم
        rsi_current = data['RSI'].iloc[-1]
        rsi_previous = data['RSI'].iloc[-2] if len(data) > 1 else rsi_current
        
        if rsi_current > 50 and rsi_current > rsi_previous:
            momentum = "BULLISH"
        elif rsi_current < 50 and rsi_current < rsi_previous:
            momentum = "BEARISH"
        else:
            momentum = "NEUTRAL"
        
        return MarketConditions(trend, volatility_level, momentum)

    def detect_rsi_divergence(self, data: pd.DataFrame) -> Tuple[bool, bool, float]:
        """
        تشخیص واگرایی RSI با قیمت
        بازگشت‌های True, False, strength برای (واگرایی مثبت, واگرایی منفی, قدرت سیگنال)
        """
        if len(data) < self.divergence_lookback:
            return False, False, 0.0
        
        prices = data['close'].tail(self.divergence_lookback)
        rsi_values = data['RSI'].tail(self.divergence_lookback)
        
        # یافتن قله‌ها و دره‌ها در قیمت و RSI
        price_peaks = self._find_peaks(prices)
        price_troughs = self._find_troughs(prices)
        rsi_peaks = self._find_peaks(rsi_values)
        rsi_troughs = self._find_troughs(rsi_values)
        
        # تشخیص واگرایی معمولی (Regular Divergence)
        bullish_divergence = False
        bearish_divergence = False
        divergence_strength = 0.0
        
        # واگرایی مثبت: قیمت کف更低، RSI کف بالاتر
        if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
            if (price_troughs[-1] < price_troughs[-2] and 
                rsi_troughs[-1] > rsi_troughs[-2]):
                bullish_divergence = True
                divergence_strength = abs(price_troughs[-1] - price_troughs[-2]) / price_troughs[-2]
        
        # واگرایی منفی: قیمت سقف بالاتر، RSI سقف پایین‌تر
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            if (price_peaks[-1] > price_peaks[-2] and 
                rsi_peaks[-1] < rsi_peaks[-2]):
                bearish_divergence = True
                divergence_strength = abs(price_peaks[-1] - price_peaks[-2]) / price_peaks[-2]
        
        return bullish_divergence, bearish_divergence, divergence_strength

    def _find_peaks(self, series: pd.Series) -> List[float]:
        """پیدا کردن قله‌ها در سری داده"""
        peaks = []
        for i in range(1, len(series)-1):
            if series.iloc[i] > series.iloc[i-1] and series.iloc[i] > series.iloc[i+1]:
                peaks.append(series.iloc[i])
        return peaks

    def _find_troughs(self, series: pd.Series) -> List[float]:
        """پیدا کردن دره‌ها در سری داده"""
        troughs = []
        for i in range(1, len(series)-1):
            if series.iloc[i] < series.iloc[i-1] and series.iloc[i] < series.iloc[i+1]:
                troughs.append(series.iloc[i])
        return troughs

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """محاسبه Average True Range برای تعیین حد ضرر پویا"""
        high = data['high']
        low = data['low']
        close_prev = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close_prev)
        tr3 = abs(low - close_prev)
        
        true_range = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = true_range.rolling(period).mean().iloc[-1]
        
        return atr

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """محاسبه حجم معامله بر اساس مدیریت ریسک"""
        risk_amount = self._portfolio_value * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
            
        position_size = risk_amount / price_risk
        max_position = self._portfolio_value * 0.1  # حداکثر 10% پورتفو
        
        return min(position_size, max_position)

    def calculate_signal_strength(self, data: pd.DataFrame, 
                                market_conditions: MarketConditions,
                                bullish_div: bool, bearish_div: bool,
                                divergence_strength: float) -> SignalStrength:
        """محاسبه قدرت سیگنال"""
        score = 0.0
        current_rsi = data['RSI'].iloc[-1]
        
        # امتیاز بر اساس موقعیت RSI
        if current_rsi < 25 or current_rsi > 75:
            score += 2.0
        elif current_rsi < 30 or current_rsi > 70:
            score += 1.5
        elif current_rsi < 35 or current_rsi > 65:
            score += 1.0
        
        # امتیاز بر اساس واگرایی
        if bullish_div or bearish_div:
            score += divergence_strength * 3
        
        # امتیاز بر اساس روند
        if market_conditions.trend == "UPTREND":
            score += 1.0
        elif market_conditions.trend == "DOWNTREND":
            score += 0.5
        
        # امتیاز بر اساس مومنتوم
        if market_conditions.momentum == "BULLISH":
            score += 0.5
        
        # طبقه‌بندی قدرت سیگنال
        if score >= 3.0:
            return SignalStrength.STRONG
        elif score >= 2.0:
            return SignalStrength.MEDIUM
        else:
            return SignalStrength.WEAK

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        تولید سیگنال معاملاتی با تحلیل جامع
        """
        try:
            if len(data) < max(self.rsi_period, self.trend_ma_period, 20):
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
            
            # 4. محاسبه ATR برای حد ضرر پویا
            atr = self.calculate_atr(data.tail(20))
            
            # 5. شرایط خرید پیشرفته
            buy_conditions = [
                # شرایط RSI
                current_rsi < self.oversold,
                # تاییدیه روند
                market_conditions.trend in ["UPTREND", "SIDEWAYS"],
                # تاییدیه مومنتوم
                market_conditions.momentum in ["BULLISH", "NEUTRAL"],
                # واگرایی مثبت (امتیاز اضافه)
                bullish_div or div_strength > self.min_divergence_strength,
                # وضعیت معاملاتی
                self._position == PositionType.OUT,
                # قدرت سیگنال قابل قبول
                signal_strength != SignalStrength.WEAK
            ]
            
            # 6. شرایط فروش پیشرفته
            sell_conditions = [
                # شرایط RSI
                current_rsi > self.overbought,
                # وضعیت معاملاتی (فقط اگر موقعیت خرید داریم)
                self._position == PositionType.LONG,
                # واگرایی منفی (اختیاری)
                bearish_div,
                # قدرت سیگنال قابل قبول
                signal_strength != SignalStrength.WEAK
            ]
            
            # 7. تولید سیگنال خرید
            if sum(buy_conditions) >= 4:  # حداقل 4 شرط از 6 شرط
                stop_loss = current_price - (atr * self.stop_loss_atr_multiplier)
                take_profit = current_price + ((current_price - stop_loss) * self.take_profit_ratio)
                position_size = self.calculate_position_size(current_price, stop_loss)
                
                self._position = PositionType.LONG
                self._current_trade = Trade(
                    entry_price=current_price,
                    entry_time=data.index[-1] if hasattr(data.index, '[-1]') else pd.Timestamp.now(),
                    position_type=PositionType.LONG,
                    quantity=position_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                reason_parts = []
                if current_rsi < self.oversold:
                    reason_parts.append(f"RSI در ناحیه اشباع فروش ({current_rsi:.1f})")
                if bullish_div:
                    reason_parts.append("واگرایی مثبت شناسایی شد")
                if market_conditions.trend == "UPTREND":
                    reason_parts.append("روند صعودی")
                
                return {
                    "action": "BUY",
                    "price": current_price,
                    "rsi": current_rsi,
                    "position_size": position_size,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "risk_reward_ratio": self.take_profit_ratio,
                    "reason": " | ".join(reason_parts),
                    "position": self._position.value,
                    "signal_strength": signal_strength.value,
                    "market_conditions": {
                        "trend": market_conditions.trend,
                        "volatility": market_conditions.volatility,
                        "momentum": market_conditions.momentum
                    },
                    "divergence_detected": bullish_div
                }
            
            # 8. تولید سیگنال فروش
            elif sum(sell_conditions) >= 3 and self._current_trade:  # حداقل 3 شرط
                # محاسبه سود/زیان
                pnl_percentage = ((current_price - self._current_trade.entry_price) / 
                                self._current_trade.entry_price * 100)
                pnl_amount = (current_price - self._current_trade.entry_price) * self._current_trade.quantity
                
                # آپدیت پورتفو
                self._portfolio_value += pnl_amount
                self._total_trades += 1
                if pnl_amount > 0:
                    self._winning_trades += 1
                self._total_pnl += pnl_amount
                
                # آپدیت معامله
                self._current_trade.exit_price = current_price
                self._current_trade.exit_time = data.index[-1] if hasattr(data.index, '[-1]') else pd.Timestamp.now()
                self._current_trade.pnl_percentage = pnl_percentage
                self._current_trade.pnl_amount = pnl_amount
                
                self._trade_history.append(self._current_trade)
                self._position = PositionType.OUT
                current_trade = self._current_trade
                self._current_trade = None
                
                reason_parts = []
                if current_rsi > self.overbought:
                    reason_parts.append(f"RSI در ناحیه اشباع خرید ({current_rsi:.1f})")
                if bearish_div:
                    reason_parts.append("واگرایی منفی شناسایی شد")
                
                return {
                    "action": "SELL",
                    "price": current_price,
                    "rsi": current_rsi,
                    "reason": " | ".join(reason_parts),
                    "position": self._position.value,
                    "pnl_percentage": pnl_percentage,
                    "pnl_amount": pnl_amount,
                    "signal_strength": signal_strength.value,
                    "trade_duration": (current_trade.exit_time - current_trade.entry_time).total_seconds() / 3600,  # ساعت
                    "performance_metrics": self.get_performance_metrics()
                }
            
            # 9. سیگنال نگهداری
            return self._create_hold_signal(
                f"منتظر سیگنال مناسب. شرایط بازار: {market_conditions.trend}, "
                f"RSI: {current_rsi:.1f}, قدرت سیگنال: {signal_strength.value}"
            )
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return self._create_hold_signal(f"خطا در تحلیل: {str(e)}")

    def _create_hold_signal(self, reason: str) -> Dict[str, Any]:
        """ایجاد سیگنال نگهداری"""
        return {
            "action": "HOLD",
            "price": self._price_data[-1] if self._price_data else 0,
            "rsi": self._rsi_data[-1] if self._rsi_data else 50,
            "reason": reason,
            "position": self._position.value,
            "signal_strength": "NEUTRAL"
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """محاسبه معیارهای عملکرد"""
        if self._total_trades == 0:
            return {}
        
        win_rate = (self._winning_trades / self._total_trades) * 100
        avg_trade = self._total_pnl / self._total_trades
        
        return {
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(self._total_pnl, 2),
            "average_trade_pnl": round(avg_trade, 2),
            "current_portfolio_value": round(self._portfolio_value, 2),
            "current_position": self._position.value
        }

    def reset_state(self):
        """بازنشانی وضعیت استراتژی (برای تست مجدد)"""
        self._position = PositionType.OUT
        self._current_trade = None
        self._trade_history = []
        self._portfolio_value = 10000.0
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = 0.0

    @property
    def position(self):
        return self._position

    @property
    def trade_history(self):
        return self._trade_history.copy()