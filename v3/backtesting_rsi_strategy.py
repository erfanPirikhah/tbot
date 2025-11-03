# backtesting_rsi_strategy.py
from backtesting import Strategy
import pandas as pd
import numpy as np

class BacktestingRsiStrategy(Strategy):
    """
    استراتژی RSI پیشرفته و حرفه‌ای برای استفاده با کتابخانه backtesting.py.
    این نسخه با تمرکز روی پایداری و خوانایی بالا نوشته شده است.
    """
    # --- پارامترهای اصلی استراتژی ---
    rsi_period = 14
    rsi_oversold = 45
    rsi_overbought = 55
    risk_per_trade = 0.02
    atr_multiplier = 2.0
    rrr_ratio = 2.0

    # --- پارامترهای ویژگی‌های پیشرفته ---
    enable_pyramiding = False
    pyramid_profit_threshold = 1.5
    pyramid_max_entries = 3
    enable_trailing_stop = False
    trailing_atr_multiplier = 1.5

    def init(self):
        def rsi_calc(close_prices, period):
            close_series = pd.Series(close_prices)
            delta = close_series.diff(1)
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            rs = gain / loss.replace(0, np.inf)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        self.rsi = self.I(rsi_calc, self.data.Close, self.rsi_period)

        def atr_calc(high, low, close, period):
            high_series = pd.Series(high)
            low_series = pd.Series(low)
            close_series = pd.Series(close)
            
            tr1 = high_series - low_series
            tr2 = abs(high_series - close_series.shift(1))
            tr3 = abs(low_series - close_series.shift(1))
            
            true_range = np.maximum.reduce([tr1, tr2, tr3])
            atr = pd.Series(true_range).rolling(window=period, min_periods=1).mean()
            return atr
        
        self.atr = self.I(atr_calc, self.data.High, self.data.Low, self.data.Close, 14)

    def next(self):
        price = self.data.Close[-1]
        current_rsi = self.rsi[-1]
        current_atr = self.atr[-1]

        if not self.position:
            if current_rsi < self.rsi_oversold:
                sl, tp = self._calculate_sl_tp(price, current_atr, is_long=True)
                size = self._calculate_position_size(price, sl)
                if size > 0:
                    self.buy(size=size, sl=sl, tp=tp)

            elif current_rsi > self.rsi_overbought:
                sl, tp = self._calculate_sl_tp(price, current_atr, is_long=False)
                size = self._calculate_position_size(price, sl, is_short=True)
                if size > 0:
                    self.sell(size=size, sl=sl, tp=tp)
        else:
            if self.enable_pyramiding and len(self.trades) < self.pyramid_max_entries:
                self._handle_pyramiding(price, current_atr)
            
            if self.enable_trailing_stop:
                self._handle_trailing_stop(price, current_atr)

    def _calculate_sl_tp(self, price, atr, is_long=True):
        if is_long:
            sl = price - atr * self.atr_multiplier
            tp = price + atr * self.atr_multiplier * self.rrr_ratio
        else:
            sl = price + atr * self.atr_multiplier
            tp = price - atr * self.atr_multiplier * self.rrr_ratio
        return sl, tp

    def _calculate_position_size(self, entry_price, stop_loss, is_short=False):
        """
        محاسبه حجم معامله بر اساس درصد ریسک ثابت (نسخه اصلاح شده برای فارکس).
        این نسخه حجم را بر اساس لات برمی‌گرداند.
        """
        risk_amount = self.equity * self.risk_per_trade
        
        if is_short:
            price_diff = stop_loss - entry_price
        else:
            price_diff = entry_price - stop_loss
        
        # اگر حد ضرر خیلی نزدیک به قیمت ورود باشد، از یک حداقل فاصله استفاده می‌کنیم
        min_price_diff = entry_price * 0.002 # 0.2%
        if price_diff < min_price_diff:
            price_diff = min_price_diff
        
        if price_diff <= 0:
            return 0
        
        # محاسبه حجم بر اساس واحد پولی (مثلا دلار)
        size_in_units = risk_amount / price_diff
        
        # --- راه‌حل اصلی: تبدیل حجم به لات ---
        # در فارکس، ۱ لات استاندارد = ۱۰۰,۰۰۰ واحد
        # ما حجم محاسبه شده را به لات تبدیل می‌کنیم.
        LOT_SIZE = 100000
        size_in_lots = size_in_units / LOT_SIZE
        
        # کتابخانه backtesting.py با اعداد اعشاری کوچک (لات) به خوبی کار می‌کند
        if size_in_lots > 0:
            return size_in_lots
        
        return 0

    def _handle_pyramiding(self, price, atr):
        if self._check_pyramiding_condition(price):
            if self.position.is_long:
                sl = price - atr * self.atr_multiplier
                size = self._calculate_position_size(price, sl)
                if size > 0: self.buy(size=size, sl=sl)
            else:
                sl = price + atr * self.atr_multiplier
                size = self._calculate_position_size(price, sl, is_short=True)
                if size > 0: self.sell(size=size, sl=sl, tp=self.position.tp)

    def _handle_trailing_stop(self, price, atr):
        if self.position.is_long:
            new_sl = price - atr * self.trailing_atr_multiplier
            if self.position.sl is None or new_sl > self.position.sl:
                self.position.sl = new_sl
        else:
            new_sl = price + atr * self.trailing_atr_multiplier
            if self.position.sl is None or new_sl < self.position.sl:
                self.position.sl = new_sl

    def _check_pyramiding_condition(self, current_price):
        entry_price = self.trades[0].entry_price
        if self.position.is_long:
            profit_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            profit_pct = ((entry_price - current_price) / entry_price) * 100
        return profit_pct >= self.pyramid_profit_threshold