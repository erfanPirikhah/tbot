# strategies/basic_strategy.py

from config import RSI_OVERSOLD, RSI_OVERBOUGHT

class RsiStrategy:
    """
    یک استراتژی معاملاتی مبتنی بر اندیکاتور RSI با مدیریت حالت.
    این کلاس نقاط ورود (خرید) و خروج (فروش) را بر اساس عبور RSI از سطوح
    اشباع خرید/فروش تشخیص می‌دهد و وضعیت فعلی (در بازار یا خارج از بازار) را حفظ می‌کند.
    """
    def __init__(self, overbought: int = RSI_OVERBOUGHT, oversold: int = RSI_OVERSOLD):
        self.overbought = overbought
        self.oversold = oversold
        self.position = "OUT_OF_MARKET"  # وضعیت فعلی: 'HOLDING' (در سهم) یا 'OUT_OF_MARKET' (خارج از سهم)
        self.last_rsi = None  # برای ذخیره آخرین مقدار RSI در تحلیل قبلی

    def generate_signal(self, current_price: float, current_rsi: float) -> dict:
        """
        بر اساس قیمت و RSI فعلی، یک سیگنال معاملاتی با جزئیات کامل تولید می‌کند.

        Args:
            current_price (float): آخرین قیمت بسته‌شدن.
            current_rsi (float): آخرین مقدار محاسبه‌شده برای RSI.

        Returns:
            dict: یک دیکشنری شامل جزئیات سیگنال.
                  مثال: {'action': 'BUY', 'price': 114500, 'rsi': 29.5, 'reason': 'RSI from oversold to neutral'}
        """
        # اگر اولین تحلیل است، فقط مقدار RSI را ذخیره کن و سیگنال نگهداری بده
        if self.last_rsi is None:
            self.last_rsi = current_rsi
            return {
                "action": "HOLD",
                "price": current_price,
                "rsi": current_rsi,
                "reason": "Initial analysis. Waiting for a signal.",
                "position": self.position
            }

        # منطق اصلی استراتژی: تشخیص عبور از سطوح (Crossover)

        # --- سیگنال خرید ---
        # اگر خارج از بازار هستیم و RSI از منطقه اشباع فروش به بالاتر از آن حرکت کرد
        if self.position == "OUT_OF_MARKET" and self.last_rsi < self.oversold and current_rsi >= self.oversold:
            self.position = "HOLDING"  # وضعیت را به "در حال نگهداری" تغییر بده
            self.last_rsi = current_rsi
            return {
                "action": "BUY",
                "price": current_price,
                "rsi": current_rsi,
                "reason": f"RSI crossed above oversold level ({self.oversold}). Potential entry point.",
                "position": self.position
            }

        # --- سیگنال فروش ---
        # اگر در حال نگهداری سهم هستیم و RSI از منطقه اشباع خرید به پایین‌تر از آن حرکت کرد
        if self.position == "HOLDING" and self.last_rsi > self.overbought and current_rsi <= self.overbought:
            self.position = "OUT_OF_MARKET"  # وضعیت را به "خارج از بازار" تغییر بده
            self.last_rsi = current_rsi
            return {
                "action": "SELL",
                "price": current_price,
                "rsi": current_rsi,
                "reason": f"RSI crossed below overbought level ({self.overbought}). Potential exit point.",
                "position": self.position
            }

        # --- اگر هیچ سیگنالی نبود ---
        # در غیر این صورت، وضعیت فعلی را حفظ کن و سیگنال نگهداری بده
        self.last_rsi = current_rsi
        return {
            "action": "HOLD",
            "price": current_price,
            "rsi": current_rsi,
            "reason": f"No clear signal. Current position is {self.position}.",
            "position": self.position
        }
