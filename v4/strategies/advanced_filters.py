# strategies/advanced_filters.py

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class AdvancedMarketFilters:
    """فیلترهای پیشرفته بازار"""
    
    @staticmethod
    def detect_market_regime(data: pd.DataFrame) -> Dict[str, Any]:
        """تشخیص رژیم بازار با چندین اندیکاتور"""
        try:
            close = data['close']
            
            # محاسبه میانگین‌های متحرک
            sma_20 = close.rolling(20).mean().iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1]
            sma_200 = close.rolling(200).mean().iloc[-1]
            
            # محاسبه RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean().iloc[-1]
            avg_loss = loss.rolling(14).mean().iloc[-1]
            rs = avg_gain / avg_loss if avg_loss != 0 else 1
            rsi = 100 - (100 / (1 + rs))
            
            # محاسبه نوسان
            volatility = close.pct_change().rolling(20).std().iloc[-1]
            
            # محاسبه حجم (اگر موجود باشد)
            if 'volume' in data.columns:
                volume_sma = data['volume'].rolling(20).mean().iloc[-1]
                current_volume = data['volume'].iloc[-1]
                volume_ratio = current_volume / volume_sma if volume_sma > 0 else 1
            else:
                volume_ratio = 1
            
            # تشخیص رژیم
            regime = "SIDEWAYS"
            confidence = 0.5
            
            # شرایط روند صعودی
            if (sma_20 > sma_50 > sma_200 and 
                close.iloc[-1] > sma_20 and 
                rsi > 40 and rsi < 80):
                regime = "BULLISH"
                confidence = min(0.8, (rsi - 40) / 40)
            
            # شرایط روند نزولی
            elif (sma_20 < sma_50 < sma_200 and 
                  close.iloc[-1] < sma_20 and 
                  rsi < 60 and rsi > 20):
                regime = "BEARISH" 
                confidence = min(0.8, (60 - rsi) / 40)
            
            # شرایط پرنوسان
            elif volatility > 0.02:
                regime = "VOLATILE"
                confidence = min(0.9, volatility / 0.03)
            
            return {
                'regime': regime,
                'confidence': round(confidence, 2),
                'sma_20': sma_20,
                'sma_50': sma_50, 
                'sma_200': sma_200,
                'rsi': rsi,
                'volatility': volatility,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            logger.error(f"خطا در تشخیص رژیم بازار: {e}")
            return {'regime': 'UNKNOWN', 'confidence': 0}
    
    @staticmethod
    def calculate_support_resistance(data: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
        """محاسبه سطوح حمایت و مقاومت"""
        try:
            high = data['high'].rolling(window).max().iloc[-1]
            low = data['low'].rolling(window).min().iloc[-1]
            close = data['close'].iloc[-1]
            
            # محاسبه سطوح مبتنی بر قیمت بسته شدن
            pivot = (high + low + close) / 3
            resistance = 2 * pivot - low
            support = 2 * pivot - high
            
            return support, resistance
            
        except Exception as e:
            logger.error(f"خطا در محاسبه حمایت/مقاومت: {e}")
            return 0, 0
    
    @staticmethod
    def fibonacci_levels(data: pd.DataFrame, period: int = 50) -> Dict[str, float]:
        """محاسبه سطوح فیبوناچی"""
        try:
            high = data['high'].tail(period).max()
            low = data['low'].tail(period).min()
            diff = high - low
            
            levels = {
                '0.0': low,
                '0.236': low + diff * 0.236,
                '0.382': low + diff * 0.382,
                '0.5': low + diff * 0.5,
                '0.618': low + diff * 0.618,
                '0.786': low + diff * 0.786,
                '1.0': high
            }
            
            return levels
            
        except Exception as e:
            logger.error(f"خطا در محاسبه فیبوناچی: {e}")
            return {}