# strategies/advanced_filters.py - COMPLETE VERSION

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class AdvancedMarketFilters:
    """فیلترهای پیشرفته بازار - نسخه کامل و بهینه"""
    
    @staticmethod
    def detect_market_regime(data: pd.DataFrame) -> Dict[str, Any]:
        """🔥 IMPROVED: تشخیص رژیم بازار با دقت بالاتر"""
        try:
            close = data['close']
            
            # میانگین‌های متحرک
            sma_20 = close.rolling(20).mean().iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1]
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean().iloc[-1]
            avg_loss = loss.rolling(14).mean().iloc[-1]
            rs = avg_gain / avg_loss if avg_loss != 0 else 1
            rsi = 100 - (100 / (1 + rs))
            
            # نوسان
            volatility = close.pct_change().rolling(20).std().iloc[-1]
            
            # 🔥 تشخیص رژیم ساده‌تر و مؤثرتر
            regime = "NEUTRAL"
            confidence = 0.5
            
            # شرایط روند صعودی
            if sma_20 > sma_50 and rsi > 45 and rsi < 75:
                regime = "BULLISH"
                confidence = min(0.85, 0.5 + (rsi - 45) / 60)
            
            # شرایط روند نزولی
            elif sma_20 < sma_50 and rsi < 55 and rsi > 25:
                regime = "BEARISH" 
                confidence = min(0.85, 0.5 + (55 - rsi) / 60)
            
            # شرایط پرنوسان
            elif volatility > 0.025:
                regime = "VOLATILE"
                confidence = min(0.9, volatility / 0.03)
            
            return {
                'regime': regime,
                'confidence': round(confidence, 2),
                'sma_20': round(sma_20, 4),
                'sma_50': round(sma_50, 4),
                'rsi': round(rsi, 2),
                'volatility': round(volatility, 4)
            }
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return {
                'regime': 'UNKNOWN', 
                'confidence': 0,
                'error': str(e)
            }
    
    @staticmethod
    def calculate_support_resistance(data: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
        """🔥 IMPROVED: محاسبه سطوح حمایت و مقاومت"""
        try:
            high = data['high'].rolling(window).max().iloc[-1]
            low = data['low'].rolling(window).min().iloc[-1]
            close = data['close'].iloc[-1]
            
            # محاسبه Pivot Points
            pivot = (high + low + close) / 3
            resistance = 2 * pivot - low
            support = 2 * pivot - high
            
            # 🔥 اعتبارسنجی
            if support >= close:
                support = close * 0.98
            if resistance <= close:
                resistance = close * 1.02
            
            logger.debug(f"Support: {support:.4f}, Resistance: {resistance:.4f}")
            return support, resistance
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            # 🔥 مقادیر پیش‌فرض منطقی
            current_price = data['close'].iloc[-1]
            return current_price * 0.98, current_price * 1.02
    
    @staticmethod
    def fibonacci_levels(data: pd.DataFrame, period: int = 50) -> Dict[str, float]:
        """🔥 FIXED: محاسبه سطوح فیبوناچی - کامل شده"""
        try:
            high = data['high'].tail(period).max()
            low = data['low'].tail(period).min()
            diff = high - low
            
            # 🔥 بررسی اعتبار
            if diff <= 0:
                logger.warning("Invalid Fibonacci calculation: high <= low")
                current_price = data['close'].iloc[-1]
                return {
                    '0.0': current_price * 0.95,
                    '0.5': current_price,
                    '1.0': current_price * 1.05
                }
            
            levels = {
                '0.0': round(low, 4),
                '0.236': round(low + diff * 0.236, 4),
                '0.382': round(low + diff * 0.382, 4),
                '0.5': round(low + diff * 0.5, 4),
                '0.618': round(low + diff * 0.618, 4),
                '0.786': round(low + diff * 0.786, 4),
                '1.0': round(high, 4)
            }
            
            logger.debug(f"Fibonacci levels calculated: {levels}")
            return levels
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {e}")
            # 🔥 مقادیر پیش‌فرض
            current_price = data['close'].iloc[-1]
            return {
                '0.0': current_price * 0.95,
                '0.5': current_price,
                '1.0': current_price * 1.05
            }
    
    @staticmethod
    def calculate_trend_strength(data: pd.DataFrame) -> Dict[str, Any]:
        """🔥 NEW: محاسبه قدرت روند"""
        try:
            close = data['close']
            
            # میانگین‌های متحرک
            ema_9 = close.ewm(span=9).mean()
            ema_21 = close.ewm(span=21).mean()
            ema_50 = close.ewm(span=50).mean()
            
            # محاسبه شیب
            slope_9 = (ema_9.iloc[-1] - ema_9.iloc[-10]) / ema_9.iloc[-10] if len(ema_9) > 10 else 0
            slope_21 = (ema_21.iloc[-1] - ema_21.iloc[-10]) / ema_21.iloc[-10] if len(ema_21) > 10 else 0
            
            # تعیین قدرت روند
            if slope_9 > 0 and slope_21 > 0:
                direction = "UPTREND"
                strength = min(1.0, (abs(slope_9) + abs(slope_21)) * 100)
            elif slope_9 < 0 and slope_21 < 0:
                direction = "DOWNTREND"
                strength = min(1.0, (abs(slope_9) + abs(slope_21)) * 100)
            else:
                direction = "SIDEWAYS"
                strength = 0.3
            
            return {
                'direction': direction,
                'strength': round(strength, 3),
                'slope_9': round(slope_9, 4),
                'slope_21': round(slope_21, 4)
            }
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return {
                'direction': 'UNKNOWN',
                'strength': 0,
                'error': str(e)
            }
    
    @staticmethod
    def detect_divergence(data: pd.DataFrame) -> Dict[str, Any]:
        """🔥 NEW: تشخیص واگرایی RSI-Price"""
        try:
            if 'RSI' not in data.columns:
                return {'divergence': 'NONE', 'type': None}
            
            close = data['close'].tail(20)
            rsi = data['RSI'].tail(20)
            
            # بررسی واگرایی صعودی (Bullish)
            if close.iloc[-1] < close.iloc[0] and rsi.iloc[-1] > rsi.iloc[0]:
                return {
                    'divergence': 'BULLISH',
                    'type': 'Regular',
                    'strength': 0.7
                }
            
            # بررسی واگرایی نزولی (Bearish)
            if close.iloc[-1] > close.iloc[0] and rsi.iloc[-1] < rsi.iloc[0]:
                return {
                    'divergence': 'BEARISH',
                    'type': 'Regular',
                    'strength': 0.7
                }
            
            return {'divergence': 'NONE', 'type': None}
            
        except Exception as e:
            logger.error(f"Error detecting divergence: {e}")
            return {'divergence': 'ERROR', 'error': str(e)}
    
    @staticmethod
    def calculate_volatility_bands(data: pd.DataFrame, period: int = 20, std_dev: int = 2) -> Dict[str, float]:
        """🔥 NEW: محاسبه باندهای نوسان"""
        try:
            close = data['close']
            
            # Bollinger Bands
            sma = close.rolling(period).mean().iloc[-1]
            std = close.rolling(period).std().iloc[-1]
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            # موقعیت قیمت نسبت به باندها
            current_price = close.iloc[-1]
            band_position = (current_price - lower_band) / (upper_band - lower_band)
            
            return {
                'upper_band': round(upper_band, 4),
                'middle_band': round(sma, 4),
                'lower_band': round(lower_band, 4),
                'band_width': round(upper_band - lower_band, 4),
                'price_position': round(band_position, 3)  # 0 to 1
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility bands: {e}")
            current_price = data['close'].iloc[-1]
            return {
                'upper_band': current_price * 1.02,
                'middle_band': current_price,
                'lower_band': current_price * 0.98,
                'band_width': current_price * 0.04,
                'price_position': 0.5
            }
    
    @staticmethod
    def get_market_strength_score(data: pd.DataFrame) -> float:
        """🔥 NEW: محاسبه امتیاز قدرت بازار (0-10)"""
        try:
            score = 5.0  # نقطه شروع
            
            # بررسی RSI
            if 'RSI' in data.columns:
                rsi = data['RSI'].iloc[-1]
                if 30 < rsi < 70:
                    score += 1.0
                if 40 < rsi < 60:
                    score += 0.5
            
            # بررسی حجم
            if 'volume' in data.columns and len(data) > 20:
                avg_volume = data['volume'].rolling(20).mean().iloc[-1]
                current_volume = data['volume'].iloc[-1]
                if current_volume > avg_volume:
                    score += 1.0
            
            # بررسی روند
            trend = AdvancedMarketFilters.calculate_trend_strength(data)
            if trend['direction'] in ['UPTREND', 'DOWNTREND']:
                score += trend['strength'] * 2
            
            # بررسی نوسان
            volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
            if 0.01 < volatility < 0.03:  # نوسان مطلوب
                score += 1.0
            
            return min(10.0, max(0.0, round(score, 1)))
            
        except Exception as e:
            logger.error(f"Error calculating market strength: {e}")
            return 5.0