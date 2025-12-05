"""
Enhanced Trend Filter Module
Implements advanced trend detection with multiple confirmations
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class AdvancedTrendFilter:
    """
    Advanced trend filter with multiple indicator confirmations
    """

    def __init__(self,
                 strength_threshold: float = 0.25,  # Lowered for more permissiveness
                 adx_threshold: float = 15.0,       # Lowered from 20 for more permissiveness
                 use_adx: bool = True,
                 use_ema_alignment: bool = True,
                 use_price_position: bool = True,
                 ema_periods: Tuple[int, int, int] = (8, 21, 50),
                 test_mode_enabled: bool = False):
        """
        Initialize Advanced Trend Filter

        Args:
            strength_threshold: Minimum trend strength required (0.0-1.0)
            adx_threshold: ADX value threshold for trend strength
            use_adx: Whether to use ADX in trend calculation
            use_ema_alignment: Whether to use EMA alignment
            use_price_position: Whether to use price vs EMA position
            ema_periods: EMA periods (fast, medium, slow)
            test_mode_enabled: Whether to use TestMode flexibility
        """
        self.strength_threshold = strength_threshold
        self.adx_threshold = adx_threshold
        self.use_adx = use_adx
        self.use_ema_alignment = use_ema_alignment
        self.use_price_position = use_price_position
        self.ema_fast_period, self.ema_medium_period, self.ema_slow_period = ema_periods
        self.test_mode_enabled = test_mode_enabled  # Store TestMode setting

    def calculate_ema_alignment_score(self, data: pd.DataFrame) -> Tuple[float, str]:
        """
        Calculate trend score based on EMA alignment
        """
        try:
            # Calculate EMAs
            close = data['close']
            ema_fast = close.ewm(span=self.ema_fast_period).mean()
            ema_medium = close.ewm(span=self.ema_medium_period).mean()
            ema_slow = close.ewm(span=self.ema_slow_period).mean()
            
            # Get current values
            e_fast = ema_fast.iloc[-1]
            e_med = ema_medium.iloc[-1]
            e_slow = ema_slow.iloc[-1]
            price = close.iloc[-1]
            
            # Calculate alignment score (0 to 1)
            # Perfect bullish alignment: EMA_fast > EMA_medium > EMA_slow and price > EMAs
            if e_fast > e_med > e_slow and price > e_fast:
                alignment_score = 1.0
                alignment_desc = "Strong bullish alignment"
            elif e_fast < e_med < e_slow and price < e_fast:  # Perfect bearish
                alignment_score = -1.0
                alignment_desc = "Strong bearish alignment"
            elif e_fast > e_med and price > e_med:  # Partial bullish
                alignment_score = 0.6
                alignment_desc = "Bullish alignment"
            elif e_fast < e_med and price < e_med:  # Partial bearish
                alignment_score = -0.6
                alignment_desc = "Bearish alignment"
            else:  # Mixed or sideways
                alignment_score = 0.0
                alignment_desc = "Sideways/weak alignment"
            
            return abs(alignment_score), alignment_desc
            
        except Exception as e:
            logger.error(f"Error in EMA alignment calculation: {e}")
            return 0.0, f"EMA calculation error: {e}"

    def calculate_price_trend_score(self, data: pd.DataFrame) -> Tuple[float, str]:
        """
        Calculate trend score based on price movement
        """
        try:
            close = data['close']
            
            # Calculate short-term trend (last 5 candles)
            if len(close) < 5:
                return 0.0, "Insufficient data for price trend"
            
            recent_5 = close.tail(5).values
            short_trend = (recent_5[-1] - recent_5[0]) / recent_5[0]
            
            # Calculate medium-term trend (last 20 candles) 
            if len(close) >= 20:
                recent_20 = close.tail(20).values
                medium_trend = (recent_20[-1] - recent_20[0]) / recent_20[0]
            else:
                medium_trend = short_trend
            
            # Combine trends with weights
            combined_trend = 0.6 * short_trend + 0.4 * medium_trend
            trend_strength = min(1.0, abs(combined_trend) * 50)  # Scale to 0-1
            
            if combined_trend > 0:
                return trend_strength, f"Bullish price trend ({combined_trend:.4f})"
            else:
                return trend_strength, f"Bearish price trend ({combined_trend:.4f})"
                
        except Exception as e:
            logger.error(f"Error in price trend calculation: {e}")
            return 0.0, f"Price trend calculation error: {e}"

    def calculate_adx_score(self, data: pd.DataFrame) -> Tuple[float, str]:
        """
        Calculate trend score based on ADX
        """
        try:
            if 'ADX' not in data.columns:
                return 0.5, "ADX not available, using neutral score"
            
            adx_value = data['ADX'].iloc[-1]
            
            # ADX strength interpretation
            if adx_value >= 25:
                strength = min(1.0, (adx_value - 25) / 50)  # Scale 25-75 to 0-1
                trend_desc = f"Strong trend (ADX: {adx_value:.1f})"
            elif adx_value >= 20:
                strength = (adx_value - 20) / 5  # Scale 20-25 to 0-1
                trend_desc = f"Moderate trend (ADX: {adx_value:.1f})"
            else:
                strength = 0.0
                trend_desc = f"Weak trend (ADX: {adx_value:.1f})"
                
            return strength, trend_desc
            
        except Exception as e:
            logger.error(f"Error in ADX calculation: {e}")
            return 0.0, f"ADX calculation error: {e}"

    def calculate_macd_trend_score(self, data: pd.DataFrame) -> Tuple[float, str]:
        """
        Calculate trend score based on MACD
        """
        try:
            if 'MACD' not in data.columns or 'MACD_Signal' not in data.columns:
                return 0.0, "MACD not available"
            
            macd = data['MACD'].iloc[-1]
            signal = data['MACD_Signal'].iloc[-1]
            macd_histogram = macd - signal
            
            # MACD trend strength
            histogram_strength = min(1.0, abs(macd_histogram) * 100)  # Scale histogram to 0-1
            
            if macd > signal:
                return histogram_strength, f"Bullish MACD ({macd_histogram:.4f})"
            else:
                return histogram_strength, f"Bearish MACD ({macd_histogram:.4f})"
                
        except Exception as e:
            logger.error(f"Error in MACD calculation: {e}")
            return 0.0, f"MACD calculation error: {e}"

    def calculate_trend_strength(self, data: pd.DataFrame, position_type: str) -> Tuple[float, str, Dict[str, float]]:
        """
        Calculate comprehensive trend strength score
        
        Args:
            data: Market data
            position_type: 'LONG' or 'SHORT'
            
        Returns:
            Tuple of (strength_score, description, component_scores)
        """
        component_scores = {}
        
        # Calculate individual scores
        if self.use_ema_alignment:
            ema_score, ema_desc = self.calculate_ema_alignment_score(data)
            component_scores['ema_alignment'] = ema_score
        else:
            ema_score, ema_desc = 0.0, "EMA alignment disabled"
            
        price_score, price_desc = self.calculate_price_trend_score(data)
        component_scores['price_trend'] = price_score
        
        if self.use_adx:
            adx_score, adx_desc = self.calculate_adx_score(data)
            component_scores['adx'] = adx_score
        else:
            adx_score, adx_desc = 0.0, "ADX disabled"
        
        # Add MACD score
        macd_score, macd_desc = self.calculate_macd_trend_score(data)
        component_scores['macd'] = macd_score
        
        # Calculate composite score (weighted average)
        total_weight = 0
        weighted_score = 0
        
        if self.use_ema_alignment:
            weighted_score += ema_score * 0.4
            total_weight += 0.4
            
        weighted_score += price_score * 0.3
        total_weight += 0.3
        
        if self.use_adx:
            weighted_score += adx_score * 0.2
            total_weight += 0.2
            
        weighted_score += macd_score * 0.1
        total_weight += 0.1
        
        if total_weight == 0:
            composite_score = 0.0
        else:
            composite_score = weighted_score / total_weight
        
        # Adjust for position type (higher score if trend aligns with position)
        if position_type == 'LONG':
            trend_aligned = ema_score > 0.3 or price_score > 0.2 or macd_score > 0.1
        else:  # SHORT
            trend_aligned = ema_score > 0.3 or price_score > 0.2 or macd_score > 0.1
            
        if trend_aligned:
            # Boost score if trends align with position
            composite_score = min(1.0, composite_score * 1.2)
        
        desc_parts = [f"Composite strength: {composite_score:.3f}"]
        desc_parts.append(f"EMA: {ema_desc}")
        desc_parts.append(f"Price: {price_desc}")
        desc_parts.append(f"ADX: {adx_desc}")
        desc_parts.append(f"MACD: {macd_desc}")
        
        description = " | ".join(desc_parts)
        
        return composite_score, description, component_scores

    def _calculate_adaptive_trend_thresholds(self, data: pd.DataFrame) -> float:
        """
        Calculate adaptive trend thresholds based on market conditions
        """
        try:
            # In trending markets, require stronger trend confirmation
            # In ranging markets, be more permissive with trend requirements
            if len(data) < 50:
                return self.strength_threshold  # Use original threshold if insufficient data

            # Calculate volatility measure
            price_change = data['close'].pct_change().rolling(20).std()
            current_volatility = price_change.iloc[-1] if not pd.isna(price_change.iloc[-1]) else 0.01

            # Calculate trend strength in the market
            close = data['close']
            ema_fast = close.ewm(span=8).mean()
            ema_slow = close.ewm(span=21).mean()

            # How consistently aligned are the EMAs?
            alignment_periods = 0
            total_periods = min(20, len(data))

            for i in range(1, total_periods):
                idx = -i
                if ema_fast.iloc[idx] > ema_slow.iloc[idx]:
                    alignment_periods += 1

            trend_consistency = alignment_periods / total_periods if total_periods > 0 else 0.5

            # Adjust threshold based on market condition
            if current_volatility > 0.02:  # High volatility market
                # In high volatility, be more permissive about trend requirements
                adaptive_threshold = max(0.15, self.strength_threshold * 0.6)
            elif trend_consistency > 0.8:  # Strong trend market
                # In strong trend market, maintain stricter requirements
                adaptive_threshold = min(0.5, self.strength_threshold * 1.0)
            else:  # Ranging or weak trend market
                # In ranging market, be more permissive
                adaptive_threshold = max(0.1, self.strength_threshold * 0.5)

            return adaptive_threshold

        except:
            # Return original threshold if calculation fails
            return self.strength_threshold

    def evaluate_trend(self, data: pd.DataFrame, position_type: str) -> Tuple[bool, str, float, Dict[str, float]]:
        """
        Evaluate if the trend supports the given position type with adaptive thresholds

        Args:
            data: Market data
            position_type: 'LONG' or 'SHORT'

        Returns:
            Tuple of (is_supporting, description, strength, component_scores)
        """
        # Calculate adaptive threshold
        adaptive_threshold = self._calculate_adaptive_trend_thresholds(data)

        strength, description, component_scores = self.calculate_trend_strength(data, position_type)

        # Calculate if trend supports the position
        # In TestMode, be more permissive
        if self.test_mode_enabled:
            # In TestMode, lower the threshold significantly to allow more signals
            effective_threshold = max(0.1, adaptive_threshold * 0.4)  # Very permissive in TestMode
            is_supporting = strength >= effective_threshold
        else:
            # In normal mode, use adaptive threshold
            is_supporting = strength >= adaptive_threshold

        # Add position alignment check with more nuanced logic
        ema_alignment_score = component_scores.get('ema_alignment', 0)
        price_trend_score = component_scores.get('price_trend', 0)
        adx_score = component_scores.get('adx', 0)
        macd_score = component_scores.get('macd', 0)

        # Determine if trend is aligned with position
        if position_type == 'LONG':
            # For LONG, we want bullish signals
            bullish_signals = sum([
                ema_alignment_score > 0.2,
                price_trend_score > 0.15,
                macd_score > 0.05
            ])
            trend_aligned_with_position = bullish_signals >= 1  # Require at least 1 bullish signal
        else:  # SHORT
            # For SHORT, we want bearish signals
            bearish_signals = sum([
                ema_alignment_score > 0.2,
                price_trend_score > 0.15,
                macd_score > 0.05
            ])
            trend_aligned_with_position = bearish_signals >= 1  # Require at least 1 bearish signal

        # Adjust support based on alignment
        if trend_aligned_with_position:
            is_supporting = True  # Override threshold if trend is aligned with position
            # Update strength to reflect the alignment
            strength = max(strength, 0.4)  # Boost strength if aligned

        if not is_supporting:
            if self.test_mode_enabled:
                # In TestMode, allow even weak trends with warning
                is_supporting = True
                result_desc = f"Trend weak but allowing in TestMode (strength: {strength:.3f}, adaptive_threshold: {adaptive_threshold:.3f})"
            else:
                result_desc = f"Trend NOT supporting {position_type} (strength: {strength:.3f}, adaptive_threshold: {adaptive_threshold:.3f})"
        else:
            result_desc = f"Trend supporting {position_type} (strength: {strength:.3f}, adaptive_threshold: {adaptive_threshold:.3f}) - {description}"

        return is_supporting, result_desc, strength, component_scores

class TrendRegimeDetector:
    """
    Advanced trend regime detection with market condition awareness
    """
    
    def __init__(self):
        self.trend_filter = AdvancedTrendFilter()
    
    def detect_regime(self, data: pd.DataFrame) -> Tuple[str, float, Dict[str, Any]]:
        """
        Detect current trend regime
        
        Returns:
            Tuple of (regime_type, confidence, details)
        """
        try:
            # Calculate trend components
            strength, _, component_scores = self.trend_filter.calculate_trend_strength(data, 'LONG')  # Just for calculation
            
            # Calculate volatility for regime detection
            if len(data) >= 20:
                returns = data['close'].pct_change().tail(20).dropna()
                volatility = returns.std() if len(returns) > 0 else 0.01
                
                # Calculate trend consistency
                close = data['close']
                if len(close) >= 50:
                    ema_fast = close.ewm(span=8).mean()
                    ema_slow = close.ewm(span=21).mean()
                    
                    trend_consistency = abs(ema_fast.iloc[-1] - ema_slow.iloc[-1]) / close.iloc[-1]
                else:
                    trend_consistency = 0.0
            else:
                volatility = 0.01
                trend_consistency = 0.0
            
            # Determine regime based on trend strength and volatility
            if trend_consistency > 0.008 and volatility < 0.02:
                regime = "STRONG_TREND"
                confidence = min(0.9, strength * 0.8 + volatility * 0.2)
            elif trend_consistency > 0.005 and volatility < 0.03:
                regime = "MODERATE_TREND"
                confidence = min(0.8, strength * 0.7 + volatility * 0.3)
            elif volatility > 0.025 and trend_consistency < 0.005:
                regime = "HIGH_VOLATILITY"
                confidence = 0.6
            elif trend_consistency < 0.003 and volatility < 0.008:
                regime = "RANGING"
                confidence = 0.7
            else:
                regime = "TRANSITION"
                confidence = 0.5
            
            details = {
                'trend_strength': strength,
                'volatility': volatility,
                'trend_consistency': trend_consistency,
                'component_scores': component_scores
            }
            
            return regime, confidence, details
            
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return "UNKNOWN", 0.3, {'error': str(e)}


def create_sample_trend_data() -> pd.DataFrame:
    """Create sample data for testing trend filter"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Create trending price data
    trend_values = []
    current = 100
    for i in range(100):
        # Add some trend with random noise
        if i < 30:
            change = 0.5 + np.random.randn() * 0.3  # Bullish trend
        elif i < 60:
            change = -0.2 + np.random.randn() * 0.4  # Bearish trend
        else:
            change = np.random.randn() * 0.2  # Sideways
        current += change
        trend_values.append(current)
    
    data = pd.DataFrame({
        'close': trend_values,
        'high': [v + abs(np.random.randn() * 0.5) for v in trend_values],
        'low': [v - abs(np.random.randn() * 0.5) for v in trend_values],
        'open': [v + np.random.randn() * 0.1 for v in trend_values],
    }, index=dates)
    
    # Add some technical indicators
    data['ADX'] = 15 + np.random.rand(100) * 25  # ADX between 15-40
    data['MACD'] = np.random.randn(100)
    data['MACD_Signal'] = np.random.randn(100)
    
    return data

def test_trend_filter():
    """Test the trend filter functionality"""
    logger.info("Testing Enhanced Trend Filter...")
    
    # Create sample data
    data = create_sample_trend_data()
    
    # Initialize trend filter
    trend_filter = AdvancedTrendFilter()
    
    # Test LONG position
    current_data = data.tail(50)  # Use recent data
    is_supporting_long, long_desc, long_strength, long_components = trend_filter.evaluate_trend(current_data, 'LONG')
    
    logger.info(f"LONG Position - Supporting: {is_supporting_long}, Strength: {long_strength:.3f}")
    logger.info(f"Description: {long_desc}")
    
    # Test SHORT position
    is_supporting_short, short_desc, short_strength, short_components = trend_filter.evaluate_trend(current_data, 'SHORT')
    
    logger.info(f"SHORT Position - Supporting: {is_supporting_short}, Strength: {short_strength:.3f}")
    logger.info(f"Description: {short_desc}")
    
    # Test regime detection
    regime_detector = TrendRegimeDetector()
    regime, conf, details = regime_detector.detect_regime(current_data)
    
    logger.info(f"Market Regime: {regime} (Confidence: {conf:.2f})")
    logger.info(f"Regime details: {details}")
    
    logger.info("Trend Filter testing completed successfully!")

if __name__ == "__main__":
    test_trend_filter()