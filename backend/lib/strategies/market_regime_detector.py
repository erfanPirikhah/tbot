"""
Enhanced Market Regime Detection Module
Implements comprehensive market regime classification with adaptive parameters
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    Advanced market regime detection with multiple indicators
    """
    
    def __init__(self):
        self.regime_confidences = {}
    
    def calculate_volatility_regime(self, data: pd.DataFrame) -> Tuple[str, float, Dict[str, float]]:
        """
        Detect market regime based on volatility characteristics
        """
        try:
            returns = data['close'].pct_change().dropna().tail(50)
            
            if len(returns) < 10:
                return "INSUFFICIENT_DATA", 0.3, {"volatility": 0.0}
            
            current_vol = returns.std()
            historical_vols = data['close'].pct_change().rolling(50).std().dropna()
            
            if len(historical_vols) < 10:
                return "NORMAL", 0.6, {"volatility": current_vol}
            
            avg_vol = historical_vols.mean()
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
            
            # Determine volatility regime
            if current_vol > historical_vols.quantile(0.8):
                regime = "HIGH_VOLATILITY"
                confidence = min(0.9, current_vol * 100)
            elif current_vol < historical_vols.quantile(0.2):
                regime = "LOW_VOLATILITY"
                confidence = min(0.8, (1 - current_vol * 100) * 0.8)
            else:
                regime = "NORMAL_VOLATILITY"
                confidence = max(0.5, min(0.7, vol_ratio))
            
            metrics = {
                "volatility": current_vol,
                "vol_ratio_to_avg": vol_ratio,
                "avg_volatility": avg_vol,
                "vol_percentile": stats.percentileofscore(historical_vols, current_vol) / 100.0
            }
            
            return regime, confidence, metrics
            
        except Exception as e:
            logger.error(f"Error in volatility regime detection: {e}")
            return "UNKNOWN", 0.3, {"error": str(e)}
    
    def calculate_trend_regime(self, data: pd.DataFrame) -> Tuple[str, float, Dict[str, float]]:
        """
        Detect market regime based on trend characteristics
        """
        try:
            close = data['close']
            
            if len(close) < 50:
                return "INSUFFICIENT_DATA", 0.3, {"trend_strength": 0.0}
            
            # Calculate EMAs for trend analysis
            ema_fast = close.ewm(span=8).mean()
            ema_medium = close.ewm(span=21).mean()
            ema_slow = close.ewm(span=50).mean()
            
            # Calculate trend strength
            trend_alignment = abs(ema_fast.iloc[-1] - ema_medium.iloc[-1]) / close.iloc[-1]
            price_position = abs(close.iloc[-1] - ema_slow.iloc[-1]) / close.iloc[-1]
            
            # Calculate directional movement
            recent_50 = close.tail(50)
            trend_slope = (recent_50.iloc[-1] - recent_50.iloc[0]) / recent_50.iloc[0]
            
            # Determine trend regime
            if trend_alignment > 0.01 and abs(trend_slope) > 0.02:
                if trend_slope > 0:
                    regime = "BULL_TREND"
                else:
                    regime = "BEAR_TREND"
                confidence = min(0.9, trend_alignment * 50)
            elif trend_alignment < 0.005 and abs(trend_slope) < 0.01:
                regime = "RANGING"
                confidence = 0.8
            else:
                regime = "TRANSITION"
                confidence = max(0.4, min(0.7, trend_alignment * 30))
            
            metrics = {
                "trend_strength": trend_alignment,
                "price_position": price_position,
                "trend_slope": trend_slope,
                "ema_alignment": trend_alignment
            }
            
            return regime, confidence, metrics
            
        except Exception as e:
            logger.error(f"Error in trend regime detection: {e}")
            return "UNKNOWN", 0.3, {"error": str(e)}
    
    def calculate_momentum_regime(self, data: pd.DataFrame) -> Tuple[str, float, Dict[str, float]]:
        """
        Detect market regime based on momentum characteristics
        """
        try:
            close = data['close']
            
            if len(close) < 20:
                return "INSUFFICIENT_DATA", 0.3, {"momentum": 0.0}
            
            # Calculate RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            
            # Calculate momentum
            returns = close.pct_change()
            momentum_5 = (close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else 0
            momentum_10 = (close.iloc[-1] / close.iloc[-10] - 1) if len(close) >= 10 else 0
            
            # Determine momentum regime
            if current_rsi < 30 and momentum_5 < -0.02:
                regime = "MOMENTUM_SELLING"
                confidence = min(0.9, (30 - current_rsi) / 30)
            elif current_rsi > 70 and momentum_5 > 0.02:
                regime = "MOMENTUM_BUYING"
                confidence = min(0.9, (current_rsi - 70) / 30)
            elif 40 <= current_rsi <= 60:
                regime = "NEUTRAL_MOMENTUM"
                confidence = 0.7
            else:
                regime = "MIXED_MOMENTUM"
                confidence = 0.5
            
            metrics = {
                "rsi": current_rsi,
                "momentum_5": momentum_5,
                "momentum_10": momentum_10,
                "momentum_regime": regime
            }
            
            return regime, confidence, metrics
            
        except Exception as e:
            logger.error(f"Error in momentum regime detection: {e}")
            return "UNKNOWN", 0.3, {"error": str(e)}
    
    def calculate_range_regime(self, data: pd.DataFrame) -> Tuple[str, float, Dict[str, float]]:
        """
        Detect market regime based on ranging vs trending characteristics
        """
        try:
            close = data['close']
            
            if len(close) < 30:
                return "INSUFFICIENT_DATA", 0.3, {"range_score": 0.0}
            
            # Calculate ATR for volatility normalization
            high = data['high']
            low = data['low']
            close_vals = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close_vals.shift())
            tr3 = abs(low - close_vals.shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1] if len(true_range) >= 14 else close_vals.iloc[-1] * 0.01
            
            # Calculate price movement vs ATR
            movement_10 = abs(close.iloc[-1] - close.iloc[-10]) if len(close) >= 10 else 0
            range_ratio = movement_10 / atr if atr > 0 else 1
            
            # Calculate HH/LL patterns
            recent_highs = high.tail(20)
            recent_lows = low.tail(20)
            hh_ll_score = self._calculate_hh_ll_score(recent_highs, recent_lows)
            
            # Determine range/trend regime
            if range_ratio < 2 and hh_ll_score < 0.3:
                regime = "STRONG_RANGING"
                confidence = 0.9
            elif range_ratio > 4 and hh_ll_score > 0.7:
                regime = "STRONG_TRENDING"
                confidence = 0.9
            elif range_ratio < 3 and hh_ll_score < 0.5:
                regime = "RANGING"
                confidence = 0.7
            elif range_ratio > 3 and hh_ll_score > 0.5:
                regime = "TRENDING"
                confidence = 0.7
            else:
                regime = "TRANSITION_ZONE"
                confidence = 0.5
            
            metrics = {
                "range_ratio": range_ratio,
                "hh_ll_score": hh_ll_score,
                "atr": atr,
                "movement_10": movement_10
            }
            
            return regime, confidence, metrics
            
        except Exception as e:
            logger.error(f"Error in range regime detection: {e}")
            return "UNKNOWN", 0.3, {"error": str(e)}
    
    def _calculate_hh_ll_score(self, highs: pd.Series, lows: pd.Series) -> float:
        """
        Calculate score for Higher Highs / Lower Lows pattern (trending) vs Flat patterns (ranging)
        """
        try:
            if len(highs) < 10:
                return 0.5
            
            # Calculate HH/LL over different periods
            max_high = highs.max()
            min_low = lows.min()
            current_high = highs.iloc[-1]
            current_low = lows.iloc[-1]
            
            # Trend score based on recent swing patterns
            trend_score = 0
            for i in range(5, len(highs)-5):
                # Check if this is a local high (HH)
                if (highs.iloc[i] > highs.iloc[i-3:i].max() and 
                    highs.iloc[i] > highs.iloc[i+1:i+4].max()):
                    if highs.iloc[i] > highs.iloc[i-10:i-5].max() if i >= 10 else True:
                        trend_score += 1
                # Check if this is a local low (LL)
                if (lows.iloc[i] < lows.iloc[i-3:i].min() and 
                    lows.iloc[i] < lows.iloc[i+1:i+4].min()):
                    if lows.iloc[i] < lows.iloc[i-10:i-5].min() if i >= 10 else True:
                        trend_score += 1
            
            normalized_score = min(1.0, trend_score / 10)  # Normalize to 0-1 range
            return normalized_score
            
        except Exception:
            return 0.5
    
    def detect_regime(self,
                     data: pd.DataFrame,
                     test_mode_enabled: bool = False) -> Tuple[str, float, Dict[str, Any]]:
        """
        Comprehensive regime detection combining all measures with TestMode flexibility
        """
        try:
            # Get individual regime assessments
            vol_regime, vol_conf, vol_metrics = self.calculate_volatility_regime(data)
            trend_regime, trend_conf, trend_metrics = self.calculate_trend_regime(data)
            mom_regime, mom_conf, mom_metrics = self.calculate_momentum_regime(data)
            range_regime, range_conf, range_metrics = self.calculate_range_regime(data)

            # Combine regimes into a composite regime
            regime_scores = {
                vol_regime: vol_conf,
                trend_regime: trend_conf,
                mom_regime: mom_conf,
                range_regime: range_conf
            }

            # Determine primary regime based on highest confidence
            primary_regime = max(regime_scores, key=regime_scores.get)
            overall_confidence = regime_scores[primary_regime]

            # Adjust for regime consistency
            regime_values = list(regime_scores.values())
            consistency_score = 1 - (np.std(regime_values) / np.mean(regime_values)) if np.mean(regime_values) > 0 else 0
            overall_confidence = (overall_confidence + consistency_score) / 2

            # In TestMode, be more permissive with confidence scores
            if test_mode_enabled:
                # Boost confidence slightly in TestMode to ensure regime detection doesn't block signals
                overall_confidence = min(1.0, overall_confidence * 1.5)

            # Final regime classification
            if "HIGH_VOLATILITY" in primary_regime:
                final_regime = "VOLATILE"
            elif "TREND" in primary_regime or "BULL" in primary_regime or "BEAR" in primary_regime:
                final_regime = "TRENDING"
            elif "RANGING" in primary_regime or "RANGE" in primary_regime:
                final_regime = "RANGING"
            elif "MOMENTUM" in primary_regime:
                final_regime = "MOMENTUM"
            elif "INSUFFICIENT_DATA" in primary_regime:
                # In TestMode, if there's insufficient data, assume a neutral regime instead of blocking
                final_regime = "NORMAL" if test_mode_enabled else "INSUFFICIENT_DATA"
                if test_mode_enabled:
                    overall_confidence = 0.5  # Neutral confidence
            else:
                final_regime = "NORMAL"

            # In TestMode, make sure we never return UNKNOWN or INSUFFICIENT_DATA as blocking categories
            if test_mode_enabled and final_regime in ["UNKNOWN", "INSUFFICIENT_DATA"]:
                final_regime = "NORMAL"
                overall_confidence = 0.5

            details = {
                "primary_regime": primary_regime,
                "final_regime": final_regime,
                "individual_regimes": {
                    "volatility_regime": {"type": vol_regime, "confidence": vol_conf},
                    "trend_regime": {"type": trend_regime, "confidence": trend_conf},
                    "momentum_regime": {"type": mom_regime, "confidence": mom_conf},
                    "range_regime": {"type": range_regime, "confidence": range_conf}
                },
                "metrics": {
                    "volatility": vol_metrics,
                    "trend": trend_metrics,
                    "momentum": mom_metrics,
                    "range": range_metrics
                },
                "overall_confidence": overall_confidence,
                "test_mode_enabled": test_mode_enabled
            }

            self.regime_confidences[final_regime] = overall_confidence
            return final_regime, overall_confidence, details

        except Exception as e:
            logger.error(f"Error in comprehensive regime detection: {e}")
            # In TestMode, return a non-blocking regime if there's an error
            if test_mode_enabled:
                return "NORMAL", 0.5, {"error": str(e), "test_mode_enabled": True}
            else:
                return "UNKNOWN", 0.3, {"error": str(e)}

    def get_adaptive_parameters(self, regime: str) -> Dict[str, Any]:
        """
        Get adaptive parameters based on current regime
        """
        # Default parameters
        params = {
            'risk_multiplier': 1.0,
            'stop_loss_multiplier': 1.0,
            'take_profit_multiplier': 1.0,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'min_position_size': 100,
            'max_spread_ratio': 0.002
        }
        
        # Adjust parameters based on regime
        if regime == "VOLATILE":
            params.update({
                'risk_multiplier': 0.7,  # Reduce risk in volatile markets
                'stop_loss_multiplier': 1.3,  # Wider stops
                'take_profit_multiplier': 1.2,  # More conservative targets
                'rsi_oversold': 25,  # Easier to trigger oversold
                'rsi_overbought': 75,  # Easier to trigger overbought
                'min_position_size': 200  # Larger min size due to higher costs
            })
        elif regime == "TRENDING":
            params.update({
                'risk_multiplier': 1.2,  # Increase risk in trending markets
                'stop_loss_multiplier': 0.9,  # Tighter stops
                'take_profit_multiplier': 1.4,  # Larger targets
                'rsi_oversold': 32,  # Harder to trigger oversold
                'rsi_overbought': 68,  # Harder to trigger overbought
            })
        elif regime == "RANGING":
            params.update({
                'risk_multiplier': 0.8,  # Slightly reduce risk
                'stop_loss_multiplier': 1.1,  # Slightly wider stops
                'take_profit_multiplier': 0.9,  # Smaller targets
                'rsi_oversold': 28,  # More sensitive to oversold
                'rsi_overbought': 72,  # More sensitive to overbought
            })
        elif regime == "MOMENTUM":
            params.update({
                'risk_multiplier': 1.1,  # Increase risk during momentum
                'stop_loss_multiplier': 1.0,  # Normal stops
                'take_profit_multiplier': 1.3,  # Capture more momentum
            })
        
        return params

class RegimeAwareStrategyParams:
    """
    Class to handle regime-dependent strategy parameters
    """

    def __init__(self, test_mode_enabled: bool = False):
        self.detector = MarketRegimeDetector()
        self.current_regime = "NORMAL"
        self.current_confidence = 0.5
        self.param_cache = {}
        self.test_mode_enabled = test_mode_enabled  # Store TestMode setting

    def update_regime(self, data: pd.DataFrame) -> Tuple[str, float]:
        """
        Update the current regime based on market data
        """
        regime, confidence, details = self.detector.detect_regime(data, test_mode_enabled=self.test_mode_enabled)
        self.current_regime = regime
        self.current_confidence = confidence
        logger.info(f"Market regime updated: {regime} (confidence: {confidence:.2f}, TestMode: {self.test_mode_enabled})")
        return regime, confidence
    
    def get_current_params(self) -> Dict[str, Any]:
        """
        Get current adaptive parameters based on regime
        """
        params = self.detector.get_adaptive_parameters(self.current_regime)
        return params
    
    def should_adjust_strategy(self) -> bool:
        """
        Determine if strategy should be adjusted based on regime confidence
        """
        return self.current_confidence > 0.6

def create_sample_regime_data() -> pd.DataFrame:
    """Create sample data for testing regime detection"""
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    np.random.seed(42)
    
    # Create different market conditions
    prices = []
    trend = 100
    
    for i in range(200):
        if i < 50:  # Bull trend
            change = 0.8 + np.random.randn() * 0.4
        elif i < 100:  # High volatility
            change = np.random.randn() * 1.2
        elif i < 150:  # Range
            change = np.random.randn() * 0.3
        else:  # Bear trend
            change = -0.5 + np.random.randn() * 0.4
            
        trend += change
        prices.append(trend)
    
    data = pd.DataFrame({
        'close': prices,
        'high': [p + abs(np.random.randn() * 0.5) for p in prices],
        'low': [p - abs(np.random.randn() * 0.5) for p in prices],
        'open': [p + np.random.randn() * 0.1 for p in prices],
    }, index=dates)
    
    return data

def test_regime_detection():
    """Test the regime detection functionality"""
    logger.info("Testing Enhanced Market Regime Detection...")
    
    # Create sample data
    data = create_sample_regime_data()
    
    # Initialize regime detector
    detector = MarketRegimeDetector()
    
    # Test on recent data (last 50 points to have sufficient history)
    recent_data = data.tail(100)
    regime, confidence, details = detector.detect_regime(recent_data)
    
    logger.info(f"Determined Regime: {regime}")
    logger.info(f"Confidence: {confidence:.3f}")
    logger.info(f"Details: {details}")
    
    # Get adaptive parameters
    params = detector.get_adaptive_parameters(regime)
    logger.info(f"Adaptive Parameters for {regime}: {params}")
    
    # Test regime-aware params class
    regime_params = RegimeAwareStrategyParams()
    updated_regime, updated_conf = regime_params.update_regime(recent_data)
    current_params = regime_params.get_current_params()
    
    logger.info(f"Regime-Aware System: {updated_regime} (conf: {updated_conf:.3f})")
    logger.info(f"Current Params: {current_params}")
    logger.info(f"Should adjust strategy: {regime_params.should_adjust_strategy()}")
    
    logger.info("Market Regime Detection testing completed successfully!")

if __name__ == "__main__":
    test_regime_detection()