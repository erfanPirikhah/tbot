"""
Enhanced Signal Contradiction Detection Module
Implements advanced contradiction detection between multiple indicators
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class SignalContradictionDetector:
    """
    Advanced contradiction detection system that identifies conflicts between indicators
    """
    
    def __init__(self):
        self.contradiction_thresholds = {
            'strong': 0.7,
            'moderate': 0.5,
            'weak': 0.3
        }
    
    def detect_rsi_price_divergence(self, data: pd.DataFrame, position_type: str) -> Tuple[bool, str, float]:
        """
        Detect RSI-price divergence
        """
        try:
            if 'RSI' not in data.columns or len(data) < 10:
                return False, "Insufficient RSI data", 0.0
            
            close = data['close']
            rsi = data['RSI']
            
            # Get recent values
            recent_close = close.tail(5)
            recent_rsi = rsi.tail(5)
            
            # Calculate trends
            close_trend = (recent_close.iloc[-1] - recent_close.iloc[0]) / recent_close.iloc[0]
            rsi_trend = (recent_rsi.iloc[-1] - recent_rsi.iloc[0]) / recent_rsi.iloc[0]
            
            # Check for divergence
            if position_type == 'LONG':
                # For LONG: bullish price move but bearish RSI move (or vice versa)
                divergence = (close_trend > 0.01 and rsi_trend < -0.01) or (close_trend < -0.01 and rsi_trend > 0.01)
            else:  # SHORT
                # For SHORT: bearish price move but bullish RSI move (or vice versa)  
                divergence = (close_trend < -0.01 and rsi_trend > 0.01) or (close_trend > 0.01 and rsi_trend < -0.01)
            
            # Calculate divergence strength
            strength = abs(close_trend - rsi_trend) if divergence else 0
            strength = min(1.0, strength * 50)  # Scale to 0-1
            
            if divergence:
                return True, f"RSI-Price divergence detected (strength: {strength:.2f})", strength
            else:
                return False, "No RSI-Price divergence", 0.0
                
        except Exception as e:
            logger.error(f"Error in RSI-Price divergence detection: {e}")
            return False, f"RSI-Price divergence error: {e}", 0.0

    def detect_macd_signal_conflict(self, data: pd.DataFrame, position_type: str) -> Tuple[bool, str, float]:
        """
        Detect contradiction between MACD and signal line
        """
        try:
            if 'MACD' not in data.columns or 'MACD_Signal' not in data.columns or len(data) < 5:
                return False, "Insufficient MACD data", 0.0
            
            macd = data['MACD']
            signal = data['MACD_Signal']
            histogram = macd - signal
            
            # Get recent values
            recent_histogram = histogram.tail(3)
            
            # Check for conflicting signals
            if position_type == 'LONG':
                # For LONG: MACD below signal (bearish) when expecting bullish
                conflict = macd.iloc[-1] < signal.iloc[-1] and recent_histogram.iloc[-1] < 0
            else:  # SHORT
                # For SHORT: MACD above signal (bullish) when expecting bearish
                conflict = macd.iloc[-1] > signal.iloc[-1] and recent_histogram.iloc[-1] > 0
            
            # Calculate conflict strength
            hist_momentum = (recent_histogram.iloc[-1] - recent_histogram.iloc[0]) / abs(recent_histogram.iloc[0]) if recent_histogram.iloc[0] != 0 else 0
            strength = min(1.0, abs(hist_momentum) * 5) if conflict else 0
            
            if conflict:
                return True, f"MACD-Signal conflict detected (strength: {strength:.2f})", strength
            else:
                return False, "No MACD-Signal conflict", 0.0
                
        except Exception as e:
            logger.error(f"Error in MACD-Signal conflict detection: {e}")
            return False, f"MACD-Signal conflict error: {e}", 0.0

    def detect_trend_indicator_conflict(self, data: pd.DataFrame, position_type: str) -> Tuple[bool, str, float]:
        """
        Detect conflicts between different trend indicators (EMA, ADX, Price Action)
        """
        try:
            if len(data) < 21:
                return False, "Insufficient data for trend analysis", 0.0
            
            close = data['close']
            
            # Calculate EMAs
            ema_fast = close.ewm(span=8).mean()
            ema_slow = close.ewm(span=21).mean()
            
            # Calculate ADX if available
            adx_available = 'ADX' in data.columns
            if adx_available:
                adx = data['ADX']
                adx_value = adx.iloc[-1]
                adx_trend = adx_value > 25  # Strong trend threshold
            else:
                adx_value = 20
                adx_trend = False
            
            # Calculate price trend
            price_trend = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] if len(close) >= 5 else 0
            
            # Determine individual signals
            ema_bullish = ema_fast.iloc[-1] > ema_slow.iloc[-1]
            price_bullish = price_trend > 0
            adx_bullish = (data['close'].iloc[-1] > data['close'].iloc[-2]) if adx_available and adx_value > 25 else None
            
            # Count conflicting signals
            bullish_signals = sum([ema_bullish, price_bullish])
            if adx_available and adx_value > 25:
                if position_type == 'LONG':
                    bullish_signals += 1 if data['high'].iloc[-1] > data['high'].iloc[-2] else 0
                else:
                    bullish_signals += 1 if data['low'].iloc[-1] < data['low'].iloc[-2] else 0
            
            total_signals = 3  # EMA, Price, ADX-direction
            alignment = abs(bullish_signals - (total_signals - bullish_signals)) / total_signals
            
            # Conflict detected when alignment is low (mixed signals)
            conflict = alignment < 0.5
            strength = 1.0 - alignment if conflict else 0
            
            if conflict:
                return True, f"Trend indicators conflict (alignment: {alignment:.2f}, strength: {strength:.2f})", strength
            else:
                return False, f"Trend indicators aligned (alignment: {alignment:.2f})", 0.0
                
        except Exception as e:
            logger.error(f"Error in trend indicator conflict detection: {e}")
            return False, f"Trend indicator conflict error: {e}", 0.0

    def detect_volatility_regime_mismatch(self, data: pd.DataFrame, position_type: str) -> Tuple[bool, str, float]:
        """
        Detect mismatch between volatility regime and position type
        """
        try:
            if len(data) < 20:
                return False, "Insufficient data for volatility analysis", 0.0
            
            returns = data['close'].pct_change().dropna().tail(20)
            current_vol = returns.std() if len(returns) > 0 else 0.01
            
            # Calculate historical volatility
            historical_vols = data['close'].pct_change().rolling(50).std().dropna()
            if len(historical_vols) == 0:
                return False, "Insufficient historical volatility data", 0.0
            
            avg_vol = historical_vols.mean()
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
            
            # Determine if volatility regime matches position
            high_vol = vol_ratio > 1.5
            low_vol = vol_ratio < 0.7
            
            # High volatility might not be ideal for mean reversion (oversold/overbought)
            # Low volatility might not be ideal for trend following
            if ((position_type == 'LONG' and high_vol) or (position_type == 'SHORT' and high_vol)):
                # For mean-reversion strategies in high volatility, it might be OK
                # But if we're using momentum/RSI signals, high vol could be problematic
                conflict = 'RSI' in data.columns and data['RSI'].iloc[-1] > 65  # Overbought
                if conflict:
                    strength = min(1.0, vol_ratio - 1.5)
                    return True, f"High volatility with reversal signal (strength: {strength:.2f})", strength
            elif ((position_type == 'LONG' and low_vol) or (position_type == 'SHORT' and low_vol)):
                # For trend-following in low volatility, might not be ideal
                conflict = True  # Generally, low vol is not good for trend strategies
                strength = min(0.7, 1.0 - vol_ratio) if vol_ratio < 0.7 else 0
                return True, f"Low volatility for trend strategy (strength: {strength:.2f})", strength
            else:
                return False, "Volatility regime matches position type", 0.0
            
            return False, "No volatility regime mismatch", 0.0
            
        except Exception as e:
            logger.error(f"Error in volatility regime mismatch detection: {e}")
            return False, f"Volatility regime error: {e}", 0.0

    def detect_timeframe_contradiction(self, data: pd.DataFrame, position_type: str) -> Tuple[bool, str, float]:
        """
        Detect contradictions between different timeframe signals if available
        """
        try:
            contradicted_htf = []
            total_htf = 0
            conflict_count = 0
            
            # Check available HTF data
            htf_timeframes = ['H4', 'D1', 'H1']
            
            for tf in htf_timeframes:
                rsi_col = f'RSI_{tf}'
                ema_fast_col = f'EMA_21_{tf}'
                ema_slow_col = f'EMA_50_{tf}'
                
                if all(col in data.columns for col in [rsi_col, ema_fast_col, ema_slow_col]):
                    total_htf += 1
                    
                    # Check HTF trend
                    htf_ema_fast = data[ema_fast_col].iloc[-1]
                    htf_ema_slow = data[ema_slow_col].iloc[-1]
                    htf_rsi = data[rsi_col].iloc[-1]
                    
                    # Determine HTF signal
                    if position_type == 'LONG':
                        htf_bullish = htf_ema_fast > htf_ema_slow and htf_rsi > 50
                    else:
                        htf_bullish = htf_ema_fast < htf_ema_slow and htf_rsi < 50
                    
                    # Check if HTF contradicts base timeframe signal
                    if position_type == 'LONG':
                        base_bullish = data.get('RSI', pd.Series([50])).iloc[-1] < 40  # Oversold
                        if base_bullish and not htf_bullish:
                            contradicted_htf.append(tf)
                            conflict_count += 1
                    else:
                        base_bearish = data.get('RSI', pd.Series([50])).iloc[-1] > 60  # Overbought
                        if base_bearish and not htf_bullish:
                            contradicted_htf.append(tf)
                            conflict_count += 1
            
            if total_htf > 0:
                conflict_ratio = conflict_count / total_htf
                strength = min(1.0, conflict_ratio * 2)  # Amplify conflict ratio
                
                if strength > 0:
                    return True, f"HTF contradiction on {conflict_count}/{total_htf} timeframes: {contradicted_htf} (strength: {strength:.2f})", strength
                else:
                    return False, "No HTF contradiction detected", 0.0
            else:
                return False, "No HTF data available", 0.0
                
        except Exception as e:
            logger.error(f"Error in timeframe contradiction detection: {e}")
            return False, f"Timeframe contradiction error: {e}", 0.0

    def detect_all_contradictions(self, data: pd.DataFrame, position_type: str) -> Dict[str, Any]:
        """
        Run all contradiction detection methods and return comprehensive results
        """
        try:
            # Initialize results dictionary
            results = {
                'total_contradictions': 0,
                'contradiction_score': 0.0,
                'contradiction_types': [],
                'details': {},
                'risk_level': 'LOW'  # LOW, MEDIUM, HIGH
            }
            
            # Run all contradiction detection methods
            contradictions = []
            
            # 1. RSI-Price Divergence
            has_div, div_desc, div_strength = self.detect_rsi_price_divergence(data, position_type)
            if has_div:
                contradictions.append(('RSI_PRICE_DIVERGENCE', div_desc, div_strength))
                results['details']['rsi_price_divergence'] = {
                    'exists': True,
                    'description': div_desc,
                    'strength': div_strength
                }
            else:
                results['details']['rsi_price_divergence'] = {
                    'exists': False,
                    'description': div_desc,
                    'strength': div_strength
                }
            
            # 2. MACD-Signal Conflict
            has_conf, conf_desc, conf_strength = self.detect_macd_signal_conflict(data, position_type)
            if has_conf:
                contradictions.append(('MACD_SIGNAL_CONFLICT', conf_desc, conf_strength))
                results['details']['macd_signal_conflict'] = {
                    'exists': True,
                    'description': conf_desc,
                    'strength': conf_strength
                }
            else:
                results['details']['macd_signal_conflict'] = {
                    'exists': False,
                    'description': conf_desc,
                    'strength': conf_strength
                }
            
            # 3. Trend Indicator Conflict
            has_trend_conf, trend_conf_desc, trend_conf_strength = self.detect_trend_indicator_conflict(data, position_type)
            if has_trend_conf:
                contradictions.append(('TREND_INDICATOR_CONFLICT', trend_conf_desc, trend_conf_strength))
                results['details']['trend_indicator_conflict'] = {
                    'exists': True,
                    'description': trend_conf_desc,
                    'strength': trend_conf_strength
                }
            else:
                results['details']['trend_indicator_conflict'] = {
                    'exists': False,
                    'description': trend_conf_desc,
                    'strength': trend_conf_strength
                }
            
            # 4. Volatility Regime Mismatch
            has_vol_conf, vol_conf_desc, vol_conf_strength = self.detect_volatility_regime_mismatch(data, position_type)
            if has_vol_conf:
                contradictions.append(('VOLATILITY_REGIME_MISMATCH', vol_conf_desc, vol_conf_strength))
                results['details']['volatility_regime_mismatch'] = {
                    'exists': True,
                    'description': vol_conf_desc,
                    'strength': vol_conf_strength
                }
            else:
                results['details']['volatility_regime_mismatch'] = {
                    'exists': False,
                    'description': vol_conf_desc,
                    'strength': vol_conf_strength
                }
            
            # 5. Timeframe Contradiction
            has_tf_conf, tf_conf_desc, tf_conf_strength = self.detect_timeframe_contradiction(data, position_type)
            if has_tf_conf:
                contradictions.append(('TIMEFRAME_CONTRADICTION', tf_conf_desc, tf_conf_strength))
                results['details']['timeframe_contradiction'] = {
                    'exists': True,
                    'description': tf_conf_desc,
                    'strength': tf_conf_strength
                }
            else:
                results['details']['timeframe_contradiction'] = {
                    'exists': False,
                    'description': tf_conf_desc,
                    'strength': tf_conf_strength
                }
            
            # Calculate overall contradiction metrics
            results['total_contradictions'] = len(contradictions)
            if contradictions:
                avg_strength = sum([c[2] for c in contradictions]) / len(contradictions)
                max_strength = max([c[2] for c in contradictions])
                
                results['contradiction_score'] = min(1.0, avg_strength * 1.5)  # Slightly amplify
                results['contradiction_types'] = [c[0] for c in contradictions]
                
                # Determine risk level
                if max_strength > 0.7:
                    results['risk_level'] = 'HIGH'
                elif max_strength > 0.4 or len(contradictions) >= 3:
                    results['risk_level'] = 'MEDIUM'
                else:
                    results['risk_level'] = 'LOW'
            else:
                results['contradiction_score'] = 0.0
                results['risk_level'] = 'LOW'
            
            # Add recommendations based on contradiction level
            if results['risk_level'] == 'HIGH':
                results['recommendation'] = "HIGH RISK: Do not take this trade due to strong contradictions"
            elif results['risk_level'] == 'MEDIUM':
                results['recommendation'] = "MEDIUM RISK: Consider reducing position size or waiting for better alignment"
            else:
                results['recommendation'] = "LOW RISK: Contradictions are minimal, trade may be considered"
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive contradiction detection: {e}")
            return {
                'total_contradictions': 0,
                'contradiction_score': 0.0,
                'contradiction_types': [],
                'details': {'error': str(e)},
                'risk_level': 'UNKNOWN',
                'recommendation': 'Error in contradiction detection'
            }

    def calculate_signal_quality_score(self, 
                                     data: pd.DataFrame, 
                                     position_type: str, 
                                     contradictions: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate an overall signal quality score considering all contradictions
        """
        try:
            base_score = 1.0  # Start with perfect score
            
            # Deduct points for different types of contradictions
            contradiction_score = contradictions.get('contradiction_score', 0.0)
            
            # Apply deductions based on contradiction strength
            signal_quality = max(0.0, base_score - contradiction_score * 0.8)  # Max 80% reduction
            
            # Additional factors that might affect quality
            if 'RSI' in data.columns:
                current_rsi = data['RSI'].iloc[-1]
                # Deduct if RSI is not strongly in oversold/overbought territory
                if position_type == 'LONG' and not (20 <= current_rsi <= 40):
                    signal_quality *= 0.9
                elif position_type == 'SHORT' and not (60 <= current_rsi <= 80):
                    signal_quality *= 0.9
            
            # Adjust based on volume confirmation (if available)
            if 'volume' in data.columns and len(data) > 10:
                avg_vol = data['volume'].rolling(10).mean().iloc[-1]
                current_vol = data['volume'].iloc[-1]
                if current_vol < avg_vol * 0.7:  # Low volume
                    signal_quality *= 0.95
            
            quality_metrics = {
                'base_score': base_score,
                'contradiction_impact': contradiction_score * 0.8,
                'final_signal_quality': signal_quality,
                'contradiction_penalty': contradiction_score * 0.8
            }
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error calculating signal quality score: {e}")
            return {
                'base_score': 1.0,
                'contradiction_impact': 0.0,
                'final_signal_quality': 0.5,
                'contradiction_penalty': 0.0
            }

class EnhancedContradictionSystem:
    """
    Comprehensive contradiction detection system with integrated risk management
    """
    
    def __init__(self):
        self.detector = SignalContradictionDetector()
        self.active_contradictions = []
    
    def analyze_signal_safety(self, 
                            data: pd.DataFrame, 
                            position_type: str,
                            regime_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Comprehensive signal safety analysis
        """
        try:
            # Run contradiction detection
            contradictions = self.detector.detect_all_contradictions(data, position_type)
            
            # Calculate signal quality
            quality_metrics = self.detector.calculate_signal_quality_score(data, position_type, contradictions)
            
            # Integrate with market regime if provided
            if regime_info:
                regime = regime_info.get('final_regime', 'NORMAL')
                # Adjust safety for regime
                if regime in ['VOLATILE', 'TRANSITION']:
                    quality_metrics['final_signal_quality'] *= 0.9  # Reduce quality in unstable regimes
            
            # Create safety assessment
            safety_assessment = {
                'is_safe': contradictions['risk_level'] in ['LOW', 'MEDIUM'] and quality_metrics['final_signal_quality'] > 0.4,
                'risk_level': contradictions['risk_level'],
                'signal_quality': quality_metrics['final_signal_quality'],
                'contradiction_summary': contradictions,
                'quality_metrics': quality_metrics,
                'position_type': position_type,
                'recommendation': contradictions['recommendation']
            }
            
            if safety_assessment['is_safe']:
                logger.info(f"Signal SAFETY CHECK PASSED for {position_type} (Quality: {quality_metrics['final_signal_quality']:.3f}, Risk: {contradictions['risk_level']})")
            else:
                logger.warning(f"Signal SAFETY CHECK FAILED for {position_type} (Quality: {quality_metrics['final_signal_quality']:.3f}, Risk: {contradictions['risk_level']})")
            
            return safety_assessment
            
        except Exception as e:
            logger.error(f"Error in signal safety analysis: {e}")
            return {
                'is_safe': False,
                'risk_level': 'ERROR',
                'signal_quality': 0.1,
                'contradiction_summary': {'error': str(e)},
                'quality_metrics': {'error': str(e)},
                'position_type': position_type,
                'recommendation': 'Error in safety assessment'
            }
    
    def should_filter_signal(self, safety_assessment: Dict[str, Any], 
                           max_contradiction_score: float = 0.6,
                           min_signal_quality: float = 0.4) -> bool:
        """
        Determine if a signal should be filtered based on contradiction level
        """
        try:
            contradiction_score = safety_assessment['contradiction_summary'].get('contradiction_score', 0)
            signal_quality = safety_assessment['signal_quality']
            
            # Filter if contradictions are too high OR signal quality is too low
            should_filter = contradiction_score > max_contradiction_score or signal_quality < min_signal_quality
            
            return should_filter
            
        except Exception as e:
            logger.error(f"Error in signal filtering decision: {e}")
            return True  # Default to filtering if there's an error

def create_sample_contradiction_data() -> pd.DataFrame:
    """Create sample data for testing contradiction detection"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Create price data with embedded contradictions
    prices = []
    trend = 100
    
    for i in range(100):
        # Create normal market condition for first 30
        if i < 30:
            change = np.random.randn() * 0.3
        # Create potential RSI-price divergence in next 20
        elif i < 50:
            if i % 2 == 0:  # Every other day goes up strongly
                change = 1.5 + np.random.randn() * 0.2
            else:  # But then drops back
                change = -1.0 + np.random.randn() * 0.2
        # Create trend vs RSI conflict in next 20
        elif i < 70:
            change = 0.7 + np.random.randn() * 0.3  # Strong uptrend
            # But RSI will appear disconnected
        # Create ranging period
        else:
            change = np.random.randn() * 0.1
            
        trend += change
        prices.append(max(10, trend))
    
    data = pd.DataFrame({
        'close': prices,
        'high': [p + abs(np.random.randn() * 0.4) for p in prices],
        'low': [p - abs(np.random.randn() * 0.4) for p in prices],
        'open': [p + np.random.randn() * 0.1 for p in prices],
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    }, index=dates)
    
    # Add some technical indicators
    # Calculate RSI with potential divergence
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Add MACD
    exp1 = data['close'].ewm(span=12).mean()
    exp2 = data['close'].ewm(span=26).mean()
    macd = exp1 - exp2
    data['MACD'] = macd
    data['MACD_Signal'] = macd.ewm(span=9).mean()
    
    # Add ADX components
    data['+DI'] = np.random.rand(len(data)) * 30 + 10
    data['-DI'] = np.random.rand(len(data)) * 30 + 10
    data['ADX'] = np.abs(data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI']) * 100
    
    # Add HTF indicators for the last 50 data points
    for tf in ['H4', 'D1']:
        data[f'RSI_{tf}'] = data['RSI'].shift(np.random.randint(1, 5)).fillna(method='bfill')
        data[f'EMA_21_{tf}'] = data['close'].ewm(span=21).mean().shift(np.random.randint(1, 3)).fillna(method='bfill')
        data[f'EMA_50_{tf}'] = data['close'].ewm(span=50).mean().shift(np.random.randint(1, 3)).fillna(method='bfill')
    
    return data

def test_contradiction_detection():
    """Test the contradiction detection functionality"""
    logger.info("Testing Enhanced Signal Contradiction Detection...")
    
    # Create sample data
    data = create_sample_contradiction_data()
    
    # Initialize contradiction detector
    detector = SignalContradictionDetector()
    contradiction_system = EnhancedContradictionSystem()
    
    # Test with recent data
    recent_data = data.tail(50)
    
    # Test LONG position contradiction detection
    contradictions_long = detector.detect_all_contradictions(recent_data, 'LONG')
    logger.info(f"LONG Position Contradictions: {contradictions_long}")
    
    # Test SHORT position contradiction detection
    contradictions_short = detector.detect_all_contradictions(recent_data, 'SHORT')
    logger.info(f"SHORT Position Contradictions: {contradictions_short}")
    
    # Calculate signal quality
    quality_long = detector.calculate_signal_quality_score(recent_data, 'LONG', contradictions_long)
    quality_short = detector.calculate_signal_quality_score(recent_data, 'SHORT', contradictions_short)
    
    logger.info(f"LONG Signal Quality: {quality_long}")
    logger.info(f"SHORT Signal Quality: {quality_short}")
    
    # Test comprehensive safety analysis
    safety_long = contradiction_system.analyze_signal_safety(recent_data, 'LONG')
    safety_short = contradiction_system.analyze_signal_safety(recent_data, 'SHORT')
    
    logger.info(f"LONG Safety Assessment: {safety_long}")
    logger.info(f"SHORT Safety Assessment: {safety_short}")
    
    # Test signal filtering
    filter_long = contradiction_system.should_filter_signal(safety_long)
    filter_short = contradiction_system.should_filter_signal(safety_short)
    
    logger.info(f"Should filter LONG: {filter_long}, SHORT: {filter_short}")
    
    # Simulate regime info for testing
    regime_info = {
        'final_regime': 'TRENDING',
        'overall_confidence': 0.7
    }
    
    safety_with_regime = contradiction_system.analyze_signal_safety(recent_data, 'LONG', regime_info)
    logger.info(f"Safety with regime info: {safety_with_regime}")
    
    logger.info("Contradiction Detection testing completed successfully!")

if __name__ == "__main__":
    test_contradiction_detection()