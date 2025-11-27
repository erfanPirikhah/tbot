"""
Enhanced Multi-Timeframe Analysis Module
Implements the improved MTF logic from the diagnostic report
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class MTFAnalyzer:
    """
    Enhanced Multi-Timeframe Analysis Module
    Implements weighted timeframe alignment with flexible requirements
    """
    
    def __init__(self, 
                 mtf_long_rsi_min: float = 40.0, 
                 mtf_short_rsi_max: float = 60.0,
                 min_alignment_score: float = 0.3,
                 timeframe_weights: Optional[Dict[str, float]] = None):
        """
        Initialize MTF Analyzer with flexible alignment requirements
        
        Args:
            mtf_long_rsi_min: Minimum RSI for LONG on HTF (was 50, now 40 for flexibility)
            mtf_short_rsi_max: Maximum RSI for SHORT on HTF (was 50, now 60 for flexibility)  
            min_alignment_score: Minimum score required for alignment (was implicit True, now 0.3)
            timeframe_weights: Weighting for different timeframes (H4/D1 highest weight)
        """
        self.mtf_long_rsi_min = mtf_long_rsi_min
        self.mtf_short_rsi_max = mtf_short_rsi_max
        self.min_alignment_score = min_alignment_score
        self.timeframe_weights = timeframe_weights or {
            'D1': 1.0,   # Daily highest weight
            'H4': 1.0,   # 4-hour high weight
            'H1': 0.7,   # 1-hour medium weight
            'M30': 0.5,  # 30-min lower weight
            'M15': 0.4,  # 15-min lowest weight
        }
    
    def calculate_mtf_alignment_score(self, 
                                    data: pd.DataFrame, 
                                    position_type: str) -> Tuple[float, List[str], Dict[str, float]]:
        """
        Calculate weighted MTF alignment score with detailed breakdown
        
        Args:
            data: DataFrame with HTF indicators (RSI_H4, RSI_D1, etc.)
            position_type: 'LONG' or 'SHORT'
            
        Returns:
            Tuple of (alignment_score, detailed_messages, component_scores)
        """
        messages = []
        component_scores = {}
        alignment_score = 0
        total_weight = 0
        
        for tf, weight in self.timeframe_weights.items():
            # Check for HTF columns
            rsi_col = f'RSI_{tf}'
            ema_fast_col = f'EMA_21_{tf}'
            ema_slow_col = f'EMA_50_{tf}'
            trend_col = f'TrendDir_{tf}'
            
            has_rsi = rsi_col in data.columns and not pd.isna(data[rsi_col].iloc[-1])
            has_ema = (
                ema_fast_col in data.columns and ema_slow_col in data.columns and
                not pd.isna(data[ema_fast_col].iloc[-1]) and not pd.isna(data[ema_slow_col].iloc[-1])
            )
            has_trend = trend_col in data.columns and not pd.isna(data[trend_col].iloc[-1])
            
            # Skip if no data for this timeframe
            if not (has_rsi or has_ema or has_trend):
                continue
            
            # Calculate alignment for this timeframe
            tf_alignment = 0
            tf_messages = []
            
            if position_type == 'LONG':
                # RSI alignment: should be above minimum threshold
                if has_rsi:
                    rsi_val = float(data[rsi_col].iloc[-1])
                    if rsi_val > self.mtf_long_rsi_min:
                        tf_alignment += 0.3  # Good alignment
                        tf_messages.append(f"RSI:{rsi_val:.1f}>{self.mtf_long_rsi_min}")
                    else:
                        tf_messages.append(f"RSI:{rsi_val:.1f}≤{self.mtf_long_rsi_min}")
                
                # EMA alignment: Fast should be above slow
                if has_ema:
                    ema_fast_val = float(data[ema_fast_col].iloc[-1])
                    ema_slow_val = float(data[ema_slow_col].iloc[-1])
                    if ema_fast_val >= ema_slow_val:
                        tf_alignment += 0.4  # Good alignment
                        tf_messages.append("EMA:bullish")
                    else:
                        tf_messages.append("EMA:bearish")
                
                # Trend direction: should be positive
                if has_trend:
                    trend_val = int(data[trend_col].iloc[-1])
                    if trend_val > 0:
                        tf_alignment += 0.3  # Good alignment
                        tf_messages.append("Trend:UP")
                    else:
                        tf_messages.append("Trend:DOWN")
            
            else:  # SHORT
                # RSI alignment: should be below maximum threshold
                if has_rsi:
                    rsi_val = float(data[rsi_col].iloc[-1])
                    if rsi_val < self.mtf_short_rsi_max:
                        tf_alignment += 0.3  # Good alignment
                        tf_messages.append(f"RSI:{rsi_val:.1f}<{self.mtf_short_rsi_max}")
                    else:
                        tf_messages.append(f"RSI:{rsi_val:.1f}≥{self.mtf_short_rsi_max}")
                
                # EMA alignment: Fast should be below slow
                if has_ema:
                    ema_fast_val = float(data[ema_fast_col].iloc[-1])
                    ema_slow_val = float(data[ema_slow_col].iloc[-1])
                    if ema_fast_val <= ema_slow_val:
                        tf_alignment += 0.4  # Good alignment
                        tf_messages.append("EMA:bearish")
                    else:
                        tf_messages.append("EMA:bullish")
                
                # Trend direction: should be negative
                if has_trend:
                    trend_val = int(data[trend_col].iloc[-1])
                    if trend_val < 0:
                        tf_alignment += 0.3  # Good alignment
                        tf_messages.append("Trend:DOWN")
                    else:
                        tf_messages.append("Trend:UP")
            
            # Apply timeframe weight
            weighted_alignment = min(1.0, tf_alignment) * weight
            alignment_score += weighted_alignment
            total_weight += weight
            component_scores[tf] = weighted_alignment
            
            if tf_alignment > 0:
                messages.append(f"{tf}: {weighted_alignment:.2f} ({' | '.join(tf_messages)})")
            else:
                messages.append(f"{tf}: 0.0 (no alignment)")
        
        # Calculate final normalized score
        if total_weight == 0:
            return 0.5, ["No HTF data available"], {}  # Neutral score if no data
        
        final_score = alignment_score / total_weight
        
        # Add overall score message
        messages.insert(0, f"MTF Composite Score: {final_score:.2f} (threshold: >{self.min_alignment_score})")
        
        return final_score, messages, component_scores
    
    def is_aligned(self, 
                   data: pd.DataFrame, 
                   position_type: str) -> Tuple[bool, List[str], float, Dict[str, float]]:
        """
        Check if MTF alignment passes requirements
        
        Args:
            data: DataFrame with HTF indicators
            position_type: 'LONG' or 'SHORT'
            
        Returns:
            Tuple of (is_aligned, messages, score, component_scores)
        """
        score, messages, component_scores = self.calculate_mtf_alignment_score(data, position_type)
        is_aligned = score >= self.min_alignment_score
        
        # Add pass/fail message
        result = "PASSED" if is_aligned else "FAILED"
        messages.append(f"MTF Alignment: {result} (score: {score:.2f}, threshold: {self.min_alignment_score:.2f})")
        
        return is_aligned, messages, score, component_scores

class EnhancedMTFModule:
    """
    Main MTF Module that integrates with the strategy
    Provides both analysis and integration capabilities
    """
    
    def __init__(self, 
                 mtf_timeframes: List[str] = None,
                 mtf_long_rsi_min: float = 40.0,
                 mtf_short_rsi_max: float = 60.0):
        """
        Initialize the Enhanced MTF Module
        """
        self.mtf_timeframes = mtf_timeframes or ['H4', 'D1', 'H1']
        self.analyzer = MTFAnalyzer(
            mtf_long_rsi_min=mtf_long_rsi_min,
            mtf_short_rsi_max=mtf_short_rsi_max,
            min_alignment_score=0.3  # More flexible than requiring all timeframes
        )
    
    def analyze_alignment(self, 
                         data: pd.DataFrame, 
                         position_type: str) -> Dict[str, Any]:
        """
        Comprehensive MTF alignment analysis
        
        Args:
            data: Current market data with HTF indicators
            position_type: 'LONG' or 'SHORT'
            
        Returns:
            Dictionary with detailed analysis results
        """
        is_aligned, messages, score, component_scores = self.analyzer.is_aligned(data, position_type)
        
        return {
            'is_aligned': is_aligned,
            'score': score,
            'messages': messages,
            'component_scores': component_scores,
            'timeframes_analyzed': self.mtf_timeframes,
            'position_type': position_type,
            'threshold': self.analyzer.min_alignment_score,
            'analysis_timestamp': pd.Timestamp.now()
        }
    
    def get_recommendation(self, 
                          data: pd.DataFrame, 
                          position_type: str) -> Dict[str, Any]:
        """
        Get MTF-based recommendation for position
        
        Args:
            data: Current market data
            position_type: 'LONG' or 'SHORT'
            
        Returns:
            Dictionary with recommendation and confidence
        """
        analysis = self.analyze_alignment(data, position_type)
        
        # Calculate confidence based on alignment score
        confidence = min(1.0, analysis['score'] * 2) if analysis['is_aligned'] else max(0.0, analysis['score'] * 2 - 1)
        
        # Determine recommendation
        if analysis['is_aligned']:
            recommendation = 'PROCEED' if position_type == 'LONG' else 'PROCEED'
            reason = f"MTF alignment supports {position_type} position (score: {analysis['score']:.2f})"
        else:
            recommendation = 'HOLD'
            reason = f"MTF alignment does not support {position_type} position (score: {analysis['score']:.2f})"
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'reason': reason,
            'analysis': analysis
        }

# Backward compatibility function
def check_mtf_alignment(data: pd.DataFrame, 
                       position_type: str, 
                       mtf_long_rsi_min: float = 40.0, 
                       mtf_short_rsi_max: float = 60.0) -> Tuple[bool, List[str]]:
    """
    Backward compatible function with the original interface but improved logic
    """
    analyzer = MTFAnalyzer(
        mtf_long_rsi_min=mtf_long_rsi_min,
        mtf_short_rsi_max=mtf_short_rsi_max,
        min_alignment_score=0.3
    )
    
    is_aligned, messages, _, _ = analyzer.is_aligned(data, position_type)
    return is_aligned, messages

# Example usage and testing functions
def create_sample_mtf_data() -> pd.DataFrame:
    """Create sample data for testing MTF functionality"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.5) + 0.2,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.5) - 0.2,
        'open': 100 + np.random.randn(100) * 0.1,
    }, index=dates)
    
    # Add HTF indicators (simulated)
    data['RSI_H4'] = 40 + np.random.rand(100) * 30  # RSI between 40-70
    data['RSI_D1'] = 40 + np.random.rand(100) * 30
    data['EMA_21_H4'] = data['close'] * (1 + np.random.randn(100) * 0.02)
    data['EMA_50_H4'] = data['close'] * (1 + np.random.randn(100) * 0.02)
    data['EMA_21_D1'] = data['close'] * (1 + np.random.randn(100) * 0.02)
    data['EMA_50_D1'] = data['close'] * (1 + np.random.randn(100) * 0.02)
    data['TrendDir_H4'] = np.random.choice([-1, 0, 1], 100)
    data['TrendDir_D1'] = np.random.choice([-1, 0, 1], 100)
    
    return data

def test_mtf_module():
    """Test the MTF module functionality"""
    logger.info("Testing Enhanced MTF Module...")
    
    # Create sample data
    data = create_sample_mtf_data()
    current_data = data.tail(1)
    
    # Initialize MTF module
    mtf_module = EnhancedMTFModule()
    
    # Test LONG position
    long_analysis = mtf_module.analyze_alignment(current_data, 'LONG')
    long_rec = mtf_module.get_recommendation(current_data, 'LONG')
    
    logger.info(f"LONG Position - Aligned: {long_analysis['is_aligned']}, Score: {long_analysis['score']:.2f}")
    logger.info(f"LONG Recommendation: {long_rec['recommendation']} (Confidence: {long_rec['confidence']:.2f})")
    
    # Test SHORT position
    short_analysis = mtf_module.analyze_alignment(current_data, 'SHORT')
    short_rec = mtf_module.get_recommendation(current_data, 'SHORT')
    
    logger.info(f"SHORT Position - Aligned: {short_analysis['is_aligned']}, Score: {short_analysis['score']:.2f}")
    logger.info(f"SHORT Recommendation: {short_rec['recommendation']} (Confidence: {short_rec['confidence']:.2f})")
    
    logger.info("MTF Module testing completed successfully!")

if __name__ == "__main__":
    test_mtf_module()