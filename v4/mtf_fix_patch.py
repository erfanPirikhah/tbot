"""
MTF Module Fix for Enhanced RSI Strategy V5
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any


def _calculate_adaptive_mtf_thresholds(self, data: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate adaptive MTF thresholds based on market conditions
    """
    # In more volatile or ranging markets, relax the MTF requirements
    try:
        if len(data) < 20:
            return self.mtf_long_rsi_min, self.mtf_short_rsi_max
        
        # Calculate volatility measure
        price_volatility = (data['high'].std() + data['low'].std()) / data['close'].mean()
        volatility_factor = max(0.7, min(1.3, price_volatility / 0.01))  # Normalize to typical volatility
        
        # In high volatility, be more permissive with MTF alignment
        adaptive_long_min = self.mtf_long_rsi_min - (5 * (volatility_factor - 1.0))
        adaptive_short_max = self.mtf_short_rsi_max + (5 * (volatility_factor - 1.0))
        
        # Ensure reasonable bounds
        adaptive_long_min = max(30, min(45, adaptive_long_min))
        adaptive_short_max = max(55, min(70, adaptive_short_max))
        
        return adaptive_long_min, adaptive_short_max
        
    except:
        # Return original values if calculation fails
        return self.mtf_long_rsi_min, self.mtf_short_rsi_max


def analyze_alignment_fixed(self, data: pd.DataFrame, position_type: str) -> Dict[str, Any]:
    """
    Enhanced MTF analysis with more adaptive and permissive logic
    """
    try:
        # Get adaptive thresholds
        adaptive_long_min, adaptive_short_max = self._calculate_adaptive_mtf_thresholds(data)
        
        # Update analyzer with adaptive thresholds (in a way that doesn't change the original design)
        # Instead of modifying object attributes, calculate the scores with the adaptive thresholds
        
        messages = []
        component_scores = {}
        alignment_score = 0
        total_weight = 0

        # Define weights for different timeframes
        timeframe_weights = {
            'D1': 1.0,   # Daily highest weight
            'H4': 1.0,   # 4-hour high weight
            'H1': 0.7,   # 1-hour medium weight
            'M30': 0.5,  # 30-min lower weight
            'M15': 0.4,  # 15-min lowest weight
        }

        for tf in self.timeframes:
            weight = timeframe_weights.get(tf, 0.5)  # Default weight
            
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
                # RSI alignment: should be above minimum threshold (using adaptive threshold)
                if has_rsi:
                    rsi_val = float(data[rsi_col].iloc[-1])
                    if rsi_val >= adaptive_long_min:  # More permissive threshold
                        tf_alignment += 0.3  # Good alignment
                        tf_messages.append(f"RSI:{rsi_val:.1f}>={adaptive_long_min}")
                    else:
                        tf_messages.append(f"RSI:{rsi_val:.1f}<{adaptive_long_min}")

                # EMA alignment: Fast should be above slow (but be more lenient)
                if has_ema:
                    ema_fast_val = float(data[ema_fast_col].iloc[-1])
                    ema_slow_val = float(data[ema_slow_col].iloc[-1])
                    if ema_fast_val >= ema_slow_val * 0.98:  # Allow slight differences
                        tf_alignment += 0.4  # Good alignment
                        tf_messages.append("EMA:bullish")
                    else:
                        tf_messages.append("EMA:bullish_close")

                # Trend direction: should be positive
                if has_trend:
                    trend_val = int(data[trend_col].iloc[-1]) if not pd.isna(data[trend_col].iloc[-1]) else 0
                    if trend_val > 0:
                        tf_alignment += 0.3  # Good alignment
                        tf_messages.append("Trend:UP")
                    else:
                        tf_messages.append(f"Trend:{'NEUTRAL' if trend_val==0 else 'DOWN'}")

            else:  # SHORT
                # RSI alignment: should be below maximum threshold (using adaptive threshold)
                if has_rsi:
                    rsi_val = float(data[rsi_col].iloc[-1])
                    if rsi_val <= adaptive_short_max:  # More permissive threshold
                        tf_alignment += 0.3  # Good alignment
                        tf_messages.append(f"RSI:{rsi_val:.1f}<={adaptive_short_max}")
                    else:
                        tf_messages.append(f"RSI:{rsi_val:.1f}>{adaptive_short_max}")

                # EMA alignment: Fast should be below slow (but be more lenient)
                if has_ema:
                    ema_fast_val = float(data[ema_fast_col].iloc[-1])
                    ema_slow_val = float(data[ema_slow_col].iloc[-1])
                    if ema_fast_val <= ema_slow_val * 1.02:  # Allow slight differences
                        tf_alignment += 0.4  # Good alignment
                        tf_messages.append("EMA:bearish")
                    else:
                        tf_messages.append("EMA:bearish_close")

                # Trend direction: should be negative
                if has_trend:
                    trend_val = int(data[trend_col].iloc[-1]) if not pd.isna(data[trend_col].iloc[-1]) else 0
                    if trend_val < 0:
                        tf_alignment += 0.3  # Good alignment
                        tf_messages.append("Trend:DOWN")
                    else:
                        tf_messages.append(f"Trend:{'NEUTRAL' if trend_val==0 else 'UP'}")

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
            # If no HTF data available, return neutral score that doesn't block
            if self.mtf_require_all:
                return {
                    'is_aligned': False,
                    'score': 0.0,
                    'messages': ["No HTF data available - MTF analysis failed"],
                    'component_scores': {},
                    'timeframes_analyzed': self.timeframes,
                    'position_type': position_type,
                    'threshold': 0.5
                }
            else:
                return {
                    'is_aligned': True,
                    'score': 0.5,
                    'messages': ["No HTF data available - MTF analysis skipped (no blocking)"],
                    'component_scores': {},
                    'timeframes_analyzed': [],
                    'position_type': position_type,
                    'threshold': 0.0
                }

        final_score = alignment_score / total_weight

        # In TestMode, be more permissive with the threshold
        effective_threshold = 0.2  # Very low threshold in TestMode
        if not self.test_mode_enabled:
            # In normal mode, use original or slightly relaxed threshold
            effective_threshold = 0.3  # Originally it might have been higher

        is_aligned = final_score >= effective_threshold

        # Add overall score message
        messages.insert(0, f"MTF Composite Score: {final_score:.2f} (threshold: >{effective_threshold})")

        return {
            'is_aligned': is_aligned,
            'score': final_score,
            'messages': messages,
            'component_scores': component_scores,
            'timeframes_analyzed': self.timeframes,
            'position_type': position_type,
            'threshold': effective_threshold
        }

    except Exception as e:
        # If analysis fails, in TestMode we should not block
        if self.test_mode_enabled:
            return {
                'is_aligned': True,
                'score': 0.5,
                'messages': [f"MTF analysis error: {e} (TestMode - allowing)"],
                'component_scores': {},
                'timeframes_analyzed': [],
                'position_type': position_type,
                'threshold': 0.0
            }
        else:
            return {
                'is_aligned': False,
                'score': 0.0,
                'messages': [f"MTF analysis error: {e}"],
                'component_scores': {},
                'timeframes_analyzed': [],
                'position_type': position_type,
                'threshold': 0.5
            }


def apply_mtf_fix_to_module(mtf_module_class):
    """
    Apply MTF fixes by updating the MTF module class methods
    """
    # Add the adaptive threshold calculation method
    mtf_module_class._calculate_adaptive_mtf_thresholds = _calculate_adaptive_mtf_thresholds
    
    # Replace the analyze_alignment method
    mtf_module_class.analyze_alignment = analyze_alignment_fixed
    
    # Update the __init__ method to handle test_mode if needed
    original_init = mtf_module_class.__init__
    
    def new_init(self, 
                 mtf_timeframes=None,
                 mtf_long_rsi_min=40.0,
                 mtf_short_rsi_max=60.0,
                 mtf_require_all=False,  # Changed from True to False for more permissive approach
                 test_mode_enabled=False):
        original_init(self, mtf_timeframes, mtf_long_rsi_min, mtf_short_rsi_max)
        self.mtf_require_all = mtf_require_all  # More permissive by default
        self.test_mode_enabled = test_mode_enabled  # Store test mode for reference
    
    mtf_module_class.__init__ = new_init
    
    return mtf_module_class


# Apply the MTF fixes to the EnhancedMTFModule class
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from strategies.mtf_analyzer import EnhancedMTFModule
apply_mtf_fix_to_module(EnhancedMTFModule)

print("✅ MTF Module Fix Applied")
print("  - Adaptive MTF thresholds based on market volatility") 
print("  - More permissive alignment scoring")
print("  - TestMode flexibility for MTF filtering")
print("  - Better error handling that doesn't block in TestMode")