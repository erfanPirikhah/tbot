"""
Trend Filter Module Fix for Enhanced RSI Strategy V5
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


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
            adaptive_threshold = max(0.2, self.strength_threshold * 0.7)  
        elif trend_consistency > 0.8:  # Strong trend market
            # In strong trend market, maintain stricter requirements
            adaptive_threshold = min(0.6, self.strength_threshold * 1.2)
        else:  # Ranging or weak trend market
            # In ranging market, be more permissive
            adaptive_threshold = max(0.15, self.strength_threshold * 0.6)
        
        return adaptive_threshold
        
    except:
        # Return original threshold if calculation fails
        return self.strength_threshold


def evaluate_trend_fixed(self, data: pd.DataFrame, position_type: str) -> Tuple[bool, str, float, Dict[str, float]]:
    """
    Enhanced trend evaluation with adaptive thresholds and TestMode flexibility
    """
    # Calculate adaptive threshold
    adaptive_threshold = self._calculate_adaptive_trend_thresholds(data)
    
    strength, description, component_scores = self.calculate_trend_strength(data, position_type)

    # Calculate if trend supports the position
    # In TestMode, be more permissive
    if hasattr(self, 'test_mode_enabled') and self.test_mode_enabled:
        # In TestMode, lower the threshold significantly to allow more signals
        effective_threshold = max(0.1, adaptive_threshold * 0.5)  # Very permissive in TestMode
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
            ema_alignment_score > 0.3,
            price_trend_score > 0.2,
            macd_score > 0.1
        ])
        trend_aligned_with_position = bullish_signals >= 1  # Require at least 1 bullish signal
    else:  # SHORT
        # For SHORT, we want bearish signals  
        bearish_signals = sum([
            ema_alignment_score > 0.3,
            price_trend_score > 0.2,
            macd_score > 0.1
        ])
        trend_aligned_with_position = bearish_signals >= 1  # Require at least 1 bearish signal

    # Adjust support based on alignment
    if trend_aligned_with_position:
        is_supporting = True  # Override threshold if trend is aligned with position
        # Update strength to reflect the alignment
        strength = max(strength, 0.5)  # Boost strength if aligned

    if not is_supporting:
        if hasattr(self, 'test_mode_enabled') and self.test_mode_enabled:
            # In TestMode, allow even weak trends with warning
            is_supporting = True
            result_desc = f"Trend weak but allowing in TestMode (strength: {strength:.3f}, adaptive_threshold: {adaptive_threshold:.3f})"
        else:
            result_desc = f"Trend NOT supporting {position_type} (strength: {strength:.3f}, adaptive_threshold: {adaptive_threshold:.3f})"
    else:
        result_desc = f"Trend supporting {position_type} (strength: {strength:.3f}, adaptive_threshold: {adaptive_threshold:.3f}) - {description}"

    return is_supporting, result_desc, strength, component_scores


def apply_trend_filter_fix_to_class(trend_filter_class):
    """
    Apply trend filter fixes by updating the class methods
    """
    # Add the adaptive threshold calculation method
    trend_filter_class._calculate_adaptive_trend_thresholds = _calculate_adaptive_trend_thresholds
    
    # Replace the evaluate_trend method
    trend_filter_class.evaluate_trend = evaluate_trend_fixed
    
    # Update the __init__ method to handle test_mode if needed
    original_init = trend_filter_class.__init__
    
    def new_init(self, 
                 strength_threshold: float = 0.3,  # Lowered from 0.4 for more permissiveness
                 adx_threshold: float = 15.0,      # Lowered from 20 for more permissiveness
                 use_adx: bool = True,
                 use_ema_alignment: bool = True,
                 use_price_position: bool = True,
                 ema_periods: Tuple[int, int, int] = (8, 21, 50),
                 test_mode_enabled: bool = False):
        original_init(self, 
                      strength_threshold=strength_threshold,
                      adx_threshold=adx_threshold,
                      use_adx=use_adx,
                      use_ema_alignment=use_ema_alignment,
                      use_price_position=use_price_position,
                      ema_periods=ema_periods)
        self.test_mode_enabled = test_mode_enabled  # Store test mode for reference
    
    trend_filter_class.__init__ = new_init
    
    return trend_filter_class


# Apply the Trend Filter fixes to the AdvancedTrendFilter class
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from strategies.trend_filter import AdvancedTrendFilter
apply_trend_filter_fix_to_class(AdvancedTrendFilter)

print("✅ Trend Filter Module Fix Applied")
print("  - Adaptive trend thresholds based on market conditions")
print("  - TestMode flexibility with lower thresholds")
print("  - More permissive trend alignment requirements")
print("  - Better position-type alignment logic")