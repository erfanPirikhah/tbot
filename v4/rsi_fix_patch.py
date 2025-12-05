"""
RSI Module Fix for Enhanced RSI Strategy V5
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


def _calculate_adaptive_rsi_thresholds(self, data: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate adaptive RSI thresholds based on market volatility and regime
    """
    # Calculate current market volatility
    if len(data) < 14:
        return self.rsi_oversold, self.rsi_overbought
    
    # Use ATR-based volatility measure for RSI threshold adjustment
    try:
        close_prices = data['close']
        # Calculate 20-period ATR to measure volatility
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - close_prices.shift())
        low_close = abs(data['low'] - close_prices.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=20).mean().iloc[-1]
        
        # Calculate current price level relative to recent range
        recent_high = data['high'].rolling(20).max().iloc[-1]
        recent_low = data['low'].rolling(20).min().iloc[-1]
        price_volatility = (recent_high - recent_low) / close_prices.iloc[-1] if close_prices.iloc[-1] != 0 else 0.01
        
        # Adjust thresholds based on volatility
        volatility_factor = max(0.5, min(2.0, price_volatility / 0.01))  # Normalize to typical volatility
        
        # In high volatility, widen the bands (make them less restrictive)
        # In low volatility, narrow the bands (more sensitive)
        adaptive_oversold = self.rsi_oversold - (5 * (volatility_factor - 1.0))
        adaptive_overbought = self.rsi_overbought + (5 * (volatility_factor - 1.0))
        
        # Ensure thresholds stay within reasonable bounds
        adaptive_oversold = max(15, min(40, adaptive_oversold))
        adaptive_overbought = max(60, min(85, adaptive_overbought))
        
        return adaptive_oversold, adaptive_overbought
        
    except Exception as e:
        # If calculation fails, return original thresholds
        return self.rsi_oversold, self.rsi_overbought


def _check_rsi_conditions(self, data: pd.DataFrame, position_type: str) -> Tuple[bool, str, float]:
    """
    Enhanced RSI condition checking with adaptive thresholds
    """
    if 'RSI' not in data.columns:
        data = self._calculate_rsi(data)
    
    current_rsi = float(data['RSI'].iloc[-1]) if 'RSI' in data.columns else 50.0
    
    # Get adaptive thresholds
    adaptive_oversold, adaptive_overbought = self._calculate_adaptive_rsi_thresholds(data)
    
    # Apply entry buffer to adaptive thresholds
    entry_threshold_buffer = self.rsi_entry_buffer
    
    if position_type == 'LONG':
        # For LONG: RSI should be below (adaptive_oversold + buffer)
        threshold = adaptive_oversold + entry_threshold_buffer
        condition_met = current_rsi <= threshold
        condition_desc = f"RSI {current_rsi:.1f} {'≤' if condition_met else '≻'} {threshold:.1f} (adaptive_oversold {adaptive_oversold:.1f} + buffer {entry_threshold_buffer})"
    else:  # SHORT
        # For SHORT: RSI should be above (adaptive_overbought - buffer) 
        threshold = adaptive_overbought - entry_threshold_buffer
        condition_met = current_rsi >= threshold
        condition_desc = f"RSI {current_rsi:.1f} {'≥' if condition_met else '≺'} {threshold:.1f} (adaptive_overbought {adaptive_overbought:.1f} - buffer {entry_threshold_buffer})"
    
    return condition_met, condition_desc, current_rsi


# Enhanced momentum check that's less restrictive
def _check_momentum_conditions(self, data: pd.DataFrame, position_type: str) -> Tuple[bool, str]:
    """
    Enhanced momentum check with more permissive thresholds
    """
    if len(data) < 3:
        return True, "Insufficient data for momentum check"
    
    # Calculate recent price moves
    if position_type == 'LONG':
        # For LONG, we want to see if there's potential for upward momentum despite recent pullback
        recent_move = (data['close'].iloc[-1] - data['close'].iloc[-3]) / data['close'].iloc[-3]
        # Be more permissive: allow slight pullbacks for long entries
        momentum_ok = recent_move > -0.01  # Allow up to 1% pullback over 3 candles
        momentum_desc = f"Momentum (LONG): {recent_move*100:.2f}% over 3 candles"
    else:  # SHORT
        # For SHORT, we want to see if there's potential for downward momentum
        recent_move = (data['close'].iloc[-3] - data['close'].iloc[-1]) / data['close'].iloc[-3]
        # Allow slight upward moves for short entries
        momentum_ok = recent_move > -0.01  # Allow up to 1% bounce in wrong direction over 3 candles
        momentum_desc = f"Momentum (SHORT): {recent_move*100:.2f}% over 3 candles"
    
    return momentum_ok, momentum_desc


# Apply to the Enhanced Rsi Strategy V5 class methods
def apply_rsi_fix_to_strategy(strategy_class):
    """
    Apply RSI fixes by updating the strategy class methods
    """
    # Replace the methods in the class
    strategy_class._calculate_adaptive_rsi_thresholds = _calculate_adaptive_rsi_thresholds
    strategy_class._check_rsi_conditions = _check_rsi_conditions
    strategy_class._check_momentum_conditions = _check_momentum_conditions
    
    # Update the check_entry_conditions method
    original_check_entry_conditions = strategy_class.check_entry_conditions
    
    def new_check_entry_conditions(self, data: pd.DataFrame, position_type) -> Tuple[bool, List[str]]:
        """Enhanced entry conditions with adaptive RSI thresholds"""
        conditions = []

        try:
            # Calculate or ensure RSI exists
            if 'RSI' not in data.columns:
                data = self._calculate_rsi(data)

            # Use enhanced RSI check with adaptive thresholds
            rsi_condition_met, rsi_condition_desc, current_rsi = self._check_rsi_conditions(data, position_type.value)
            conditions.append(rsi_condition_desc)

            if not rsi_condition_met:
                self._diagnostic_counters['rsi_blocks'] += 1 if hasattr(self, '_diagnostic_counters') else None
                return False, [f"RSI not in entry zone for {position_type.value} ({current_rsi:.1f})"]

            # Use enhanced momentum check that's more permissive
            momentum_ok, momentum_desc = self._check_momentum_conditions(data, position_type.value)
            
            # In TestMode, be more permissive with momentum, otherwise check if we're in normal mode
            if not self.test_mode_enabled and not momentum_ok:
                # Use more restrictive momentum check in normal mode
                if len(data) >= 3:
                    if position_type == 'LONG':
                        recent_move = (data['close'].iloc[-1] - data['close'].iloc[-3]) / data['close'].iloc[-3]
                        if recent_move < -0.015:  # Only fail for more significant pullbacks
                            return False, [f"Significant negative momentum: {recent_move*100:.2f}% over 3 candles"]
                    else:  # SHORT
                        recent_move = (data['close'].iloc[-3] - data['close'].iloc[-1]) / data['close'].iloc[-3]
                        if recent_move < -0.015:  # Only fail for more significant moves against short
                            return False, [f"Significant positive momentum: {recent_move*100:.2f}% over 3 candles"]
                conditions.append(momentum_desc)
            else:
                # In test mode or when momentum is OK, just add condition
                conditions.append(momentum_desc)

            # Apply ENHANCED trend filter if enabled
            if self.trend_filter and self.enable_trend_filter:
                trend_ok, trend_desc, trend_conf, _ = self.trend_filter.evaluate_trend(data, position_type.value)
                
                # In TestMode, be more permissive about trend
                if not self.test_mode_enabled and not trend_ok:
                    return False, [f"Trend filter: {trend_desc}"]
                elif not trend_ok:
                    # In TestMode, log but don't necessarily block
                    conditions.append(f"Trend: {trend_desc} (TestMode - may allow)")
                else:
                    conditions.append(f"Trend: {trend_desc}")

            # Apply ENHANCED MTF analysis if enabled
            if self.mtf_analyzer and self.enable_mtf:
                mtf_result = self.mtf_analyzer.analyze_alignment(data, position_type.value)
                mtf_ok = mtf_result['is_aligned']
                mtf_desc = mtf_result['messages'][-1] if mtf_result['messages'] else "MTF analysis error"
                
                # In TestMode, be more permissive
                if not self.test_mode_enabled and not mtf_ok:
                    return False, [f"MTF filter: {mtf_desc}"]
                elif not mtf_ok:
                    # In TestMode, log but don't necessarily block
                    conditions.append(f"MTF: {mtf_desc} (TestMode - may allow)")
                else:
                    # Include all MTF messages for transparency
                    for msg in mtf_result['messages'][:-1]:  # All but the summary
                        conditions.append(f"MTF: {msg}")

            # Check time from last trade
            candles_since_last = len(data) - 1 - self._last_trade_index
            
            # In TestMode, use potentially relaxed spacing
            effective_min_spacing = self.min_candles_between
            if self.test_mode_enabled:
                from config.parameters import TEST_MODE_CONFIG
                effective_min_spacing = TEST_MODE_CONFIG.get('min_candles_between', effective_min_spacing)
            
            spacing_ok = candles_since_last >= effective_min_spacing

            if not spacing_ok:
                return False, [f"Insufficient spacing: {candles_since_last} vs {effective_min_spacing} min"]

            conditions.append(f"Trade spacing: {candles_since_last} candles OK")

            # Check if in consecutive loss pause
            if len(data) - 1 <= self._pause_until_index:
                return False, [f"Paused after {self._consecutive_losses} consecutive losses"]

            # Check signal safety using contradiction detector
            contradiction_score = 0.0
            contradiction_ok = True
            
            if not self.test_mode_enabled or not self.bypass_contradiction_detection:
                regime_info, conf, regime_details = self.regime_detector.detect_regime(data)
                safety_assessment = self.contradiction_detector.analyze_signal_safety(
                    data, position_type.value, regime_details
                )

                # Add contradiction information to conditions
                contradiction_score = safety_assessment['contradiction_summary'].get('contradiction_score', 0.0)
                conditions.append(f"Contradictions: {safety_assessment['risk_level']} (score: {contradiction_score:.2f})")

                # Check if we should filter the signal based on contradictions (skip in TestMode)
                should_filter = self.contradiction_detector.should_filter_signal(safety_assessment)
                contradiction_ok = not (should_filter and contradiction_score > 0.3 and not self.test_mode_enabled)
                
                if not contradiction_ok:
                    return False, [f"Signal filtered due to contradictions: {safety_assessment['recommendation']}"]
            else:
                # In TestMode with contradiction bypass, just add a note
                conditions.append("Contradictions: SKIPPED (TestMode)")

            return True, conditions

        except Exception as e:
            self.diagnostic_logger.error(f"Error checking entry conditions: {e}")
            return False, [f"Error in entry conditions: {e}"]

    strategy_class.check_entry_conditions = new_check_entry_conditions

    return strategy_class


# Apply the RSI fixes to the EnhancedRsiStrategyV5 class
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from strategies.enhanced_rsi_strategy_v5 import EnhancedRsiStrategyV5
apply_rsi_fix_to_strategy(EnhancedRsiStrategyV5)

print("✅ RSI Module Fix Applied")
print("  - Adaptive RSI thresholds based on market volatility")
print("  - More permissive momentum checks")
print("  - Enhanced entry condition logic")
print("  - Maintains TestMode flexibility")