"""
Diagnostic instrumented version of Enhanced RSI Strategy V5
to map signal flow and identify blocking points
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime

# Import the original classes
from strategies.enhanced_rsi_strategy_v5 import EnhancedRsiStrategyV5, PositionType
from strategies.mtf_analyzer import EnhancedMTFModule
from strategies.trend_filter import AdvancedTrendFilter
from strategies.market_regime_detector import MarketRegimeDetector
from strategies.risk_manager import DynamicRiskManager
from strategies.contradiction_detector import EnhancedContradictionSystem


class DiagnosticEnhancedRsiStrategyV5(EnhancedRsiStrategyV5):
    """
    Diagnostic version of Enhanced RSI Strategy V5 that instruments
    each decision module to track signal blocking
    """
    
    def __init__(self, **kwargs):
        # Initialize the original strategy
        super().__init__(**kwargs)
        
        # Diagnostic counters for each module
        self._diagnostic_counters = {
            'total_signals_checked': 0,
            'rsi_blocks': 0,
            'mtf_blocks': 0,
            'trend_blocks': 0,
            'momentum_blocks': 0,
            'regime_blocks': 0,
            'contradiction_blocks': 0,
            'risk_blocks': 0,
            'position_spacing_blocks': 0,
            'consecutive_loss_pause_blocks': 0,
            'total_entries_attempted': 0,
            'successful_entries': 0,
            'total_exits': 0,
        }
        
        # Track signal flow for each entry attempt
        self._signal_flow_log = []
        
        # Create diagnostic logger
        self.diagnostic_logger = logging.getLogger(f"{__name__}.diagnostic")
        
    def _log_signal_flow(self, iteration, step, condition_result, details=""):
        """Log the signal flow at each step"""
        flow_record = {
            'iteration': iteration,
            'step': step,
            'condition_result': condition_result,
            'details': details,
            'timestamp': datetime.now()
        }
        self._signal_flow_log.append(flow_record)

    def check_entry_conditions(self, data: pd.DataFrame, position_type: PositionType) -> Tuple[bool, List[str]]:
        """Enhanced entry conditions with diagnostic tracking"""
        self._diagnostic_counters['total_signals_checked'] += 1
        
        conditions = []
        diagnostic_details = {
            'rsi_ok': False,
            'mtf_ok': False,
            'trend_ok': False,
            'momentum_ok': False,
            'regime_ok': False,
            'contradiction_ok': False,
            'risk_ok': False
        }

        try:
            # Calculate or ensure RSI exists
            if 'RSI' not in data.columns:
                data = self._calculate_rsi(data)

            current_rsi = float(data['RSI'].iloc[-1]) if 'RSI' in data.columns else 50.0

            # BASIC RSI check
            if position_type == PositionType.LONG:
                rsi_condition = current_rsi <= (self.rsi_oversold + self.rsi_entry_buffer)
                rsi_ok = rsi_condition
                conditions.append(f"RSI: {current_rsi:.1f} ({'OK' if rsi_ok else 'FAIL'})")
            else:
                rsi_condition = current_rsi >= (self.rsi_overbought - self.rsi_entry_buffer)
                rsi_ok = rsi_condition
                conditions.append(f"RSI: {current_rsi:.1f} ({'OK' if rsi_ok else 'FAIL'})")

            if not rsi_ok:
                self._diagnostic_counters['rsi_blocks'] += 1
                self._log_signal_flow(len(data)-1, "RSI_CHECK", False, f"RSI {current_rsi:.2f} not in zone for {position_type.value}")
                return False, [f"RSI not in entry zone for {position_type.value} ({current_rsi:.1f})"]
            
            diagnostic_details['rsi_ok'] = True
            self._log_signal_flow(len(data)-1, "RSI_CHECK", True, f"RSI {current_rsi:.2f} OK")

            # MOMENTUM check
            momentum_ok = True
            if len(data) >= 3:
                # Check if price is moving in the expected direction
                if position_type == PositionType.LONG:
                    recent_move = (data['close'].iloc[-1] - data['close'].iloc[-3]) / data['close'].iloc[-3]
                    momentum_ok = recent_move > -0.005  # Allow slight pullback
                else:
                    recent_move = (data['close'].iloc[-3] - data['close'].iloc[-1]) / data['close'].iloc[-3]
                    momentum_ok = recent_move > -0.005

            if not momentum_ok:
                if not self.test_mode_enabled:  # Only block in normal mode, not test mode
                    self._diagnostic_counters['momentum_blocks'] += 1
                    self._log_signal_flow(len(data)-1, "MOMENTUM_CHECK", False, f"Momentum concern: {recent_move:.4f}")
                    return False, ["Momentum check: Concern over direction"]
                else:
                    # In TestMode, log but don't block
                    conditions.append("Momentum check: Concern over direction (TestMode - NOT blocking)")

            diagnostic_details['momentum_ok'] = momentum_ok
            if momentum_ok:
                self._log_signal_flow(len(data)-1, "MOMENTUM_CHECK", True, f"Momentum OK: {recent_move:.4f}")

            # Apply ENHANCED trend filter if enabled
            if self.trend_filter and self.enable_trend_filter:
                trend_ok, trend_desc, trend_conf, _ = self.trend_filter.evaluate_trend(data, position_type.value)
                
                # In TestMode, be more permissive
                if not self.test_mode_enabled and not trend_ok:
                    self._diagnostic_counters['trend_blocks'] += 1
                    self._log_signal_flow(len(data)-1, "TREND_CHECK", False, f"Trend issue: {trend_desc}")
                    return False, [f"Trend filter: {trend_desc}"]
                elif not trend_ok:
                    # In TestMode, log but don't necessarily block if other conditions are strong
                    conditions.append(f"Trend: {trend_desc} (TestMode - may allow)")
                    trend_ok = True  # Override in TestMode
                
                if trend_ok:
                    diagnostic_details['trend_ok'] = True
                    conditions.append(f"Trend: {trend_desc}")
                    self._log_signal_flow(len(data)-1, "TREND_CHECK", True, f"Trend OK: {trend_desc}")

            # Apply ENHANCED MTF analysis if enabled
            if self.mtf_analyzer and self.enable_mtf:
                mtf_result = self.mtf_analyzer.analyze_alignment(data, position_type.value)
                mtf_ok = mtf_result['is_aligned']
                mtf_desc = mtf_result['messages'][-1] if mtf_result['messages'] else "MTF analysis error"
                
                # In TestMode, be more permissive
                if not self.test_mode_enabled and not mtf_ok:
                    self._diagnostic_counters['mtf_blocks'] += 1
                    self._log_signal_flow(len(data)-1, "MTF_CHECK", False, f"MTF issue: {mtf_desc}")
                    return False, [f"MTF filter: {mtf_desc}"]
                elif not mtf_ok:
                    # In TestMode, log but don't necessarily block
                    conditions.append(f"MTF: {mtf_desc} (TestMode - may allow)")
                    mtf_ok = True  # Override in TestMode
                
                if mtf_ok:
                    diagnostic_details['mtf_ok'] = True
                    # Include all MTF messages for transparency
                    for msg in mtf_result['messages'][:-1]:  # All but the summary
                        conditions.append(f"MTF: {msg}")
                    self._log_signal_flow(len(data)-1, "MTF_CHECK", True, f"MTF OK: {mtf_desc}")

            # Check time from last trade (SPACING)
            candles_since_last = len(data) - 1 - self._last_trade_index
            spacing_ok = candles_since_last >= self.min_candles_between
            
            # In TestMode, use potentially relaxed spacing
            effective_min_spacing = self.min_candles_between
            if self.test_mode_enabled:
                from config.parameters import TEST_MODE_CONFIG
                effective_min_spacing = TEST_MODE_CONFIG.get('min_candles_between', effective_min_spacing)
            
            spacing_ok = candles_since_last >= effective_min_spacing

            if not spacing_ok:
                self._diagnostic_counters['position_spacing_blocks'] += 1
                self._log_signal_flow(len(data)-1, "SPACING_CHECK", False, f"Spacing {candles_since_last} < {effective_min_spacing}")
                return False, [f"Insufficient spacing: {candles_since_last} vs {effective_min_spacing} min"]

            conditions.append(f"Trade spacing: {candles_since_last} candles OK")
            self._log_signal_flow(len(data)-1, "SPACING_CHECK", True, f"Spacing OK: {candles_since_last} >= {effective_min_spacing}")

            # Check if in consecutive loss pause
            if len(data) - 1 <= self._pause_until_index:
                self._diagnostic_counters['consecutive_loss_pause_blocks'] += 1
                self._log_signal_flow(len(data)-1, "PAUSE_CHECK", False, f"On pause due to consecutive losses")
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
                    self._diagnostic_counters['contradiction_blocks'] += 1
                    self._log_signal_flow(len(data)-1, "CONTRADICTION_CHECK", False, f"Contradiction score: {contradiction_score:.2f}")
                    return False, [f"Signal filtered due to contradictions: {safety_assessment['recommendation']}"]
                else:
                    self._log_signal_flow(len(data)-1, "CONTRADICTION_CHECK", True, f"Contradiction OK: {contradiction_score:.2f}")
                    diagnostic_details['contradiction_ok'] = True
            else:
                # In TestMode with contradiction bypass, just add a note
                conditions.append("Contradictions: SKIPPED (TestMode)")
                self._log_signal_flow(len(data)-1, "CONTRADICTION_CHECK", True, "Contradiction check bypassed in TestMode")
                diagnostic_details['contradiction_ok'] = True

            return True, conditions

        except Exception as e:
            self.diagnostic_logger.error(f"Error checking entry conditions: {e}")
            return False, [f"Error in entry conditions: {e}"]

    def generate_signal(self, data: pd.DataFrame, current_index: int = 0) -> Dict[str, Any]:
        """Generate signal with comprehensive diagnostic tracking"""
        # Call the parent's signal processing but with added diagnostics
        signal = super().generate_signal(data, current_index)
        
        # Track entries attempted vs successful
        if signal['action'] in ['BUY', 'SELL']:
            self._diagnostic_counters['total_entries_attempted'] += 1
            if signal['action'] in ['BUY', 'SELL']:
                self._diagnostic_counters['successful_entries'] += 1
                
        elif signal['action'] in ['EXIT', 'PARTIAL_EXIT']:
            self._diagnostic_counters['total_exits'] += 1
            
        return signal

    def get_diagnostic_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report"""
        total_checked = self._diagnostic_counters['total_signals_checked']
        
        if total_checked > 0:
            blocking_percentages = {
                'rsi_blocks_pct': (self._diagnostic_counters['rsi_blocks'] / total_checked) * 100,
                'mtf_blocks_pct': (self._diagnostic_counters['mtf_blocks'] / total_checked) * 100,
                'trend_blocks_pct': (self._diagnostic_counters['trend_blocks'] / total_checked) * 100,
                'momentum_blocks_pct': (self._diagnostic_counters['momentum_blocks'] / total_checked) * 100,
                'regime_blocks_pct': (self._diagnostic_counters['regime_blocks'] / total_checked) * 100,
                'contradiction_blocks_pct': (self._diagnostic_counters['contradiction_blocks'] / total_checked) * 100,
                'risk_blocks_pct': (self._diagnostic_counters['risk_blocks'] / total_checked) * 100,
                'spacing_blocks_pct': (self._diagnostic_counters['position_spacing_blocks'] / total_checked) * 100,
                'pause_blocks_pct': (self._diagnostic_counters['consecutive_loss_pause_blocks'] / total_checked) * 100,
            }
        else:
            blocking_percentages = {}
        
        report = {
            'summary': {
                'total_signals_checked': self._diagnostic_counters['total_signals_checked'],
                'total_entries_attempted': self._diagnostic_counters['total_entries_attempted'],
                'successful_entries': self._diagnostic_counters['successful_entries'],
                'total_exits': self._diagnostic_counters['total_exits'],
            },
            'blocking_analysis': {
                'rsi_blocks': self._diagnostic_counters['rsi_blocks'],
                'mtf_blocks': self._diagnostic_counters['mtf_blocks'],
                'trend_blocks': self._diagnostic_counters['trend_blocks'],
                'momentum_blocks': self._diagnostic_counters['momentum_blocks'],
                'regime_blocks': self._diagnostic_counters['regime_blocks'],
                'contradiction_blocks': self._diagnostic_counters['contradiction_blocks'],
                'risk_blocks': self._diagnostic_counters['risk_blocks'],
                'position_spacing_blocks': self._diagnostic_counters['position_spacing_blocks'],
                'consecutive_loss_pause_blocks': self._diagnostic_counters['consecutive_loss_pause_blocks'],
            },
            'blocking_percentages': blocking_percentages,
            'signal_flow_log_count': len(self._signal_flow_log),
            'timestamp': datetime.now().isoformat()
        }
        
        return report

    def get_signal_flow_log(self) -> List[Dict[str, Any]]:
        """Return the detailed signal flow log"""
        return self._signal_flow_log.copy()