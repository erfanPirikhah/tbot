"""
Enhanced strategy with comprehensive diagnostic capabilities - V5
Updated to work with the enhanced V5 strategy with all improvements
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from strategies.enhanced_rsi_strategy_v5 import EnhancedRsiStrategyV5, PositionType
from diagnostic.diagnostic_system import DiagnosticSystem
import logging

logger = logging.getLogger(__name__)

class DiagnosticEnhancedRsiStrategy(EnhancedRsiStrategyV5):  # Updated to inherit from V5
    """Enhanced RSI Strategy V5 with comprehensive diagnostic logging"""

    def __init__(self, diagnostic_system: DiagnosticSystem = None, *args, **kwargs):
        super().__init__(*args, **kwargs)  # This now calls V5's __init__ which includes all enhancements
        self.diagnostic_system = diagnostic_system or DiagnosticSystem()
        self._additional_metrics = {}

    def generate_signal_with_diagnostics(self, data: pd.DataFrame, current_index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate signal and return both signal and diagnostic data - V5 enhanced"""
        # Call parent method to get the signal (now uses V5 logic)
        signal = self.generate_signal(data, current_index)

        # Prepare enhanced diagnostic data
        diagnostic_data = {}

        # Capture current market conditions and indicators
        if len(data) > 0:
            current_row = data.iloc[-1]
            diagnostic_data = {
                'current_price': float(current_row['close']),
                'rsi': float(current_row['RSI']) if 'RSI' in current_row else 50,
                'atr': float(current_row['ATR']) if 'ATR' in current_row else 0,
                'bb_upper': float(current_row['BB_Upper']) if 'BB_Upper' in current_row else 0,
                'bb_lower': float(current_row['BB_Lower']) if 'BB_Lower' in current_row else 0,
                'ema_fast': float(current_row['EMA_21']) if 'EMA_21' in current_row else 0,
                'ema_slow': float(current_row['EMA_50']) if 'EMA_50' in current_row else 0,
                'adx': float(current_row['ADX']) if 'ADX' in current_row else 0,
                'macd': float(current_row['MACD']) if 'MACD' in current_row else 0,
            }

        # Add enhanced signal-specific diagnostic information
        diagnostic_data.update({
            'signal_action': signal.get('action', 'HOLD'),
            'signal_reason': signal.get('reason', ''),
            'entry_conditions': getattr(self, '_last_entry_conditions', []),
            'position_state': self._position.value,
            'current_portfolio_value': self._portfolio_value,
            'total_trades': self._total_trades,
            'consecutive_losses': self._consecutive_losses,
            'current_regime': self._current_regime,
            'contradiction_score': self._contradiction_report.get('contradiction_score', 0.0) if hasattr(self, '_contradiction_report') else 0.0,
            'enhanced_indicators_available': True
        })

        return signal, diagnostic_data

    def check_entry_conditions(self, data: pd.DataFrame, position_type: PositionType) -> Tuple[bool, List[str]]:
        """Enhanced entry conditions with detailed diagnostic logging - V5 with all improvements"""
        # This now uses the enhanced V5 implementation which includes:
        # - Advanced trend filter
        # - Enhanced MTF analysis
        # - Contradiction detection
        # - Regime-based adjustments
        # So we can just call the parent method and add diagnostic info
        result, conditions = super().check_entry_conditions(data, position_type)

        # Add additional diagnostic details specific to V5 enhancements
        try:
            if 'RSI' in data.columns:
                current_rsi = float(data['RSI'].iloc[-1])
                if position_type == PositionType.LONG:
                    conditions.append(f"RSI: {current_rsi:.2f} (threshold: {self.rsi_oversold + self.rsi_entry_buffer:.2f})")
                else:
                    conditions.append(f"RSI: {current_rsi:.2f} (threshold: {self.rsi_overbought - self.rsi_entry_buffer:.2f})")

            # Enhanced trend analysis using the new trend filter
            if self.trend_filter:
                try:
                    trend_ok, trend_desc, trend_conf = self.trend_filter.evaluate(data, position_type.value)
                    conditions.append(f"Trend Filter: {trend_desc} (conf: {trend_conf:.2f})")
                except Exception:
                    conditions.append("Trend Filter: Error evaluating trend")

            # Enhanced MTF analysis
            if self.mtf_analyzer:
                try:
                    mtf_result = self.mtf_analyzer.analyze_alignment(data, position_type.value)
                    conditions.append(f"MTF Analysis: {mtf_result['messages'][-1] if mtf_result['messages'] else 'Error'}")
                except Exception:
                    conditions.append("MTF Analysis: Error evaluating MTF")

            # Contradiction information
            contradiction_score = getattr(self, '_contradiction_report', {}).get('contradiction_score', 0.0)
            conditions.append(f"Contradictions: {contradiction_score:.3f}")

            # Regime information
            conditions.append(f"Market Regime: {self._current_regime}")

            return result, conditions

        except Exception as e:
            logger.error(f"Error in enhanced entry conditions V5: {e}")
            return False, [f"Error: {str(e)}"]

    def check_exit_conditions(self, data: pd.DataFrame, current_index: int) -> Optional[Dict[str, Any]]:
        """Enhanced exit conditions with detailed diagnostic logging - V5"""
        try:
            exit_signal = super().check_exit_conditions(data, current_index)

            if exit_signal and self._current_trade:
                # Add enhanced diagnostic info to exit signal using V5 trade structure
                current_price = data['close'].iloc[-1]
                entry_price = self._current_trade.entry_price
                quantity = self._current_trade.quantity

                # Calculate additional metrics
                if self._position == PositionType.LONG:
                    pnl_amount = (current_price - entry_price) * quantity
                else:
                    pnl_amount = (entry_price - current_price) * quantity

                pnl_percentage = (pnl_amount / (entry_price * quantity)) * 100 if (entry_price * quantity) != 0 else 0

                exit_signal['diagnostic_info'] = {
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'quantity': quantity,
                    'pnl_amount': pnl_amount,
                    'pnl_percentage': pnl_percentage,
                    'trade_duration': current_index - self._last_trade_index,
                    'highest_profit': getattr(self._current_trade, 'highest_profit', 0),
                    'stop_loss': self._current_trade.stop_loss,
                    'take_profit': self._current_trade.take_profit,
                    'entry_regime': getattr(self._current_trade, 'entry_regime', 'UNKNOWN'),
                    'contradiction_score_at_entry': getattr(self._current_trade, 'contradiction_score', 0.0)
                }

            return exit_signal

        except Exception as e:
            logger.error(f"Error in enhanced exit conditions V5: {e}")
            return None