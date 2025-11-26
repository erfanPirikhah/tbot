"""
Enhanced strategy with comprehensive diagnostic capabilities
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from strategies.enhanced_rsi_strategy_v4 import EnhancedRsiStrategyV4, PositionType
from diagnostic.diagnostic_system import DiagnosticSystem
import logging

logger = logging.getLogger(__name__)

class DiagnosticEnhancedRsiStrategy(EnhancedRsiStrategyV4):
    """Enhanced RSI Strategy with comprehensive diagnostic logging"""
    
    def __init__(self, diagnostic_system: DiagnosticSystem = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostic_system = diagnostic_system or DiagnosticSystem()
        self._additional_metrics = {}
        
    def generate_signal_with_diagnostics(self, data: pd.DataFrame, current_index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate signal and return both signal and diagnostic data"""
        # Call parent method to get the signal
        signal = self.generate_signal(data, current_index)
        
        # Prepare diagnostic data
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
        
        # Add signal-specific diagnostic information
        diagnostic_data.update({
            'signal_action': signal.get('action', 'HOLD'),
            'signal_reason': signal.get('reason', ''),
            'entry_conditions': getattr(self, '_last_entry_conditions', []),
            'position_state': self._position.value,
            'current_portfolio_value': self._portfolio_value,
            'total_trades': self._total_trades,
            'consecutive_losses': self._consecutive_losses
        })
        
        return signal, diagnostic_data
    
    def check_entry_conditions(self, data: pd.DataFrame, position_type: PositionType) -> Tuple[bool, List[str]]:
        """Enhanced entry conditions with detailed diagnostic logging"""
        conditions = []
        try:
            # Store original behavior but capture more details
            result, cond_list = super().check_entry_conditions(data, position_type)
            
            # Store conditions for later diagnostic
            self._last_entry_conditions = cond_list
            conditions.extend(cond_list)
            
            # Additional diagnostic details
            if 'RSI' in data.columns:
                current_rsi = float(data['RSI'].iloc[-1])
                if position_type == PositionType.LONG:
                    conditions.append(f"RSI: {current_rsi:.2f} (threshold: {self.rsi_oversold + self.rsi_entry_buffer:.2f})")
                else:
                    conditions.append(f"RSI: {current_rsi:.2f} (threshold: {self.rsi_overbought - self.rsi_entry_buffer:.2f})")
            
            # Trend analysis
            if 'EMA_21' in data.columns and 'EMA_50' in data.columns:
                ema21 = float(data['EMA_21'].iloc[-1])
                ema50 = float(data['EMA_50'].iloc[-1])
                
                if position_type == PositionType.LONG:
                    trend_ok = ema21 >= ema50
                    conditions.append(f"Trend: {'OK' if trend_ok else 'FAIL'} (EMA21:{ema21:.4f} vs EMA50:{ema50:.4f})")
                else:
                    trend_ok = ema21 <= ema50
                    conditions.append(f"Trend: {'OK' if trend_ok else 'FAIL'} (EMA21:{ema21:.4f} vs EMA50:{ema50:.4f})")
            
            # Distance from last trade
            candles_since_last = len(data) - 1 - self._last_trade_index
            conditions.append(f"Gap: {candles_since_last}/{self.min_candles_between} candles")
            
            return result, conditions
            
        except Exception as e:
            logger.error(f"Error in enhanced entry conditions: {e}")
            return False, [f"Error: {str(e)}"]
    
    def check_exit_conditions(self, data: pd.DataFrame, current_index: int) -> Optional[Dict[str, Any]]:
        """Enhanced exit conditions with detailed diagnostic logging"""
        try:
            exit_signal = super().check_exit_conditions(data, current_index)
            
            if exit_signal and self._current_trade:
                # Add diagnostic info to exit signal
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
                    'take_profit': self._current_trade.take_profit
                }
            
            return exit_signal
            
        except Exception as e:
            logger.error(f"Error in enhanced exit conditions: {e}")
            return None