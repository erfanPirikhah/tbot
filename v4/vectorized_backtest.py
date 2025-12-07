
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class VectorizedBacktest:
    """
    High-performance vectorized backtesting engine for Enhanced RSI Strategy.
    Calculates signals and PnL using pandas/numpy vector operations instead of iteration.
    """
    
    def __init__(self, data: pd.DataFrame, params: Dict[str, Any]):
        self.data = data.copy()
        self.params = params
        
    def run(self):
        """Run the vectorized backtest"""
        df = self.data
        
        # 1. Calculate Indicators (Vectorized)
        self._calculate_indicators(df)
        
        # 2. Calculate Adaptive Thresholds (Vectorized)
        self._calculate_adaptive_thresholds(df)
        
        # 3. Generate Signals (Vectorized)
        self._generate_signals(df)
        
        # 4. Simulate Trades (Vectorized-ish / Event Loop optimized)
        # Fully vectorized trade management is hard with complex exits (trailing, partial),
        # but we can do a very fast simplified version or a Numba-optimized loop.
        # For 'Analysis' report purposes, a simplified vectorized approach 
        # that assumes standard exits is usually sufficient for hyperparameter tuning.
        results = self._simulate_trades_vectorized_simplified(df)
        
        return results

    def _calculate_indicators(self, df):
        """Calculate RSI, ATR, EMAs vectorized"""
        close = df['close']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR
        high = df['high']
        low = df['low']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Volatility (Std Dev)
        df['Volatility'] = close.pct_change().rolling(20).std()
        
    def _calculate_adaptive_thresholds(self, df):
        """Calculate adaptive thresholds allowing for vectorized optimization"""
        # Base thresholds
        rsi_oversold = self.params.get('rsi_oversold', 30)
        rsi_overbought = self.params.get('rsi_overbought', 70)
        
        # Volatility adjustment (Vectorized)
        # Normalize volatility: (Current - Mean) / Std ? Or just raw ratio vs baseline
        # Using similar logic to strategy but vectorized
        baseline_vol = 0.01
        vol_factor = (df['Volatility'] / baseline_vol).clip(0.5, 2.0)
        
        # Adaptive Bands
        # High Vol -> Widen bands
        df['RSI_Lower'] = rsi_oversold - (5 * (vol_factor - 1.0))
        df['RSI_Upper'] = rsi_overbought + (5 * (vol_factor - 1.0))
        
        # Clamp
        df['RSI_Lower'] = df['RSI_Lower'].clip(15, 40)
        df['RSI_Upper'] = df['RSI_Upper'].clip(60, 85)

    def _generate_signals(self, df):
        """Generate Entry Signals"""
        # Long Entry: RSI < Adaptive Lower
        df['Signal_Long'] = (df['RSI'] < df['RSI_Lower']) 
        
        # Short Entry: RSI > Adaptive Upper
        df['Signal_Short'] = (df['RSI'] > df['RSI_Upper'])
        
        # Filter: Avoid consecutive signals (simple approach: shift)
        # For rigorous backtest, we need state. But for pure signal analysis:
        df['Entry'] = 0
        df.loc[df['Signal_Long'], 'Entry'] = 1
        df.loc[df['Signal_Short'], 'Entry'] = -1
        
    def _simulate_trades_vectorized_simplified(self, df):
        """
        Fast simulation of trades assuming fixed TP/SL for speed.
        Useful for finding good RSI params.
        """
        entries = df[df['Entry'] != 0].copy()
        
        trades = []
        if len(entries) == 0:
            return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0}
            
        atr_mult = self.params.get('sl_atr_multiplier', 2.0)
        risk_reward = self.params.get('risk_reward', 2.0)
        
        # Iterate only through entries (much faster than iterating all candles)
        # This is "Event-Based" on filtered signals
        for time, row in entries.iterrows():
            entry_price = row['close']
            atr = row['ATR']
            direction = row['Entry'] # 1 or -1
            
            if pd.isna(atr): continue
            
            sl_dist = atr * atr_mult
            tp_dist = sl_dist * risk_reward # Simplified RR
            
            if direction == 1: # Long
                sl = entry_price - sl_dist
                tp = entry_price + tp_dist
            else: # Short
                sl = entry_price + sl_dist
                tp = entry_price - tp_dist
                
            # Find exit - Look forward from entry time
            # We treat the rest of the dataframe as a numpy array for speed
            future_data = df.loc[time:].iloc[1:] 
            if len(future_data) == 0: continue
            
            # Vectorized search for exit
            # Long Exit: Low < SL or High > TP
            # Short Exit: High > SL or Low < TP
            
            if direction == 1:
                # Find first index where Low < SL OR High > TP
                # (We can use searchsorted or argmax)
                hit_sl = future_data['low'] < sl
                hit_tp = future_data['high'] > tp
            else:
                hit_sl = future_data['high'] > sl
                hit_tp = future_data['low'] < tp
                
            # Get earliest occurrence
            sl_idx = hit_sl.idxmax() if hit_sl.any() else None
            tp_idx = hit_tp.idxmax() if hit_tp.any() else None
            
            exit_type = None
            pnl_pct = 0
            
            if sl_idx and tp_idx:
                if sl_idx < tp_idx:
                    exit_type = 'SL'
                    exit_price = sl
                else:
                    exit_type = 'TP'
                    exit_price = tp
            elif sl_idx:
                exit_type = 'SL'
                exit_price = sl
            elif tp_idx:
                exit_type = 'TP'
                exit_price = tp
            else:
                exit_type = 'END'
                exit_price = future_data['close'].iloc[-1]
                
            if direction == 1:
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price
                
            trades.append({
                'entry_time': time,
                'type': 'LONG' if direction == 1 else 'SHORT',
                'pnl': pnl_pct,
                'exit_type': exit_type
            })
            
        # Compile results
        if not trades:
            return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0}
            
        trade_df = pd.DataFrame(trades)
        total_pnl = trade_df['pnl'].sum()
        win_rate = len(trade_df[trade_df['pnl'] > 0]) / len(trade_df)
        
        return {
            'total_trades': len(trade_df),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': trade_df['pnl'].mean()
        }

if __name__ == "__main__":
    # Test stub
    print("Vectorized Backtest Module Loaded")
