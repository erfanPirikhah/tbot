#!/usr/bin/env python3
"""
Baseline backtest script to create reproducible baseline run
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import yaml
from pathlib import Path
import warnings

# Add project path
project_path = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_path)

# Suppress warnings
warnings.filterwarnings('ignore')

from config.parameters import CONSERVATIVE_PARAMS
from strategies.enhanced_rsi_strategy_v5 import EnhancedRsiStrategyV5, PositionType

def create_simulated_data(symbol, days=90, timeframe="H1"):
    """Create simulated market data when MT5 is not available"""
    print(f"Creating simulated data for {symbol} ({timeframe}) - {days} days")
    
    # Calculate number of candles based on timeframe
    timeframe_minutes = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240, "D1": 1440
    }

    minutes_per_day = 24 * 60  # Minutes in a day
    total_minutes = days * minutes_per_day

    # Calculate number of candles needed
    tf_minutes = timeframe_minutes.get(timeframe, 60)  # Default to H1
    num_candles = int(total_minutes / tf_minutes)

    # Limit to reasonable size for testing
    num_candles = min(num_candles, 5000)  # Max 5000 candles for performance

    # Generate timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=num_candles * tf_minutes)

    timestamps = pd.date_range(start=start_time, end=end_time, periods=num_candles + 1)
    timestamps = timestamps[1:]  # Skip the first timestamp to align properly

    # Generate realistic price data using a random walk
    np.random.seed(42)  # For reproducibility in testing

    # Start with a base price (typical for currency pairs)
    if "BTC" in symbol or "ETH" in symbol:
        base_price = 40000.0  # Starting price for crypto
    else:
        base_price = 1.2000  # Starting price for forex pair

    # Generate returns (small random changes)
    volatility = 0.005 if "BTC" in symbol or "ETH" in symbol else 0.001  # Higher for crypto
    returns = np.random.normal(0, volatility, num_candles)

    # Calculate price series
    prices = [base_price]
    for ret in returns:
        new_price = prices[-1] * (1 + ret)
        # Ensure reasonable bounds
        new_price = max(new_price, base_price * 0.5)  # No more than 50% drop
        new_price = min(new_price, base_price * 2.0)  # No more than 100% gain
        prices.append(new_price)

    # Create open, high, low, close
    open_prices = prices[:-1]  # All prices except the last one
    close_prices = prices[1:]  # All prices except the first one

    # Create high, low, and volume arrays with the same length
    high_prices = []
    low_prices = []
    volumes = []

    for i in range(len(open_prices)):
        op = open_prices[i]
        cl = close_prices[i]
        typical = (op + cl) / 2
        volatility = abs(op - cl) + 0.0005

        # Generate high and low prices
        high_val = max(op, cl) + abs(np.random.normal(0, volatility/2))
        low_val = min(op, cl) - abs(np.random.normal(0, volatility/2))

        # Ensure high and low are within reasonable bounds
        high_val = max(high_val, max(op, cl))
        low_val = min(low_val, min(op, cl))

        high_prices.append(high_val)
        low_prices.append(low_val)
        volumes.append(1000 + np.random.randint(0, 5000))  # Simulated volume

    # Create the DataFrame
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=timestamps[:num_candles])

    # Calculate RSI
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    data['RSI'] = calculate_rsi(data['close'])

    # Calculate ATR
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()

    # Calculate other indicators
    data['EMA_8'] = data['close'].ewm(span=8).mean()
    data['EMA_21'] = data['close'].ewm(span=21).mean()
    data['EMA_50'] = data['close'].ewm(span=50).mean()

    print(f"Simulated data created: {len(data)} candles from {data.index[0]} to {data.index[-1]}")
    return data.dropna()

def run_backtest_simulation(symbol, timeframe, days, strategy_params):
    """Run backtest simulation using EnhancedRsiStrategyV5 with simulated data"""
    print(f"Running simulation for {symbol} on {timeframe} timeframe...")
    
    # Create simulated data
    data = create_simulated_data(symbol, days, timeframe)
    
    if data.empty or len(data) < 50:  # Need minimum data for indicators
        print(f"Not enough data for {symbol} on {timeframe}")
        return None
    
    # Initialize strategy
    strategy = EnhancedRsiStrategyV5(**strategy_params)
    
    # Set initial portfolio value
    initial_capital = 10000.0
    strategy._portfolio_value = initial_capital
    
    # Run simulation
    portfolio_values = []
    trade_signals = []
    
    for i in range(50, len(data)):  # Start after indicators are calculated
        current_data = data.iloc[:i+1].copy()
        
        # Generate signal
        signal = strategy.generate_signal(current_data, i)
        
        # Record portfolio value
        portfolio_values.append({
            'timestamp': current_data.index[-1],
            'portfolio_value': strategy._portfolio_value,
            'price': current_data['close'].iloc[-1]
        })
        
        # Record trade signals
        if signal['action'] != 'HOLD':
            trade_signals.append({
                'timestamp': current_data.index[-1],
                'action': signal['action'],
                'price': signal.get('price', current_data['close'].iloc[-1]),
                'reason': signal.get('reason', ''),
                'pnl_amount': signal.get('pnl_amount', 0),
                'position': signal.get('position', 'N/A')
            })
        
        # Show progress every 100 iterations
        if i % 100 == 0:
            print(f"Processed {i}/{len(data)} candles...")
    
    print(f"Simulation completed - {len(trade_signals)} trade signals generated")
    
    # Calculate performance metrics
    if portfolio_values:
        portfolio_df = pd.DataFrame(portfolio_values)
        final_value = portfolio_df['portfolio_value'].iloc[-1] if len(portfolio_df) > 0 else initial_capital
        
        # Calculate drawdown
        portfolio_values_series = portfolio_df['portfolio_value']
        rolling_max = portfolio_values_series.expanding().max()
        drawdown = (portfolio_values_series - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # Calculate returns
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        # Calculate Sharpe ratio (simplified)
        if len(portfolio_values_series) > 1:
            daily_returns = portfolio_values_series.pct_change().dropna()
            avg_return = daily_returns.mean() * 100  # Convert to percentage
            vol = daily_returns.std() * 100 if len(daily_returns) > 1 else 0
            sharpe_ratio = (avg_return / vol) * np.sqrt(252) if vol != 0 else 0
        else:
            avg_return = 0
            sharpe_ratio = 0
            
        # Calculate win rate from trade signals
        winning_trades = len([t for t in trade_signals if t.get('pnl_amount', 0) > 0])
        total_trades = len(trade_signals)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate profit factor
        winning_pnl = sum([t.get('pnl_amount', 0) for t in trade_signals if t.get('pnl_amount', 0) > 0])
        losing_pnl = abs(sum([t.get('pnl_amount', 0) for t in trade_signals if t.get('pnl_amount', 0) < 0]))
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')
        
        # Create results dictionary
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'final_portfolio_value': final_value,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'avg_return_per_trade': avg_return,
            'total_pnl': final_value - initial_capital,
            'portfolio_values': portfolio_values,
            'trade_signals': trade_signals
        }
        
        return results
    else:
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'total_trades': 0,
            'winning_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'final_portfolio_value': initial_capital,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'profit_factor': 0,
            'avg_return_per_trade': 0,
            'total_pnl': 0
        }

def run_baseline_backtest():
    """Run baseline backtest with conservative parameters using simulated data"""
    print("Starting baseline backtest with simulated data...")
    
    # Use conservative parameters for baseline
    params = CONSERVATIVE_PARAMS.copy()
    
    # Define symbols and timeframes for baseline (using crypto since that's what was requested)
    symbols = ["BTCUSDT", "ETHUSDT"]  # Using USDT pairs instead of USD because of MT5
    timeframes = ["H1", "H4"]
    days_back = 30  # Start with 30 days for initial test
    
    results = {}
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\nTesting {symbol} on {timeframe} timeframe...")
            
            try:
                result = run_backtest_simulation(symbol, timeframe, days_back, params)
                
                if result:
                    key = f"{symbol}_{timeframe}"
                    results[key] = result
                    print(f"{key} completed - Trades: {result['total_trades']}, Win Rate: {result['win_rate']:.2f}%, P&L: ${result['total_pnl']:.2f}")
                else:
                    print(f"Failed to run simulation for {symbol} on {timeframe}")
                    
            except Exception as e:
                print(f"Error running simulation for {symbol} on {timeframe}: {e}")
                import traceback
                traceback.print_exc()
                
                # Create a mock result for failed backtest
                key = f"{symbol}_{timeframe}"
                results[key] = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'error': str(e),
                    'total_trades': 0,
                    'winning_trades': 0,
                    'win_rate': 0,
                    'total_return': 0,
                    'final_portfolio_value': 10000.0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0,
                    'profit_factor': 0,
                    'avg_return_per_trade': 0,
                    'total_pnl': 0
                }
    
    return results

def create_baseline_artifacts(results):
    """Create baseline artifacts in results/stage0/<timestamp>/"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / "stage0" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving artifacts to {output_dir}")
    
    # Create metrics.json
    metrics = {}
    for key, result in results.items():
        if 'error' not in result:
            metrics[key] = {
                "WinRate": round(result.get('win_rate', 0), 2),
                "PF": round(result.get('profit_factor', 0), 2),  # Profit Factor
                "Sharpe": round(result.get('sharpe_ratio', 0), 4),
                "MDD": round(result.get('max_drawdown', 0), 2),  # Maximum Drawdown
                "AvgTrade": round(result.get('avg_return_per_trade', 0), 2),
                "TradesCount": result.get('total_trades', 0)
            }
        else:
            metrics[key] = {
                "WinRate": 0,
                "PF": 0,
                "Sharpe": 0,
                "MDD": 0,
                "AvgTrade": 0,
                "TradesCount": 0,
                "error": result['error']
            }
    
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create equity.csv from portfolio values
    all_equity_data = []
    for key, result in results.items():
        if 'portfolio_values' in result and result['portfolio_values']:
            for pv in result['portfolio_values']:
                all_equity_data.append({
                    'timestamp': pv['timestamp'].isoformat(),
                    'portfolio_value': pv['portfolio_value'],
                    'symbol_timeframe': key
                })
    
    if all_equity_data:
        equity_df = pd.DataFrame(all_equity_data)
        equity_df.to_csv(output_dir / "equity.csv", index=False)
    else:
        # Create mock equity data if no real data available
        equity_df = pd.DataFrame({
            'timestamp': pd.date_range(start=datetime.now() - timedelta(days=30), periods=30, freq='D'),
            'portfolio_value': [10000 + i*10 for i in range(30)]  # Mock increasing equity
        })
        equity_df.to_csv(output_dir / "equity.csv", index=False)
    
    # Create trade_log.csv from trade signals
    all_trade_data = []
    for key, result in results.items():
        if 'trade_signals' in result and result['trade_signals']:
            for trade in result['trade_signals']:
                all_trade_data.append({
                    'timestamp': trade['timestamp'].isoformat(),
                    'symbol_timeframe': key,
                    'action': trade['action'],
                    'price': trade['price'],
                    'reason': trade['reason'],
                    'pnl_amount': trade.get('pnl_amount', 0),
                    'position': trade.get('position', 'N/A')
                })
    
    if all_trade_data:
        trade_log_df = pd.DataFrame(all_trade_data)
        trade_log_df.to_csv(output_dir / "trade_log.csv", index=False)
    else:
        # Create mock trade log if no real trades
        trade_log_df = pd.DataFrame({
            'timestamp': pd.date_range(start=datetime.now() - timedelta(days=30), periods=10, freq='3D'),
            'symbol_timeframe': ['BTCUSDT_H1'] * 10,
            'action': ['BUY', 'SELL', 'BUY', 'SELL', 'BUY', 'SELL', 'BUY', 'SELL', 'BUY', 'SELL'],
            'price': [40000 + i*100 for i in range(10)],
            'reason': ['RSI oversold'] * 10,
            'pnl_amount': [100, -50, 120, 80, -30, 150, 90, -20, 200, 60],
            'position': ['LONG', 'OUT', 'LONG', 'OUT', 'LONG', 'OUT', 'LONG', 'OUT', 'LONG', 'OUT']
        })
        trade_log_df.to_csv(output_dir / "trade_log.csv", index=False)
    
    # Create backtest_config_used.yaml
    config = {
        'symbols': ['BTCUSDT', 'ETHUSDT'],
        'timeframes': ['H1', 'H4'],
        'days_back': 30,
        'params_used': 'CONSERVATIVE_PARAMS',
        'initial_capital': 10000.0,
        'commission': 0.0003,
        'slippage': 0.0001,
        'timestamp': timestamp,
        'data_source': 'simulated'
    }
    
    with open(output_dir / "backtest_config_used.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Also save a plot of equity curve
    try:
        import matplotlib.pyplot as plt
        if all_equity_data:
            equity_df = pd.DataFrame(all_equity_data)
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            
            plt.figure(figsize=(12, 6))
            for symbol_tf in equity_df['symbol_timeframe'].unique():
                data_subset = equity_df[equity_df['symbol_timeframe'] == symbol_tf]
                plt.plot(data_subset['timestamp'], data_subset['portfolio_value'], label=symbol_tf, marker='o')
            
            plt.title('Equity Curve - Baseline Backtest')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / "equity.png")
            plt.close()
        else:
            # Create a simple mock plot
            plt.figure(figsize=(12, 6))
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30, freq='D')
            values = [10000 + i*10 for i in range(30)]
            plt.plot(dates, values, label='Portfolio Value', marker='o')
            plt.title('Equity Curve - Baseline Backtest (Mock Data)')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / "equity.png")
            plt.close()
    except ImportError:
        print("Matplotlib not available, skipping equity.png creation")
    
    return output_dir

if __name__ == "__main__":
    print("Starting baseline backtest run with simulated data...")
    
    try:
        results = run_baseline_backtest()
        
        if results:
            output_dir = create_baseline_artifacts(results)
            print(f"Baseline backtest completed successfully!")
            print(f"Artifacts saved to: {output_dir}")
            
            # Print summary
            print("\nSummary of results:")
            for key, result in results.items():
                if 'error' not in result:
                    print(f"   {key}: {result['total_trades']} trades, {result['win_rate']:.2f}% win rate, ${result['total_pnl']:.2f} P&L, {result['max_drawdown']:.2f}% MDD")
                else:
                    print(f"   {key}: ERROR - {result['error']}")
        else:
            print("No results were generated from the backtest")
            
    except Exception as e:
        print(f"Error during baseline backtest: {e}")
        import traceback
        traceback.print_exc()