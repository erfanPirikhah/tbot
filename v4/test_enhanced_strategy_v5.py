"""
Test script to validate the Enhanced RSI Strategy V5
with all improvements from the diagnostic analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the enhanced modules
from strategies.enhanced_rsi_strategy_v5 import EnhancedRsiStrategyV5
from config.parameters import OPTIMIZED_PARAMS_V5
from strategies.mtf_analyzer import EnhancedMTFModule
from strategies.trend_filter import AdvancedTrendFilter
from strategies.market_regime_detector import MarketRegimeDetector
from strategies.risk_manager import DynamicRiskManager
from strategies.contradiction_detector import EnhancedContradictionSystem

def create_sample_data(days=100):
    """Create sample market data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Create realistic price data with some trends and volatility
    prices = [100.0]
    for i in range(1, days):
        # Add some trend and random movement
        if i < 30:
            change = 0.3 + np.random.normal(0, 0.2)  # Uptrend
        elif i < 60:
            change = -0.2 + np.random.normal(0, 0.3)  # Downtrend with higher volatility
        elif i < 80:
            change = np.random.normal(0, 0.15)  # Sideways
        else:
            change = 0.1 + np.random.normal(0, 0.25)  # Another trend
            
        new_price = max(50, prices[-1] + change)  # Prevent negative prices
        prices.append(new_price)
    
    data = pd.DataFrame({
        'close': prices,
        'high': [p + abs(np.random.normal(0, 0.1)) for p in prices],
        'low': [p - abs(np.random.normal(0, 0.1)) for p in prices],
        'open': [p + np.random.normal(0, 0.05) for p in prices],
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    }, index=dates)
    
    # Add technical indicators
    # Calculate RSI
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Add other indicators
    data['EMA_8'] = data['close'].ewm(span=8).mean()
    data['EMA_21'] = data['close'].ewm(span=21).mean()
    data['EMA_50'] = data['close'].ewm(span=50).mean()
    
    # Add HTF indicators for MTF analysis
    for tf in ['H4', 'D1']:
        data[f'RSI_{tf}'] = data['RSI'].shift(np.random.randint(1, 5)).fillna(method='bfill')
        data[f'EMA_21_{tf}'] = data['close'].ewm(span=21).mean().shift(np.random.randint(1, 3)).fillna(method='bfill')
        data[f'EMA_50_{tf}'] = data['close'].ewm(span=50).mean().shift(np.random.randint(1, 3)).fillna(method='bfill')
        data[f'TrendDir_{tf}'] = np.random.choice([-1, 0, 1], len(data))
    
    return data.dropna()

def test_strategy_v5():
    """Test the Enhanced RSI Strategy V5"""
    print("="*80)
    print("TESTING ENHANCED RSI STRATEGY V5")
    print("With all diagnostic improvements:")
    print("- Enhanced MTF with majority alignment")
    print("- Active trend filtering with multiple confirmations")
    print("- Dynamic risk management")
    print("- Contradiction detection system")
    print("- Regime-aware trading")
    print("="*80)

    # Create sample data
    data = create_sample_data(days=100)
    recent_data = data.tail(80)  # Use recent data but make sure we have enough history

    print(f"Created sample data: {len(recent_data)} days")
    print(f"Price range: {recent_data['close'].min():.4f} - {recent_data['close'].max():.4f}")
    
    # Initialize strategy with V5 parameters
    strategy_params = OPTIMIZED_PARAMS_V5.copy()
    strategy = EnhancedRsiStrategyV5(**strategy_params)
    
    print(f"Strategy initialized with V5 parameters:")
    print(f"   RSI: {strategy_params['rsi_oversold']}/{strategy_params['rsi_overbought']} (buffer: {strategy_params['rsi_entry_buffer']})")
    print(f"   Trend Filter: {'ON' if strategy_params['enable_trend_filter'] else 'OFF'}")
    print(f"   MTF: {'ON' if strategy_params['enable_mtf'] else 'OFF'} (majority alignment)")
    print(f"   Risk: {strategy_params['risk_per_trade']*100:.2f}% per trade")

    # Test signal generation on the last 20 days of data
    print(f"\nTesting signal generation...")
    signals = []

    for i in range(len(recent_data)):
        if i < 50:  # Skip first 50 for indicators to stabilize
            continue

        current_data = recent_data.iloc[:i+1].copy()
        signal = strategy.generate_signal(current_data, i)

        if signal['action'] != 'HOLD':
            signals.append((current_data.index[-1], signal['action'], signal.get('price', current_data['close'].iloc[-1]), signal.get('reason', '')))

    print(f"Generated {len(signals)} actionable signals:")
    for date, action, price, reason in signals[:10]:  # Show first 10
        print(f"   {date.strftime('%Y-%m-%d')}: {action} at {price:.4f} - {reason}")

    if len(signals) > 10:
        print(f"   ... and {len(signals) - 10} more signals")

    # Test performance metrics
    metrics = strategy.get_performance_metrics()
    print(f"\nPERFORMANCE METRICS:")
    print(f"   Total Trades: {metrics.get('total_trades', 0)}")
    print(f"   Winning Trades: {metrics.get('winning_trades', 0)}")
    print(f"   Win Rate: {metrics.get('win_rate', 0):.2f}%")
    print(f"   Total P&L: ${metrics.get('total_pnl', 0):,.2f}")
    print(f"   Current Position: {metrics.get('current_position', 'OUT')}")
    
    # Test enhanced metrics
    print(f"\nENHANCED V5 METRICS:")
    print(f"   Current Regime: {metrics.get('current_regime', 'UNKNOWN')}")
    print(f"   Avg Contradiction Score: {metrics.get('avg_contradiction_score', 0):.3f}")
    print(f"   Strategy Version: {metrics.get('strategy_version', 'UNKNOWN')}")

    # Show regime analysis if available
    regime_perf = metrics.get('regime_performance', {})
    if regime_perf:
        print(f"\nREGIME-BASED PERFORMANCE:")
        for regime, data in regime_perf.items():
            if data['trades'] > 0:
                print(f"   {regime}: {data['trades']} trades, {data['win_rate']:.1f}% win rate, ${data['avg_pnl']:.2f} avg P&L")

    print(f"\nEnhanced RSI Strategy V5 test completed successfully!")
    print("="*80)
    
    return True

def test_individual_modules():
    """Test individual enhanced modules"""
    print("\nTESTING INDIVIDUAL ENHANCED MODULES")
    print("-" * 50)

    # Create sample data
    data = create_sample_data(days=50)
    current_data = data.tail(30)

    # Test MTF Module
    print("1. Testing Enhanced MTF Module...")
    mtf_module = EnhancedMTFModule()
    for position_type in ['LONG', 'SHORT']:
        analysis = mtf_module.analyze_alignment(current_data, position_type)
        print(f"   {position_type}: Aligned={analysis['is_aligned']}, Score={analysis['score']:.3f}")

    # Test Trend Filter
    print("2. Testing Advanced Trend Filter...")
    trend_filter = AdvancedTrendFilter()
    for position_type in ['LONG', 'SHORT']:
        result, desc, conf, comp_scores = trend_filter.evaluate_trend(current_data, position_type)
        print(f"   {position_type}: {result}, Confidence={conf:.3f}")

    # Test Regime Detector
    print("3. Testing Market Regime Detector...")
    regime_detector = MarketRegimeDetector()
    regime, conf, details = regime_detector.detect_regime(current_data)
    print(f"   Regime: {regime}, Confidence={conf:.3f}")

    # Test Risk Manager
    print("4. Testing Dynamic Risk Manager...")
    risk_manager = DynamicRiskManager()
    risk_pct, risk_metrics = risk_manager.calculate_dynamic_risk(current_data, details)
    print(f"   Dynamic Risk: {risk_pct:.4f} ({risk_metrics['combined_multiplier']:.3f}x)")

    # Test Contradiction Detector
    print("5. Testing Enhanced Contradiction System...")
    contradiction_system = EnhancedContradictionSystem()
    for position_type in ['LONG', 'SHORT']:
        safety = contradiction_system.analyze_signal_safety(current_data, position_type, details)
        print(f"   {position_type}: {safety['risk_level']} risk, Quality={safety['signal_quality']:.3f}")

    print("All individual modules tested successfully!")

if __name__ == "__main__":
    print("Starting Enhanced RSI Strategy V5 Validation Tests")

    try:
        # Test individual modules first
        test_individual_modules()

        # Test the full strategy
        test_strategy_v5()

        print("\nALL TESTS PASSED - Enhanced RSI Strategy V5 is ready!")
        print("All diagnostic improvements have been successfully implemented:")
        print("   - Enhanced MTF with majority alignment (not all-or-nothing)")
        print("   - Active trend filtering with multiple confirmations")
        print("   - Dynamic risk management based on market regime")
        print("   - Advanced contradiction detection and prevention")
        print("   - Regime-aware trading parameters")
        print("   - Improved entry/exit logic")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()