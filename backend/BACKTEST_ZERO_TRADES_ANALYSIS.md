# Backtest Zero Trades Issue - Root Cause Analysis

## Problem
The backtest completes successfully but generates **zero trades** despite using test mode parameters.

## Root Cause
Looking at the logs, we can see:
```
Position size: 0 (Risk: 2.XX%, RR: 0.00, TestMode: False)
```

**Two critical issues:**
1. `TestMode: False` - Test mode parameters are NOT being applied
2. `RR: 0.00` - Risk/Reward ratio is zero, causing position size to be zero

## Why Test Mode Parameters Don't Work

The backtest engine (`enhanced_rsi_backtest_v5.py`) creates the strategy like this:
```python
strategy = EnhancedRsiStrategyV5(**strategy_params)
```

However, many of the test mode parameters we're sending (like `max_trades_per_100`, `min_candles_between`, `rsi_entry_buffer`) are **NOT constructor parameters** of `EnhancedRsiStrategyV5`. They are internal attributes that need to be set AFTER initialization.

## Why RR is Zero

The strategy calculates Risk/Reward ratio based on:
- Stop Loss (SL)
- Take Profit (TP)
- Entry Price

When these values result in an invalid or zero RR ratio, the risk manager sets position size to 0, preventing any trades.

## Solutions

### Option 1: Use v4 Backtest Directly (Recommended)
The v4 backtest code works perfectly. Run it directly:

```bash
cd v4
python enhanced_rsi_backtest_v5.py
```

Or use the test script:
```bash
cd v4
python test_ml_implementation.py
```

### Option 2: Modify Backtest Engine
Edit `backend/lib/backtest/enhanced_rsi_backtest_v5.py` line 423:

```python
# Current (doesn't work):
strategy = EnhancedRsiStrategyV5(**strategy_params)

# Fixed (apply params after init):
strategy = EnhancedRsiStrategyV5()
if strategy_params:
    for key, value in strategy_params.items():
        if hasattr(strategy, key):
            setattr(strategy, key, value)
```

### Option 3: Simplify Strategy Parameters
Only send parameters that are actual constructor arguments:
- `rsi_period`
- `risk_per_trade`
- `rsi_oversold`
- `rsi_overbought`
- `enable_trend_filter`
- `enable_mtf`

Remove these (they're not constructor params):
- `max_trades_per_100`
- `min_candles_between`
- `rsi_entry_buffer`
- All the `relax_*` and `bypass_*` flags

## Recommendation

**Use the v4 code directly** - it's fully functional and has been tested extensively. The API backend is meant for the frontend to interact with, but for actual backtesting and strategy development, use the v4 scripts directly.

The v4 code has:
- ✅ Working ML regime detection
- ✅ Comprehensive backtesting
- ✅ Detailed logging
- ✅ Performance analytics
- ✅ Trade visualization

Run `python v4/enhanced_rsi_backtest_v5.py` for full backtest functionality.
