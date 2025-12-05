# Baseline Report

## Run Summary

### Environment
- Repository: D:\Project\crypto\v4
- Timestamp: 2025-12-05 11:29:40
- Runtime: Windows 10/11
- Python: 3.13

### Test Configuration
- Symbols: BTCUSDT, ETHUSDT
- Timeframes: H1, H4
- Days back: 30
- Strategy: Enhanced RSI Strategy V5 (Conservative Parameters)
- Data source: Simulated (MT5 unavailable)
- Initial capital: $10,000.00
- Commission: 0.03%
- Slippage: 0.01%

### Runtime Status
- Status: SUCCESS
- Runtime duration: ~30 seconds
- No crashes or errors during execution
- All artifacts generated successfully
- Note: No actual trades were generated due to conservative parameters filtering all signals

### Warnings
- MT5 connection unavailable - using simulated data
- All signals were filtered by the contradiction detection system (safety checks failed)
- Zero trades across all symbol/timeframe combinations

## Performance Metrics

| Symbol_TF | WinRate | PF | Sharpe | MDD | AvgTrade | TradesCount |
|-----------|---------|----|--------|-----|----------|-------------|
| BTCUSDT_H1 | 0.00% | Inf | 0.0000 | 0.00% | 0.00 | 0 |
| BTCUSDT_H4 | 0.00% | Inf | 0.0000 | 0.00% | 0.00 | 0 |
| ETHUSDT_H1 | 0.00% | Inf | 0.0000 | 0.00% | 0.00 | 0 |
| ETHUSDT_H4 | 0.00% | Inf | 0.0000 | 0.00% | 0.00 | 0 |

## Top 10 Critical Issues Identified

1. **Signal Filtering**: Contradiction detection system is overly conservative - blocking all trades (Signal SAFETY CHECK FAILED messages seen in output)

2. **MT5 Dependency**: System heavily relies on MT5 which was unavailable during test

3. **No Trading Results**: Zero trades generated across all configurations indicates potential over-filtering

4. **Risk Management**: Risk filters too strict preventing any position entries

5. **Market Regime Detection**: May be marking simulated data as unsuitable for trading

6. **Parameter Configuration**: Conservative parameters may be too restrictive

7. **MongoDB Dependency**: Logging system relies on MongoDB for operation

8. **Lack of Fallback**: No viable fallback for when MT5 is unavailable

9. **Performance Tracking**: No real performance metrics due to zero trades

10. **Indicators Compatibility**: Simulated data may not trigger RSI-based signals effectively

## Recommendations

1. **Calibrate Contradiction Detection**: Review and adjust the contradiction detection system to allow more acceptable trades
2. **Improve Data Source Fallback**: Enhance simulated data generation to better mimic real market conditions
3. **Parameter Tuning**: Adjust conservative parameters to allow for more trading opportunities while maintaining risk management
4. **Add Test Mode**: Implement a testing mode that bypasses certain filters for validation
5. **Multiple Data Sources**: Implement CryptoCompare or other data sources as primary option for crypto pairs