# DataProvider System Structure Documentation

## Overview
This document describes the DataProvider system implemented in Stage 1, which provides robust data fetching capabilities with fallback mechanisms and TestMode functionality.

## Architecture

### DataProvider Interface
```python
class DataProvider(ABC):
    def fetch_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame
    def test_connection(self) -> bool
    def get_available_symbols(self) -> List[str]
```

### Implemented Providers

#### 1. MT5Provider
- Primary data source
- Connects to MetaTrader 5
- Graceful degradation when unavailable

#### 2. CryptoCompareProvider  
- Secondary data source
- Uses cryptocompare library
- Fallback for crypto pairs when MT5 unavailable

#### 3. ImprovedSimulatedProvider
- Tertiary/final fallback
- Generates realistic market data with:
  - Volatility clustering
  - Random walks with trend patterns
  - ATR-driven candle generation
  - RSI-triggering conditions
  - Proper OHLC relationships

## Provider Registry
```python
class DataProviderRegistry:
    - Initializes providers in order of preference
    - Implements automatic failover
    - Returns first successful data fetch
    - Tracks provider status
```

## Data Fetcher Integration
- `data_fetcher.py` now uses the Provider Registry
- Backward compatible with existing code
- Accepts TestMode parameter

## TestMode Implementation

### Configuration
TestMode can be enabled with parameters:
```python
test_mode_enabled: bool
bypass_contradiction_detection: bool
relax_risk_filters: bool  
relax_entry_conditions: bool
enable_all_signals: bool
```

### Behavior Changes in TestMode
- Contradiction detection bypassed
- Conservative filters relaxed
- Entry conditions loosened
- Parameter overrides applied

## Integration Points

### TradingBotV4 Initialization
- Accepts `test_mode` parameter
- Passes test_mode to DataFetcher
- Applies TestMode parameters to strategy initialization

### Backtest Engine
- Accepts injected `data_fetcher` parameter
- Uses DataProvider when available
- Falls back to original MT5 method when not available

### Strategy V5
- New TestMode parameters in constructor
- Conditional logic for contradiction bypassing
- Dynamic parameter adjustment based on TestMode

## Fallback Logic
1. Try MT5Provider (if available)
2. Try CryptoCompareProvider (if available) 
3. Use ImprovedSimulatedProvider (always available)

## Usage Examples

### Basic Usage
```python
bot = TradingBotV4(test_mode=True)
bot.run_backtest(symbol="BTCUSDT", test_mode=True)
```

### Provider Registry Direct
```python
registry = DataProviderRegistry(test_mode=True)
result = registry.get_data("BTCUSDT", "H1", 100)
if result['success']:
    data = result['data']
```

## Files Created/Modified

### New Files
- `providers/data_provider.py` - Provider interface and implementations
- `providers/provider_registry.py` - Provider registry with failover
- `test_data_provider_improvements.py` - Unit tests
- `stage1_smoke_test.py` - Smoke tests

### Modified Files
- `main.py` - Added TestMode support, integrated DataProvider
- `backtest/enhanced_rsi_backtest_v5.py` - Added data_fetcher parameter support
- `config/parameters.py` - Added TEST_MODE_CONFIG
- `strategies/enhanced_rsi_strategy_v5.py` - Added TestMode parameters and logic

## Key Features

### Improved Simulated Data
- Realistic volatility clustering
- Trend and ranging market patterns
- Proper RSI behavior (goes above/below thresholds)
- Accurate OHLC relationships
- Volume simulation

### Robust Error Handling
- Graceful fallback between providers
- Error logging and diagnostics
- Connection testing for all providers

### TestMode Benefits
- Bypasses aggressive filters
- Enables signal generation for testing
- Allows strategy validation without production constraints
- Maintains all core functionality

## Performance Considerations
- Minimal overhead when using existing MT5
- Fast simulated data generation for testing
- Efficient provider selection
- Cached connection tests

This architecture provides a solid foundation for reliable data access with comprehensive fallback mechanisms.