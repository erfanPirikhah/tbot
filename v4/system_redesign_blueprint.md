# System Redesign Blueprint

## Current State Analysis
Based on the diagnostic analysis, the current system has several critical issues:
- Low win rate (~31%) due to overly restrictive filters
- High sensitivity to market regime changes
- MTF logic too restrictive (all-or-nothing approach)
- No effective trend filter (intentionally disabled)
- Static parameters across all market conditions
- Suboptimal risk/reward ratios
- Inadequate trade management

## Redesigned Architecture

### 1. Strategy Logic Module
- Main strategy class with improved entry/exit logic
- Enhanced RSI calculations with momentum confirmation
- Modular filter system with scoring mechanism

### 2. Trend Filter Module
- Multi-indicator trend detection (EMA alignment, ADX, price action)
- Trend strength scoring with dynamic thresholds
- Integration with entry logic

### 3. MTF Module
- Weighted timeframe alignment (H4, H1, D1)
- Flexible alignment requirements (not all-or-nothing)
- Individual timeframe scoring

### 4. Market Regime Detection
- Volatility-based regime classification
- Trend vs ranging detection
- Regime-specific parameter adjustment

### 5. Volatility Filter Module
- Dynamic volatility thresholds
- Historical volatility context
- Market condition adaptation

### 6. Enhanced Entry/Exit Logic
- Momentum confirmation for entries
- Improved trailing stops
- Optimized partial exit thresholds

### 7. Dynamic Risk Management
- Volatility-adjusted position sizing
- Regime-specific risk percentages
- Risk-on/risk-off switches

### 8. Parameter Profiles
- Market regime-specific parameter sets
- Adaptive parameter selection
- Performance-based optimization

### 9. Contradiction Detection
- Multi-indicator conflict detection
- Signal quality scoring
- Risk mitigation for conflicting signals

### 10. Modular Components
- Separated concerns for maintainability
- Interface-based design for easy testing
- Plug-and-play filter system

## Mapping: What to KEEP, REMOVE, REWRITE, MERGE, DEPRECATE, ADD

### KEEP:
- Core RSI calculation logic (with improvements)
- Fundamental trade management structure
- Portfolio tracking functionality
- Basic MongoDB logging structure
- Strategy class inheritance pattern

### REMOVE:
- All-or-nothing MTF requirements (mtf_require_all: True)
- Static risk management
- Inflexible RSI thresholds (35/65)
- Disabled trend filter approach
- Fixed stop loss multipliers

### REWRITE:
- MTF alignment logic (weighted scoring)
- Entry condition validation
- Trend detection algorithm
- Market regime detection
- Risk management system
- Exit logic with dynamic trailing stops

### MERGE:
- Advanced filter logic with main strategy
- Regime detection with risk management
- Contradiction detection with entry validation

### DEPRECATE:
- Old parameter sets for static conditions
- Old MTF implementation
- Fixed volatility multipliers

### ADD:
- Market regime classifier
- Dynamic risk calculator
- Contradiction detection system
- Filter scoring mechanism
- Timeframe weighting system
- Enhanced logging for new metrics

## Required Improvements Implementation List

1. **MTF Logic**: Change from "all-align" to "majority-align" with weighted scoring
2. **Trend Filter**: Implement multi-indicator trend detection with strength scoring
3. **Market Regime**: Add volatility-based regime detection and parameter adaptation
4. **Entry Conditions**: Add momentum confirmation and volume validation
5. **Risk Management**: Create dynamic risk based on volatility and regime
6. **Exit Logic**: Optimize trailing stop activation and partial exit thresholds
7. **Parameter Tuning**: Adjust RSI levels to 30/70 with 3 buffer
8. **Contradiction Detection**: Implement multi-indicator conflict checking
9. **Code Structure**: Modularize components with clear interfaces
10. **Database Logging**: Add regime, filter scores, and dynamic risk logging

## Implementation Priority

### Phase 1 - Quick Wins (Immediate Win Rate Improvement)
1. Adjust RSI levels to 30/70 with 3 buffer
2. Change MTF from "all" to "majority" alignment
3. Reduce MTF thresholds to 40/60

### Phase 2 - Medium Term (Risk & Adaptation)
1. Implement market regime detection
2. Create dynamic risk management
3. Enhance trend filter with multiple confirmations
4. Add volatility-based adjustments

### Phase 3 - Long Term (Advanced Features)
1. Complete architecture refactoring
2. Advanced regime detection
3. Multi-indicator contradiction detection
4. Performance optimization

## Component Interface Definitions

### IFilter Interface
```python
from abc import ABC, abstractmethod
from typing import Tuple, List

class IFilter(ABC):
    @abstractmethod
    def evaluate(self, data, position_type) -> Tuple[bool, str, float]:
        """
        Returns: (is_passed, description, confidence_score)
        """
        pass
```

### IRegimeDetector Interface
```python
from abc import ABC, abstractmethod
from typing import Tuple

class IRegimeDetector(ABC):
    @abstractmethod
    def detect_regime(self, data) -> Tuple[str, float]:
        """
        Returns: (regime_type, confidence)
        Regime types: 'TRENDING', 'RANGING', 'VOLATILE', 'NORMAL', 'UNKNOWN'
        """
        pass
```

### IRiskManager Interface
```python
from abc import ABC, abstractmethod

class IRiskManager(ABC):
    @abstractmethod
    def calculate_position_size(self, data, entry_price, stop_loss, regime) -> float:
        pass
    
    @abstractmethod
    def calculate_dynamic_risk(self, data, regime) -> float:
        pass
```

This redesign blueprint addresses all issues from the diagnostic report and provides a clear path for implementing the comprehensive improvements needed for the system.