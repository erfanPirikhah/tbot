#!/usr/bin/env python3
"""
Test script to verify AdvancedMarketFilters integration.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project path
project_path = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_path)

def test_advanced_filters_integration():
    """Test that AdvancedMarketFilters is properly integrated into strategies."""
    print("Testing Advanced Filters Integration...")
    
    try:
        # Test 1: Import AdvancedMarketFilters
        from strategies.advanced_filters import AdvancedMarketFilters
        print("[OK] AdvancedMarketFilters import successful")
        
        # Test 2: Import EnhancedRsiStrategyV5 with AdvancedMarketFilters
        from strategies.enhanced_rsi_strategy_v5 import EnhancedRsiStrategyV5
        print("[OK] EnhancedRsiStrategyV5 import successful")
        
        # Test 3: Import EnsembleStrategyV4 with AdvancedMarketFilters
        from strategies.ensemble_strategy_v4 import EnsembleRsiStrategyV4
        print("[OK] EnsembleStrategyV4 import successful")
        
        # Test 4: Create sample data
        sample_data = pd.DataFrame({
            'open': np.random.random(50) * 100 + 1000,
            'high': np.random.random(50) * 105 + 1000,
            'low': np.random.random(50) * 95 + 1000,
            'close': np.random.random(50) * 100 + 1000,
            'volume': np.random.random(50) * 1000
        })
        sample_data['close'].iloc[0] = 1000  # Start with a known value
        for i in range(1, len(sample_data)):
            sample_data.loc[sample_data.index[i], 'close'] = sample_data['close'].iloc[i-1] + (np.random.random() - 0.5) * 10
        
        sample_data.index = pd.date_range(start='2023-01-01', periods=50, freq='H')
        print("[OK] Sample data created")
        
        # Test 5: Test EnhancedRsiStrategyV5 with advanced filters
        enhanced_strategy = EnhancedRsiStrategyV5(
            enable_advanced_filters=True,
            advanced_filter_confidence_threshold=0.6,
            market_strength_min_score=3.0
        )
        print("[OK] EnhancedRsiStrategyV5 with advanced filters initialized")
        
        # Test 6: Test EnsembleStrategyV4 with advanced filters
        ensemble_strategy = EnsembleRsiStrategyV4(
            enable_advanced_filters=True,
            advanced_filter_confidence_threshold=0.6,
            market_strength_min_score=3.0
        )
        print("[OK] EnsembleStrategyV4 with advanced filters initialized")
        
        # Test 7: Test AdvancedMarketFilters directly
        regime = AdvancedMarketFilters.detect_market_regime(sample_data)
        print(f"[OK] Market regime detection: {regime}")
        
        trend = AdvancedMarketFilters.calculate_trend_strength(sample_data)
        print(f"[OK] Trend strength calculation: {trend}")
        
        support, resistance = AdvancedMarketFilters.calculate_support_resistance(sample_data)
        print(f"[OK] Support/Resistance levels: Support={support:.2f}, Resistance={resistance:.2f}")
        
        # Test 8: Test filter evaluation methods
        from strategies.enhanced_rsi_strategy_v5 import PositionType
        eval_result = enhanced_strategy._evaluate_advanced_filters(sample_data, PositionType.LONG)
        print(f"[OK] Enhanced strategy filter evaluation: {eval_result[0]} - {len(eval_result[1])} conditions")
        
        eval_result_ensemble = ensemble_strategy._evaluate_advanced_filters(sample_data, PositionType.SHORT)
        print(f"[OK] Ensemble strategy filter evaluation: {eval_result_ensemble[0]} - {len(eval_result_ensemble[1])} conditions")
        
        # Test 9: Test entry conditions with advanced filters
        entry_result = enhanced_strategy.check_entry_conditions(sample_data, PositionType.LONG)
        print(f"[OK] Enhanced strategy entry conditions: {entry_result[0]} - {len(entry_result[1])} conditions")
        
        entry_result_ensemble = ensemble_strategy.check_entry_conditions(sample_data, PositionType.LONG)
        print(f"[OK] Ensemble strategy entry conditions: {entry_result_ensemble[0]} - {len(entry_result_ensemble[1])} conditions")
        
        print("\n[SUCCESS] All tests passed! Advanced filters integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_advanced_filters_integration()
    if success:
        print("\n[SUCCESS] Integration test completed successfully!")
    else:
        print("\n[ERROR] Integration test failed!")
        sys.exit(1)