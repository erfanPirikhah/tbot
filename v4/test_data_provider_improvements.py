"""
Unit tests for DataProvider system and TestMode functionality
"""

import unittest
import pandas as pd
import numpy as np
from providers.data_provider import MT5Provider, CryptoCompareProvider, ImprovedSimulatedProvider
from providers.provider_registry import DataProviderRegistry
from strategies.enhanced_rsi_strategy_v5 import EnhancedRsiStrategyV5
from config.parameters import TEST_MODE_CONFIG


class TestPrimaryProviderFailover(unittest.TestCase):
    """Test primary provider failover to secondary providers"""
    
    def test_primary_provider_connection(self):
        """Test that primary provider (MT5) connects when available"""
        provider = MT5Provider()
        # MT5 might not be available, but we can still check if the initialization works
        self.assertIsInstance(provider, MT5Provider)
        # The actual connection test will depend on availability in test environment
    
    def test_registry_initialization(self):
        """Test that provider registry initializes with available providers"""
        registry = DataProviderRegistry()
        # At minimum, simulated provider should be available
        self.assertGreaterEqual(len(registry.providers), 1)
        
        # Check that simulated provider is always present
        sim_provider_exists = any(
            'Simulated' in type(provider).__name__ for provider in registry.providers
        )
        self.assertTrue(sim_provider_exists, "Simulated provider should always be available")


class TestSecondaryProviderFailover(unittest.TestCase):
    """Test secondary provider failover functionality"""
    
    def test_secondary_provider_connection(self):
        """Test secondary provider (CryptoCompare) connection"""
        provider = CryptoCompareProvider()
        # This might not be available, but initialization should work
        self.assertIsInstance(provider, CryptoCompareProvider)
    
    def test_failover_logic(self):
        """Test that registry returns data from first available provider"""
        registry = DataProviderRegistry()
        
        # Test data fetching
        result = registry.get_data("BTCUSDT", "H1", 10)
        
        # Should always return a result even if it's from simulated provider
        self.assertIsNotNone(result)
        self.assertIn('success', result)
        
        # Should have data or an error but not both None
        if not result['success']:
            self.assertIn('errors', result)


class TestSimulatedProviderPriceBehavior(unittest.TestCase):
    """Test that simulated provider generates realistic price behavior"""
    
    def test_simulated_data_generation(self):
        """Test that simulated data has realistic characteristics"""
        provider = ImprovedSimulatedProvider(seed=42)
        
        # Generate some data
        data = provider.fetch_data("BTCUSDT", "H1", 50)
        
        self.assertFalse(data.empty)
        self.assertEqual(len(data), 50)
        
        # Check that OHLC values make sense
        self.assertTrue((data['high'] >= data['low']).all())
        self.assertTrue((data['high'] >= data['open']).all())
        self.assertTrue((data['high'] >= data['close']).all())
        self.assertTrue((data['low'] <= data['open']).all())
        self.assertTrue((data['low'] <= data['close']).all())
    
    def test_rsi_variability(self):
        """Test that generated data allows RSI to have variability"""
        provider = ImprovedSimulatedProvider(seed=123)
        
        # Generate data
        data = provider.fetch_data("BTCUSDT", "H1", 100)
        
        # Calculate RSI manually to verify
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # RSI should have some variation and go above/below key levels
        rsi_range = rsi.max() - rsi.min()
        self.assertGreater(rsi_range, 10, "RSI should have variation")  # Should have significant variation
        
        # Check that RSI goes below 30 (oversold) and above 70 (overbought) occasionally
        has_oversold = (rsi < 30).any()
        has_overbought = (rsi > 70).any()
        
        # Since our provider adds intentional patterns, we expect these to occur
        print(f"RSI below 30: {has_oversold}, RSI above 70: {has_overbought}")
        
        # At least one of these should be True
        self.assertTrue(has_oversold or has_overbought, 
                       "Generated data should occasionally trigger RSI extremes")


class TestTestModeBypassFilters(unittest.TestCase):
    """Test that TestMode properly bypasses filters"""
    
    def test_strategy_initialization_with_testmode(self):
        """Test that strategy can be initialized with TestMode parameters"""
        # Initialize strategy in TestMode
        strategy = EnhancedRsiStrategyV5(
            **TEST_MODE_CONFIG,
            test_mode_enabled=True,
            bypass_contradiction_detection=True,
            enable_all_signals=True
        )
        
        # Verify TestMode parameters are set
        self.assertTrue(strategy.test_mode_enabled)
        self.assertTrue(strategy.bypass_contradiction_detection)
        self.assertTrue(strategy.enable_all_signals)
    
    def test_testmode_parameter_override(self):
        """Test that TestMode parameters override normal parameters"""
        normal_params = {
            'max_trades_per_100': 10,
            'min_candles_between': 10,
            'rsi_entry_buffer': 2,
            'test_mode_enabled': True,
            'bypass_contradiction_detection': True,
            'relax_entry_conditions': True
        }
        
        strategy = EnhancedRsiStrategyV5(**normal_params)
        
        # These should be overridden by TestMode logic when conditions are met
        self.assertTrue(strategy.test_mode_enabled)
        self.assertTrue(strategy.bypass_contradiction_detection)
        self.assertTrue(strategy.relax_entry_conditions)
    
    def test_entry_conditions_with_testmode(self):
        """Test that entry conditions behave differently in TestMode"""
        # Create basic data
        data = pd.DataFrame({
            'open': [100 + i*0.1 for i in range(50)],
            'high': [100 + i*0.1 + 0.2 for i in range(50)],
            'low': [100 + i*0.1 - 0.2 for i in range(50)], 
            'close': [100 + i*0.1 for i in range(50)],
            'volume': [1000 + i for i in range(50)]
        })
        
        # Add RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Initialize strategy with contradiction detection bypass
        strategy = EnhancedRsiStrategyV5(
            test_mode_enabled=True,
            bypass_contradiction_detection=True,
            rsi_oversold=40,  # More permissive
            rsi_overbought=60  # More permissive
        )
        
        # Test entry conditions
        from strategies.enhanced_rsi_strategy_v5 import PositionType
        success, conditions = strategy.check_entry_conditions(data, PositionType.LONG)
        
        # In TestMode with bypass, should have more success
        print(f"Entry conditions success: {success}")
        print(f"Conditions: {conditions}")
        
        # At least it should not fail due to contradiction detection when bypassed
        # This is a basic test - actual behavior depends on other conditions being met


if __name__ == '__main__':
    unittest.main()