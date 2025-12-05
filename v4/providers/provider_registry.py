"""
Provider Registry with failover logic
"""

from typing import List, Optional, Dict, Any
from providers.data_provider import MT5Provider, CryptoCompareProvider, ImprovedSimulatedProvider


class DataProviderRegistry:
    """Registry for managing data providers with automatic failover"""
    
    def __init__(self, test_mode: bool = False):
        self.providers: List = []
        self.test_mode = test_mode
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available providers in order of preference"""
        # 1. MT5 Provider (primary)
        mt5_provider = MT5Provider()
        if mt5_provider.test_connection():
            self.providers.append(mt5_provider)
        
        # 2. CryptoCompare Provider (secondary)
        cc_provider = CryptoCompareProvider()
        if cc_provider.test_connection():
            self.providers.append(cc_provider)
        
        # 3. Improved Simulated Provider (final fallback)
        sim_provider = ImprovedSimulatedProvider()
        if sim_provider.test_connection():
            self.providers.append(sim_provider)
        
        # If no providers are available, add the simulated one anyway
        if not self.providers:
            self.providers.append(ImprovedSimulatedProvider())
    
    def get_data(self, symbol: str, timeframe: str, limit: int) -> Dict[str, Any]:
        """Fetch data from the first available provider with failover"""
        errors = {}
        
        for i, provider in enumerate(self.providers):
            try:
                data = provider.fetch_data(symbol, timeframe, limit)
                
                # Verify the data is valid
                if data.empty or len(data) < limit * 0.8:  # At least 80% of requested data
                    continue
                    
                return {
                    'data': data,
                    'source': type(provider).__name__,
                    'provider_index': i,
                    'success': True
                }
                
            except Exception as e:
                errors[type(provider).__name__] = str(e)
                continue
        
        # If all providers failed
        return {
            'data': None,
            'source': 'None',
            'provider_index': -1,
            'success': False,
            'errors': errors
        }
    
    def get_available_symbols(self) -> Dict[str, List[str]]:
        """Get available symbols from all providers"""
        result = {}
        for provider in self.providers:
            try:
                symbols = provider.get_available_symbols()
                result[type(provider).__name__] = symbols
            except Exception as e:
                result[type(provider).__name__] = []
        
        return result
    
    def test_all_connections(self) -> Dict[str, bool]:
        """Test connection to all providers"""
        result = {}
        for provider in self.providers:
            result[type(provider).__name__] = provider.test_connection()
        return result