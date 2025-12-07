"""
Updated data fetcher using the new provider system
"""

import pandas as pd
from providers.provider_registry import DataProviderRegistry
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class DataFetcher:
    """Data fetcher that uses the provider registry with failover capabilities"""
    
    def __init__(self, test_mode: bool = False):
        self.provider_registry = DataProviderRegistry(test_mode=test_mode)
        logger.info(f"DataFetcher initialized with test_mode={test_mode}")
    
    def fetch_market_data(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
        data_source: str = "AUTO"
    ) -> pd.DataFrame:
        """Fetch market data using provider registry with failover"""
        try:
            result = self.provider_registry.get_data(symbol, interval, limit)
            
            if result['success']:
                logger.info(f"✅ Data fetched from {result['source']}: {len(result['data'])} candles for {symbol}")
                return result['data']
            else:
                logger.error(f"❌ Failed to fetch data for {symbol} from all providers:")
                for provider, error in result.get('errors', {}).items():
                    logger.error(f"  {provider}: {error}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error in fetch_market_data: {e}")
            return pd.DataFrame()
    
    def test_connection(self, data_source: str = "AUTO") -> bool:
        """Test connection to data providers"""
        try:
            connections = self.provider_registry.test_all_connections()
            logger.info(f"Provider connections: {connections}")
            
            # Return True if at least one provider is available
            return any(connections.values())
        except Exception as e:
            logger.error(f"Error testing connections: {e}")
            return False
    
    def get_available_symbols(self, data_source: str = "AUTO") -> List[str]:
        """Get available symbols from providers"""
        try:
            symbols_by_provider = self.provider_registry.get_available_symbols()
            
            # Combine all symbols from all providers
            all_symbols = set()
            for provider_symbols in symbols_by_provider.values():
                all_symbols.update(provider_symbols)
                
            return list(all_symbols)
        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            # Return some default symbols as fallback
            return ["BTCUSDT", "ETHUSDT", "EURUSD", "GBPUSD"]