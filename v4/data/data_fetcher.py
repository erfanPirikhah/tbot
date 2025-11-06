# data/data_fetcher.py

import pandas as pd
import cryptocompare
from datetime import datetime
import logging
from typing import Optional, Dict, Any
import sys
import os
import warnings

# Set project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

warnings.filterwarnings('ignore')

# Import MT5 module
try:
    from .mt5_data import mt5_fetcher, MT5_AVAILABLE
except ImportError as e:
    MT5_AVAILABLE = False
    mt5_fetcher = None
    logging.warning(f"MetaTrader5 not available: {e}")

logger = logging.getLogger(__name__)

class DataFetcher:
    """Main class for fetching data from various sources"""
    
    def __init__(self, crypto_api_key: Optional[str] = None):
        self.crypto_api_key = crypto_api_key
        if crypto_api_key:
            self.set_cryptocompare_api_key(crypto_api_key)
        
        logger.info("✅ DataFetcher initialized successfully")

    def fetch_market_data(
        self, 
        symbol: str, 
        interval: str, 
        limit: int = 100, 
        data_source: str = "AUTO"
    ) -> pd.DataFrame:
        """
        Fetch market data from various sources
        
        Args:
            symbol: Target symbol
            interval: Data timeframe
            limit: Number of data points needed
            data_source: Data source (AUTO, MT5, CRYPTOCOMPARE)
        """
        try:
            from config.market_config import SYMBOL_MAPPING, TIMEFRAME_MAPPING
            
            # Auto-detect data source
            if data_source == "AUTO":
                data_source = self._detect_data_source(symbol)
            
            logger.info(f"📥 Fetching data for {symbol} from {data_source} with timeframe {interval}")
            
            if data_source == "MT5" and MT5_AVAILABLE:
                return self.fetch_mt5_data(symbol, interval, limit)
            else:
                return self.fetch_cryptocompare_data(symbol, interval, limit)
                
        except Exception as e:
            logger.error(f"❌ Error fetching data: {e}")
            raise

    def _detect_data_source(self, symbol: str) -> str:
        """Auto-detect data source based on symbol"""
        from config.market_config import SYMBOL_MAPPING
        
        # Check MT5 symbols
        mt5_symbols = SYMBOL_MAPPING["MT5"]
        if symbol.upper() in mt5_symbols.values() or symbol in mt5_symbols.values():
            if MT5_AVAILABLE and mt5_fetcher and mt5_fetcher.ensure_connected():
                return "MT5"
        
        # Default to CryptoCompare
        return "CRYPTOCOMPARE"

    def fetch_mt5_data(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """Fetch data from MetaTrader5"""
        try:
            # Timeframe mapping
            interval_map = {
                "1 minute": "M1", "1m": "M1", "M1": "M1",
                "5 minutes": "M5", "5m": "M5", "M5": "M5",
                "15 minutes": "M15", "15m": "M15", "M15": "M15",
                "30 minutes": "M30", "30m": "M30", "M30": "M30",
                "1 hour": "H1", "1h": "H1", "H1": "H1",
                "4 hours": "H4", "4h": "H4", "H4": "H4",
                "1 day": "D1", "1d": "D1", "D1": "D1",
                "1 week": "W1", "1w": "W1", "W1": "W1"
            }
            
            mt5_interval = interval_map.get(interval, "H1")
            logger.info(f"🔍 Fetching MT5 data for {symbol} with timeframe {mt5_interval}")
            
            if not MT5_AVAILABLE or not mt5_fetcher:
                raise ValueError("❌ MT5 not available")
                
            data = mt5_fetcher.fetch_market_data(symbol, mt5_interval, limit)
            
            if data.empty:
                raise ValueError(f"❌ No data received from MT5 for {symbol}")
                
            logger.info(f"✅ Received {len(data)} records from MT5 for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error fetching MT5 data for {symbol}: {str(e)}")
            raise

    def fetch_cryptocompare_data(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """Fetch data from CryptoCompare"""
        try:
            from config.market_config import CRYPTOCOMPARE_INTERVAL_MAP
            
            interval_param = CRYPTOCOMPARE_INTERVAL_MAP.get(interval)
            
            if not interval_param:
                raise ValueError(f"❌ Timeframe '{interval}' not supported.")

            logger.info(f"🔍 Fetching data from CryptoCompare for {symbol} ({interval_param})")

            # Data limit
            actual_limit = min(limit, 2000)  # API limit
            
            if interval_param in ['1m', '5m', '15m', '30m']:
                data = cryptocompare.get_historical_price_minute(
                    symbol, 
                    currency='USD', 
                    limit=actual_limit,
                    toTs=datetime.now()
                )
            elif interval_param == '1h':
                data = cryptocompare.get_historical_price_hour(
                    symbol, 
                    currency='USD', 
                    limit=actual_limit,
                    toTs=datetime.now()
                )
            elif interval_param == '1d':
                data = cryptocompare.get_historical_price_day(
                    symbol,
                    currency='USD',
                    limit=actual_limit,
                    toTs=datetime.now()
                )
            else:  # '1w'
                data = cryptocompare.get_historical_price_day(
                    symbol,
                    currency='USD', 
                    limit=min(actual_limit * 7, 2000),
                    toTs=datetime.now()
                )

            if not data:
                raise Exception("❌ No data received from API.")

            df = pd.DataFrame(data)
            
            # Process and standardize data
            df = self._process_cryptocompare_data(df, interval_param)
            
            if limit and len(df) > limit:
                df = df.tail(limit)
            
            logger.info(f"✅ Received {len(df)} records from CryptoCompare for {symbol}")
            return df

        except Exception as e:
            logger.error(f"❌ Error fetching data from CryptoCompare: {str(e)}")
            raise

    def _process_cryptocompare_data(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Process CryptoCompare data"""
        try:
            # Column mapping
            column_mapping = {
                'time': 'open_time',
                'open': 'open', 
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volumefrom': 'volume',
                'volumeto': 'volume_usd'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Convert time
            if 'open_time' in df.columns:
                df['open_time'] = pd.to_datetime(df['open_time'], unit='s')
                df.set_index('open_time', inplace=True)
            
            # Convert data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove invalid data
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            df = df.sort_index().reset_index(drop=False)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error processing CryptoCompare data: {e}")
            return df

    def get_current_price(self, symbol: str, data_source: str = "AUTO") -> float:
        """Get current price - error resistant version"""
        logger.info(f"💰 Getting current price for {symbol}")
        
        # Detect data source
        if data_source == "AUTO":
            data_source = self._detect_data_source(symbol)
        
        logger.info(f"🔍 Using data source: {data_source} for symbol: {symbol}")
        
        if data_source == "MT5" and MT5_AVAILABLE:
            return self._get_mt5_current_price(symbol)
        else:
            return self._get_cryptocompare_current_price(symbol)

    def _get_mt5_current_price(self, symbol: str) -> float:
        """Get price from MT5"""
        try:
            price = mt5_fetcher.get_current_price(symbol)
            if price > 0:
                logger.info(f"✅ MT5 price for {symbol}: {price:.5f}")
                return price
            else:
                logger.warning(f"⚠️ MT5 price for {symbol} is invalid")
                return self._get_fallback_price(symbol, "MT5")
                
        except Exception as e:
            logger.error(f"❌ Error getting MT5 price for {symbol}: {e}")
            return self._get_fallback_price(symbol, "MT5")

    def _get_cryptocompare_current_price(self, symbol: str) -> float:
        """Get price from CryptoCompare"""
        try:
            price_data = cryptocompare.get_price(symbol, currency='USD')
            if price_data and symbol in price_data:
                price = float(price_data[symbol]['USD'])
                logger.info(f"✅ CryptoCompare price for {symbol}: {price:.5f}")
                return price
            else:
                logger.warning(f"⚠️ CryptoCompare price for {symbol} not found")
                return self._get_fallback_price(symbol, "CRYPTOCOMPARE")
                
        except Exception as e:
            logger.error(f"❌ Error getting current price for {symbol}: {str(e)}")
            return self._get_fallback_price(symbol, "CRYPTOCOMPARE")

    def _get_fallback_price(self, symbol: str, data_source: str) -> float:
        """Get fallback price from historical data"""
        try:
            logger.info(f"🔄 Using fallback for {symbol} price")
            
            # Get recent historical data
            data = self.fetch_market_data(symbol, "H1", 2, data_source)
            
            if not data.empty and 'close' in data.columns:
                price = float(data['close'].iloc[-1])
                logger.info(f"✅ Fallback price from historical data for {symbol}: {price:.5f}")
                return price
            else:
                logger.error(f"❌ Fallback also failed for {symbol}")
                return 0.0
                
        except Exception as e:
            logger.error(f"❌ Error in price fallback for {symbol}: {e}")
            return 0.0

    def get_symbol_info(self, symbol: str, data_source: str = "AUTO") -> Optional[Dict[str, Any]]:
        """Get symbol information"""
        if data_source == "AUTO":
            data_source = self._detect_data_source(symbol)
        
        if data_source == "MT5" and MT5_AVAILABLE:
            return mt5_fetcher.get_symbol_info(symbol)
        else:
            return self._get_cryptocompare_symbol_info(symbol)

    def _get_cryptocompare_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol info from CryptoCompare"""
        try:
            # This function can be completed with other APIs
            return {
                'name': symbol,
                'description': f'Cryptocurrency {symbol}',
                'source': 'CRYPTOCOMPARE'
            }
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None

    def get_available_symbols(self, data_source: str = "MT5") -> list:
        """Get list of available symbols"""
        if data_source == "MT5" and MT5_AVAILABLE:
            return mt5_fetcher.get_available_symbols()
        else:
            from config.market_config import SYMBOL_MAPPING
            return list(SYMBOL_MAPPING["CRYPTOCOMPARE"].values())

    def set_cryptocompare_api_key(self, api_key: str):
        """Set API Key for CryptoCompare"""
        try:
            cryptocompare.cryptocompare._set_api_key_parameter(api_key)
            self.crypto_api_key = api_key
            logger.info("✅ CryptoCompare API Key set successfully")
        except Exception as e:
            logger.error(f"❌ Error setting CryptoCompare API key: {e}")

    def test_connection(self, data_source: str = "AUTO") -> bool:
        """Test connection to data sources"""
        try:
            if data_source in ["AUTO", "MT5"] and MT5_AVAILABLE:
                if mt5_fetcher and mt5_fetcher.ensure_connected():
                    logger.info("✅ MT5 connection test: PASSED")
                    return True
                else:
                    logger.warning("⚠️ MT5 connection test: FAILED")
            
            if data_source in ["AUTO", "CRYPTOCOMPARE"]:
                # Simple test with Bitcoin price
                price = self.get_current_price("BTC", "CRYPTOCOMPARE")
                if price > 0:
                    logger.info("✅ CryptoCompare connection test: PASSED")
                    return True
                else:
                    logger.warning("⚠️ CryptoCompare connection test: FAILED")
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False

# Create global instance
data_fetcher = DataFetcher()