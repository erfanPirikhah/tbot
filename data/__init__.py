# data/__init__.py

from .data_fetcher import fetch_market_data, get_current_price, set_cryptocompare_api_key
from .mt5_data import MT5_AVAILABLE, mt5_fetcher

__all__ = [
    'fetch_market_data', 
    'get_current_price', 
    'set_cryptocompare_api_key',
    'MT5_AVAILABLE',
    'mt5_fetcher'
]