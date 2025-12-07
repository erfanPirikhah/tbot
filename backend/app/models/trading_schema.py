from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime

class TradingStartRequest(BaseModel):
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    strategy_params: Optional[Dict[str, Any]] = None
    duration_hours: Optional[int] = None

class TradingStatusResponse(BaseModel):
    is_running: bool
    current_position: str
    portfolio_value: float
    active_symbols: List[str]
    uptime_seconds: Optional[int] = None

class Position(BaseModel):
    symbol: str
    side: str  # LONG/SHORT
    entry_price: float
    current_price: float
    quantity: float
    pnl_percentage: float
    pnl_amount: float
    stop_loss: float
    take_profit: float

class TradeHistory(BaseModel):
    timestamp: datetime
    symbol: str
    side: str
    action: str  # ENTRY/EXIT
    price: float
    quantity: Optional[float] = None
    pnl_percentage: Optional[float] = None
    pnl_amount: Optional[float] = None
    reason: Optional[str] = None
