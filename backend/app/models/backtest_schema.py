from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime

class BacktestRequest(BaseModel):
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    days: int = 30
    strategy_params: Optional[Dict[str, Any]] = None

class BacktestResponse(BaseModel):
    task_id: str
    status: str
    message: str

class BacktestResult(BaseModel):
    task_id: str
    status: str
    timestamp: datetime
    metrics: Optional[Dict[str, Any]] = None
    equity_curve: Optional[List[Dict[str, Any]]] = None
    trades: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
