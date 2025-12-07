from pydantic import BaseModel
from typing import Optional, Dict, Any

class StrategyConfigRequest(BaseModel):
    rsi_period: Optional[int] = 14
    risk_per_trade: Optional[float] = 0.015
    rsi_oversold: Optional[int] = 30
    rsi_overbought: Optional[int] = 70
    enable_trend_filter: Optional[bool] = True
    enable_mtf: Optional[bool] = True
    additional_params: Optional[Dict[str, Any]] = None

class StrategyParametersResponse(BaseModel):
    current_params: Dict[str, Any]
    strategy_class: str

class MLRetrainRequest(BaseModel):
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    days: int = 90
    
class MLRetrainResponse(BaseModel):
    status: str
    message: str
    model_path: Optional[str] = None
