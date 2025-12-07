from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

router = APIRouter()

class RiskConfig(BaseModel):
    risk_per_trade: float
    max_position_size: float
    max_daily_loss: float

# Global risk configuration
_risk_config = {
    "risk_per_trade": 0.015,
    "max_position_size": 0.1,
    "max_daily_loss": 0.05
}

@router.get("/symbols")
def get_available_symbols():
    """Get list of available trading symbols"""
    # Common crypto and forex symbols
    symbols = {
        "crypto": [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", 
            "DOGEUSDT", "XRPUSDT", "SOLUSDT", "DOTUSDT"
        ],
        "forex": [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
            "USDCAD", "NZDUSD", "EURGBP", "EURJPY"
        ]
    }
    
    return {
        "symbols": symbols,
        "total": sum(len(v) for v in symbols.values())
    }

@router.get("/timeframes")
def get_available_timeframes():
    """Get supported timeframes"""
    timeframes = {
        "intraday": ["1m", "5m", "15m", "30m", "1h", "4h"],
        "daily": ["1d"],
        "weekly": ["1w"]
    }
    
    return {
        "timeframes": timeframes,
        "descriptions": {
            "1m": "1 Minute",
            "5m": "5 Minutes",
            "15m": "15 Minutes",
            "30m": "30 Minutes",
            "1h": "1 Hour",
            "4h": "4 Hours",
            "1d": "1 Day",
            "1w": "1 Week"
        }
    }

@router.get("/risk")
def get_risk_config():
    """Get current risk configuration"""
    return _risk_config

@router.put("/risk")
def update_risk_config(config: RiskConfig):
    """Update risk management configuration"""
    global _risk_config
    
    try:
        # Validate ranges
        if not (0 < config.risk_per_trade <= 0.05):
            raise HTTPException(status_code=400, detail="risk_per_trade must be between 0 and 0.05")
        
        if not (0 < config.max_position_size <= 1.0):
            raise HTTPException(status_code=400, detail="max_position_size must be between 0 and 1.0")
        
        if not (0 < config.max_daily_loss <= 0.2):
            raise HTTPException(status_code=400, detail="max_daily_loss must be between 0 and 0.2")
        
        # Update config
        _risk_config = config.dict()
        
        return {
            "status": "success",
            "message": "Risk configuration updated",
            "config": _risk_config
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
