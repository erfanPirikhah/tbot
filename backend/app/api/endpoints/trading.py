from fastapi import APIRouter, HTTPException
from app.models.trading_schema import (
    TradingStartRequest, TradingStatusResponse, 
    Position, TradeHistory
)
from app.services.trading_service import trading_service

router = APIRouter()

@router.post("/start")
async def start_trading(request: TradingStartRequest):
    """Start live trading"""
    try:
        result = await trading_service.start_trading(
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy_params=request.strategy_params,
            duration_hours=request.duration_hours
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_trading():
    """Stop live trading"""
    try:
        result = await trading_service.stop_trading()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=TradingStatusResponse)
def get_trading_status():
    """Get current trading status"""
    return trading_service.get_status()

@router.get("/positions")
def get_positions():
    """Get open positions"""
    return {"positions": trading_service.get_positions()}

@router.get("/history")
def get_trade_history(limit: int = 100):
    """Get trade history"""
    return {"trades": trading_service.get_history(limit)}
