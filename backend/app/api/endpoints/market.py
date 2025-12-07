from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.services.market_service import market_service
from lib.data.data_fetcher import DataFetcher

router = APIRouter()
data_fetcher = DataFetcher()

@router.get("/analysis/{symbol}")
def get_market_analysis(symbol: str, timeframe: str = "1h"):
    """Get current market analysis (Price, Regime, Direction)"""
    result = market_service.analyze_market(symbol=symbol, timeframe=timeframe)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@router.get("/ohlcv/{symbol}")
def get_ohlcv_data(
    symbol: str, 
    timeframe: str = Query("1h", description="Timeframe"),
    limit: int = Query(100, description="Number of candles")
):
    """Get OHLCV historical data"""
    try:
        data = data_fetcher.fetch_market_data(symbol, timeframe, limit=limit)
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        # Convert to dict format
        result = data.reset_index().to_dict(orient='records')
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "count": len(result),
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/indicators/{symbol}")
def get_indicators(
    symbol: str,
    timeframe: str = Query("1h"),
    limit: int = Query(100)
):
    """Get calculated technical indicators"""
    try:
        data = data_fetcher.fetch_market_data(symbol, timeframe, limit=limit)
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        # Calculate basic indicators
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Convert to dict
        result = data.tail(50).reset_index().to_dict(orient='records')
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "indicators": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
