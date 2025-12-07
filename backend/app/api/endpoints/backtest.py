from fastapi import APIRouter, HTTPException
from app.services.backtest_service import BacktestService
from app.models.backtest_schema import BacktestRequest, BacktestResponse, BacktestResult

router = APIRouter()

@router.post("/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """Start a new backtest task"""
    try:
        task_id = await BacktestService.start_backtest(
            symbol=request.symbol,
            timeframe=request.timeframe,
            days=request.days,
            strategy_params=request.strategy_params
        )
        return {"task_id": task_id, "status": "started", "message": "Backtest started in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{task_id}/results", response_model=BacktestResult)
async def get_test_results(task_id: str):
    """Get the results of a specific backtest"""
    result = BacktestService.get_result(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Backtest not found")
    
    return result

@router.get("/{task_id}/equity-curve")
async def get_equity_curve(task_id: str):
    """Get equity curve data for visualization"""
    result = BacktestService.get_result(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Backtest not found")
    
    if result.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Backtest not completed yet")
    
    equity_curve = result.get("equity_curve", [])
    
    return {
        "task_id": task_id,
        "equity_curve": equity_curve,
        "count": len(equity_curve)
    }
