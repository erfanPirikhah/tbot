from fastapi import APIRouter, Query
from typing import Optional, List
from datetime import datetime, timedelta
from app.services.trading_service import trading_service
from app.services.backtest_service import BACKTEST_RESULTS

router = APIRouter()

@router.get("/performance")
def get_performance_report():
    """Get overall performance metrics"""
    try:
        # Get data from trading service
        history = trading_service.get_history(limit=1000)
        
        if not history:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0
            }
        
        # Calculate metrics
        total_trades = len(history)
        wins = [t for t in history if t.get('pnl_percentage', 0) > 0]
        losses = [t for t in history if t.get('pnl_percentage', 0) < 0]
        
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum(t.get('pnl_amount', 0) for t in history)
        
        avg_win = sum(t.get('pnl_amount', 0) for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(t.get('pnl_amount', 0) for t in losses) / len(losses)) if losses else 0
        
        profit_factor = (avg_win * len(wins)) / (avg_loss * len(losses)) if losses else 0
        
        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "wins": len(wins),
            "losses": len(losses)
        }
        
    except Exception as e:
        return {"error": str(e)}

@router.get("/trades")
def get_trades_report(
    symbol: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = Query(100, le=1000)
):
    """Get filtered trade history"""
    try:
        trades = trading_service.get_history(limit=limit)
        
        # Apply filters
        if symbol:
            trades = [t for t in trades if t.get('symbol') == symbol]
        
        if start_date:
            start = datetime.fromisoformat(start_date)
            trades = [t for t in trades if t.get('timestamp', datetime.min) >= start]
        
        if end_date:
            end = datetime.fromisoformat(end_date)
            trades = [t for t in trades if t.get('timestamp', datetime.max) <= end]
        
        return {
            "count": len(trades),
            "trades": trades
        }
        
    except Exception as e:
        return {"error": str(e)}

@router.get("/daily-stats")
def get_daily_stats(days: int = Query(30, le=365)):
    """Get daily statistics"""
    try:
        history = trading_service.get_history(limit=10000)
        
        if not history:
            return {"stats": []}
        
        # Group by date
        daily_data = {}
        
        for trade in history:
            timestamp = trade.get('timestamp')
            if not timestamp:
                continue
            
            date_key = timestamp.date() if isinstance(timestamp, datetime) else datetime.fromisoformat(str(timestamp)).date()
            
            if date_key not in daily_data:
                daily_data[date_key] = {
                    "date": str(date_key),
                    "trades": 0,
                    "pnl": 0.0,
                    "wins": 0,
                    "losses": 0
                }
            
            daily_data[date_key]["trades"] += 1
            pnl = trade.get('pnl_amount', 0)
            daily_data[date_key]["pnl"] += pnl
            
            if pnl > 0:
                daily_data[date_key]["wins"] += 1
            elif pnl < 0:
                daily_data[date_key]["losses"] += 1
        
        # Convert to list and sort
        stats = sorted(daily_data.values(), key=lambda x: x['date'], reverse=True)[:days]
        
        return {"stats": stats}
        
    except Exception as e:
        return {"error": str(e)}
