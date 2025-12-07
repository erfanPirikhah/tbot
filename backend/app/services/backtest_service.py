import uuid
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import json
import os

from lib.backtest.enhanced_rsi_backtest_v5 import EnhancedRSIBacktestV5
from lib.strategies.enhanced_rsi_strategy_v5 import EnhancedRsiStrategyV5

logger = logging.getLogger("BacktestService")

# Simple in-memory storage for demo purposes
# In production, use a Database (SQLite/Postgres)
BACKTEST_RESULTS: Dict[str, Any] = {}

class BacktestService:
    @staticmethod
    async def start_backtest(symbol: str, timeframe: str, days: int, strategy_params: Optional[Dict] = None) -> str:
        task_id = str(uuid.uuid4())
        
        # Initialize record
        BACKTEST_RESULTS[task_id] = {
            "task_id": task_id,
            "status": "running",
            "timestamp": datetime.now(),
            "symbol": symbol,
            "timeframe": timeframe
        }
        
        # Run asynchronously (fire and forget for this conceptual demo)
        # In production, use Celery or BackgroundTasks
        asyncio.create_task(BacktestService._run_backtest_logic(task_id, symbol, timeframe, days, strategy_params))
        
        return task_id

    @staticmethod
    async def _run_backtest_logic(task_id: str, symbol: str, timeframe: str, days: int, params: Optional[Dict]):
        try:
            logger.info(f"Starting backtest {task_id} for {symbol}...")
            
            # 1. Initialize Strategy
            # Filter out invalid parameters (like additionalProp1 from Swagger UI)
            valid_params = {}
            if params:
                # Only include known strategy parameters
                known_params = {
                    'rsi_period', 'risk_per_trade', 'rsi_oversold', 'rsi_overbought',
                    'enable_trend_filter', 'enable_mtf', 'test_mode_enabled',
                    'bypass_contradiction_detection', 'relax_risk_filters',
                    'relax_entry_conditions', 'enable_all_signals', 'enable_short_trades'
                }
                valid_params = {k: v for k, v in params.items() if k in known_params}
            
            strategy = EnhancedRsiStrategyV5(**valid_params)
            
            # 2. Run Backtest
            backtester = EnhancedRSIBacktestV5(
                initial_capital=10000.0,
                commission=0.0003,
                slippage=0.0001
            )
            
            # Run backtest with strategy
            results = backtester.run_backtest(
                symbol=symbol,
                timeframe=timeframe,
                days_back=days,
                strategy_params=params or {}
            )
            
            # 3. Store Results
            BACKTEST_RESULTS[task_id]["status"] = "completed"
            BACKTEST_RESULTS[task_id]["metrics"] = results.get("metrics", {})
            
            # Convert equity curve to list for JSON serialization if needed
            if "daily_stats" in results and isinstance(results["daily_stats"], pd.DataFrame):
                 BACKTEST_RESULTS[task_id]["equity_curve"] = results["daily_stats"].reset_index().to_dict(orient="records")
            
            BACKTEST_RESULTS[task_id]["trades"] = results.get("trades", [])
            
            logger.info(f"Backtest {task_id} completed successfully.")
            
        except Exception as e:
            logger.error(f"Backtest {task_id} failed: {e}")
            BACKTEST_RESULTS[task_id]["status"] = "failed"
            BACKTEST_RESULTS[task_id]["error"] = str(e)

    @staticmethod
    def get_result(task_id: str) -> Optional[Dict]:
        return BACKTEST_RESULTS.get(task_id)
