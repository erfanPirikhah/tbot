from fastapi import APIRouter
from app.api.endpoints import system

api_router = APIRouter()

api_router.include_router(system.router, prefix="/system", tags=["system"])
from app.api.endpoints import strategies
api_router.include_router(strategies.router, prefix="/strategies", tags=["strategies"])

from app.api.endpoints import backtest
api_router.include_router(backtest.router, prefix="/backtest", tags=["backtest"])

from app.api.endpoints import market
api_router.include_router(market.router, prefix="/market", tags=["market"])

from app.api.endpoints import trading
api_router.include_router(trading.router, prefix="/trading", tags=["trading"])

from app.api.endpoints import reports
api_router.include_router(reports.router, prefix="/reports", tags=["reports"])

from app.api.endpoints import config
api_router.include_router(config.router, prefix="/config", tags=["config"])
