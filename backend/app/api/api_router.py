from fastapi import APIRouter
from app.api.endpoints import system

api_router = APIRouter()

api_router.include_router(system.router, prefix="/system", tags=["system"])
from app.api.endpoints import strategies
api_router.include_router(strategies.router, prefix="/strategies", tags=["strategies"])
