from fastapi import APIRouter
import os
import importlib
from typing import Dict, List, Any

router = APIRouter()

@router.get("/list")
def list_strategies():
    """
    List available strategies in the lib/strategies directory
    """
    strategies_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "lib", "strategies")
    
    strategies = []
    if os.path.exists(strategies_dir):
        for f in os.listdir(strategies_dir):
            if f.endswith(".py") and f != "__init__.py":
                strategies.append(f[:-3])
    
    return {"strategies": strategies}

@router.get("/ml-status")
def get_ml_status():
    """
    Check the status of the ML Model
    """
    try:
        from lib.strategies.ml_regime_detector import MLMarketRegimeDetector
        detector = MLMarketRegimeDetector()
        return {
            "ml_enabled": True,
            "model_loaded": detector.model_loaded,
            "model_type": str(type(detector.model)) if detector.model else "None"
        }
    except ImportError as e:
        return {"ml_enabled": False, "error": f"Import Error: {e}"}
    except Exception as e:
        return {"ml_enabled": False, "error": str(e)}
