from fastapi import APIRouter, HTTPException
from app.models.strategy_schema import (
    StrategyConfigRequest, StrategyParametersResponse,
    MLRetrainRequest, MLRetrainResponse
)
from lib.strategies.enhanced_rsi_strategy_v5 import EnhancedRsiStrategyV5
import logging

router = APIRouter()
logger = logging.getLogger("StrategyConfig")

# Global strategy instance for configuration
_strategy_instance = None
_current_params = {}

@router.post("/configure")
def configure_strategy(config: StrategyConfigRequest):
    """Configure strategy parameters"""
    global _strategy_instance, _current_params
    
    try:
        # Convert to dict and filter None values
        params = {k: v for k, v in config.dict().items() if v is not None and k != 'additional_params'}
        
        # Add additional params if provided
        if config.additional_params:
            params.update(config.additional_params)
        
        # Create new strategy instance
        _strategy_instance = EnhancedRsiStrategyV5(**params)
        _current_params = params
        
        logger.info(f"Strategy configured with params: {params}")
        return {"status": "success", "message": "Strategy configured", "params": params}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/parameters", response_model=StrategyParametersResponse)
def get_strategy_parameters():
    """Get current strategy parameters"""
    return {
        "current_params": _current_params,
        "strategy_class": "EnhancedRsiStrategyV5"
    }

@router.post("/ml/retrain", response_model=MLRetrainResponse)
async def retrain_ml_model(request: MLRetrainRequest):
    """Retrain ML model"""
    try:
        # Import training modules
        from lib.ml_training.prepare_data import prepare_dataset
        from lib.ml_training.train_model import train_model
        
        logger.info(f"Starting ML retraining for {request.symbol}")
        
        # Prepare data
        dataset_path = "lib/ml_training/dataset.csv"
        prepare_dataset(
            symbol=request.symbol,
            interval=request.timeframe,
            limit=request.days * 24,  # Convert days to hours for 1h timeframe
            output_path=dataset_path
        )
        
        # Train model
        model_path = "lib/ml_training/regime_model.joblib"
        train_model(
            input_path=dataset_path,
            output_path=model_path
        )
        
        return {
            "status": "success",
            "message": "Model retrained successfully",
            "model_path": model_path
        }
        
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        return {
            "status": "error",
            "message": str(e),
            "model_path": None
        }
