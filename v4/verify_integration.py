import logging
import sys
import os

# Add parent to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.enhanced_rsi_strategy_v5 import EnhancedRsiStrategyV5
from strategies.ml_regime_detector import MLMarketRegimeDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntegrationVerify")

def verify_strategy_integration():
    logger.info("Initializing EnhancedRsiStrategyV5...")
    strategy = EnhancedRsiStrategyV5()
    
    detector_type = type(strategy.regime_detector).__name__
    logger.info(f"Strategy loaded with Regime Detector: {detector_type}")
    
    if detector_type == "MLMarketRegimeDetector":
        logger.info("✅ SUCCESS: Strategy is using ML-Enhanced Detector")
        if strategy.regime_detector.model_loaded:
             logger.info("✅ ML Model is loaded in the detector")
        else:
             logger.warning("⚠️ ML Model NOT loaded (using fallback within ML class)")
    else:
        logger.error(f"❌ FAILURE: Strategy is using {detector_type}, expected MLMarketRegimeDetector")

if __name__ == "__main__":
    verify_strategy_integration()
