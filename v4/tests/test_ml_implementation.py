import logging
import pandas as pd
import sys
import os

# Add parent to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.ml_regime_detector import MLMarketRegimeDetector
from strategies.market_regime_detector import create_sample_regime_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ML_Test")

def test_ml_integration():
    logger.info("Initializing MLMarketRegimeDetector...")
    
    # Point to the actual trained model
    detector = MLMarketRegimeDetector(model_path="ml_training/regime_model.joblib",
                                     features_path="ml_training/features.json")
    
    logger.info(f"Model loaded: {detector.model_loaded}")
    
    # Test detection (should use fallback)
    data = create_sample_regime_data()
    regime, conf, details = detector.detect_regime(data)
    
    logger.info(f"Detected Regime: {regime}")
    logger.info(f"Confidence: {conf}")
    logger.info(f"Details: {details}")
    
    if not detector.model_loaded and "original_regime" not in details:
        logger.info("✅ SUCCESS: Correctly fell back to legacy logic (no model found).")
    else:
        logger.warning("⚠️ unexpected behavior in fallback.")

if __name__ == "__main__":
    test_ml_integration()
