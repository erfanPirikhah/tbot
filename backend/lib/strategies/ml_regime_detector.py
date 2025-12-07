import joblib
import pandas as pd
import numpy as np
import logging
import os
import json
from typing import Tuple, Dict, Any

# Fix import path - try relative import first, then absolute
try:
    from .market_regime_detector import MarketRegimeDetector
except ImportError:
    from strategies.market_regime_detector import MarketRegimeDetector

logger = logging.getLogger(__name__)

# Get the base directory dynamically
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_MODEL_PATH = os.path.join(_BASE_DIR, "ml_training", "regime_model.joblib")
_DEFAULT_FEATURES_PATH = os.path.join(_BASE_DIR, "ml_training", "features.json")

class MLMarketRegimeDetector(MarketRegimeDetector):
    """
    ML-based Market Regime Detector using a pre-trained XGBoost/RandomForest model.
    Replaces the heuristic rules with probability-based classification.
    """
    
    def __init__(self, 
                 model_path: str = None,
                 features_path: str = None):
        super().__init__()
        self.model = None
        self.feature_names = []
        self.model_loaded = False
        
        # Use dynamic paths if not provided
        if model_path is None:
            model_path = _DEFAULT_MODEL_PATH
        if features_path is None:
            features_path = _DEFAULT_FEATURES_PATH
        
        self._load_model(model_path, features_path)
        
    def _load_model(self, model_path: str, features_path: str):
        """Load the trained model and feature list"""
        try:
            if os.path.exists(model_path) and os.path.exists(features_path):
                self.model = joblib.load(model_path)
                with open(features_path, 'r') as f:
                    self.feature_names = json.load(f)
                self.model_loaded = True
                logger.info(f"✅ ML Regime Model loaded from {model_path}")
            else:
                logger.warning(f"⚠️ ML Model not found at {model_path}. Falling back to heuristic mode.")
                self.model_loaded = False
        except Exception as e:
            logger.error(f"❌ Error loading ML model: {e}")
            self.model_loaded = False

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate the exact same features used in training.
        Must match FeatureEngineer.add_technical_indicators logic.
        """
        data = df.copy()
        close = data['close']
        high = data['high']
        low = data['low']
        
        # 1. Trend Indicators
        data['sma_50'] = close.rolling(window=50).mean()
        data['sma_200'] = close.rolling(window=200).mean()
        data['sma_50_200_ratio'] = data['sma_50'] / data['sma_200']
        data['price_to_sma_50'] = close / data['sma_50']
        
        data['ema_9'] = close.ewm(span=9, adjust=False).mean()
        data['ema_21'] = close.ewm(span=21, adjust=False).mean()
        data['ema_9_21_ratio'] = data['ema_9'] / data['ema_21']
        
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        data['macd'] = exp12 - exp26
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']
        
        # 2. Momentum Indicators
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        low_min = low.rolling(window=14).min()
        high_max = high.rolling(window=14).max()
        data['stoch_k'] = 100 * ((close - low_min) / (high_max - low_min))
        data['stoch_d'] = data['stoch_k'].rolling(window=3).mean()
        
        # 3. Volatility Indicators
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data['atr'] = tr.rolling(window=14).mean()
        data['atr_ratio'] = data['atr'] / close
        
        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        data['bb_upper'] = sma_20 + (std_20 * 2)
        data['bb_lower'] = sma_20 - (std_20 * 2)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / sma_20
        data['bb_position'] = (close - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # 4. Price Action
        data['body_size'] = abs(close - data['open']) / close
        data['shadow_upper'] = (high - data[['open', 'close']].max(axis=1)) / close
        data['shadow_lower'] = (data[['open', 'close']].min(axis=1) - low) / close
        data['close_change_1'] = close.pct_change(1)
        data['close_change_3'] = close.pct_change(3)
        data['close_change_5'] = close.pct_change(5)
        
        return data

    def detect_regime(self,
                     data: pd.DataFrame,
                     test_mode_enabled: bool = False) -> Tuple[str, float, Dict[str, Any]]:
        """
        Detect regime using ML model.
        Returns: (RegimeName, Confidence, Details)
        """
        # Fallback to legacy rule-based if model not loaded
        if not self.model_loaded:
            return super().detect_regime(data, test_mode_enabled)
            
        try:
            # Need at least 200 candles for SMA 200
            if len(data) < 200:
                logger.warning("Insufficient data for ML features (requires 200+ candles). Using fallback.")
                return super().detect_regime(data, test_mode_enabled)

            # 1. Generate Features
            df_features = self.generate_features(data)
            
            # Get last row (current moment)
            current_features = df_features.iloc[[-1]] 
            
            # Ensure columns match training exactly
            try:
                X = current_features[self.feature_names]
            except KeyError as e:
                logger.error(f"Feature mismatch: {e}. Falling back.")
                return super().detect_regime(data, test_mode_enabled)
                
            # 2. Predict Probability
            # probability of class 1 (Win)
            prob_win = self.model.predict_proba(X)[0][1]
            
            # 3. Determine Threshold
            # In TestMode, we might be more lenient
            threshold = 0.55
            if test_mode_enabled:
                threshold = 0.45
                
            # 4. Classify Regime
            confidence = prob_win
            
            if prob_win > threshold:
                # Map high probability of win to "FAVORABLE"
                # To be compatible with existing logic that expects TRENDING/VOLATILE/RANGING:
                # We can map to 'TRENDING' if we want to signal "Go", or introduce specific ML regime.
                # Let's map to "FAVORABLE" and handle it in RiskManager or return specific existing tags.
                # Currently: returning 'TRENDING' as a proxy for "Good to trade"
                final_regime = "TRENDING" 
                
                # Boost confidence if it's very high
                if prob_win > 0.7:
                     final_regime = "STRONG_TREND"
            else:
                # Low probability of win
                final_regime = "RANGING" # Or "CHOPPY" / "UNFAVORABLE"
                confidence = 1.0 - prob_win # Confidence in it being bad
                
            details = {
                "ml_probability": prob_win,
                "ml_threshold": threshold,
                "model_type": "XGBoost" if "XGB" in str(type(self.model)) else "RandomForest",
                "test_mode": test_mode_enabled,
                "original_regime": "ML_OVERRIDE"
            }
            
            return final_regime, prob_win, details

        except Exception as e:
            logger.error(f"Error in ML regime detection: {e}")
            return super().detect_regime(data, test_mode_enabled)
