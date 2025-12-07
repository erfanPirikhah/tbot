from typing import Dict, Any
import pandas as pd
import logging
from lib.data.data_fetcher import DataFetcher
from lib.strategies.ml_regime_detector import MLMarketRegimeDetector

logger = logging.getLogger("MarketService")

class MarketAnalysisService:
    def __init__(self):
        # Initialize components once
        self.fetcher = DataFetcher()
        self.regime_detector = MLMarketRegimeDetector()

    def analyze_market(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Fetch latest data and analyze regime
        """
        try:
            # 1. Fetch Data (Enough for analysis)
            data = self.fetcher.fetch_market_data(symbol, timeframe, limit=300)
            
            if data.empty:
                return {
                    "symbol": symbol,
                    "error": "No data available",
                    "price": 0.0
                }
            
            current_price = data['close'].iloc[-1]
            
            # 2. Detect Regime
            regime, conf, details = self.regime_detector.detect_regime(data)
            
            # 3. Determine Bullish/Bearish based on regime
            # Simplified Logic:
            # TRENDING + ML Prob > 0.55 => Bullish (if we assume ML predicts long success)
            # Actually, ML Predicts "Success", which usually means Long success in this context.
            
            market_state = "NEUTRAL"
            if regime in ["TRENDING", "STRONG_TREND", "FAVORABLE"]:
                 market_state = "BULLISH" # Assuming model trained on Longs
            elif regime in ["RANGING", "UNFAVORABLE"]:
                 market_state = "NEUTRAL" # or Range
            
            # Heuristic check for Bearish if price is dropping fast
            # (Override or component)
            # For now, let's trust the regime, but add a simple trend direction check
            sma50 = data['close'].rolling(50).mean().iloc[-1]
            if current_price < sma50 and market_state == "BULLISH":
                market_state = "WEAK_BULLISH" # Conflict
            elif current_price < sma50 and market_state == "NEUTRAL":
                market_state = "BEARISH"

            return {
                "symbol": symbol,
                "price": current_price,
                "market_state": market_state,
                "regime": regime,
                "confidence": conf,
                "details": details,
                "timestamp": str(data.index[-1])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}

market_service = MarketAnalysisService()
