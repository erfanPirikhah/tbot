"""
Enhanced Dynamic Risk Management Module
Implements adaptive risk management based on market regime and volatility
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DynamicRiskManager:
    """
    Advanced dynamic risk management system
    """
    
    def __init__(self,
                 base_risk_per_trade: float = 0.015,  # 1.5% base risk
                 max_position_ratio: float = 0.4,     # 40% max of portfolio
                 min_position_size: float = 100,
                 volatility_factor: float = 0.5,      # How much volatility affects risk
                 trend_factor: float = 0.3,           # How much trend affects risk
                 regime_factor: float = 0.2,          # How much regime affects risk
                 max_risk_multiplier: float = 2.0,    # Maximum risk multiplier
                 min_risk_multiplier: float = 0.5):   # Minimum risk multiplier
        """
        Initialize Dynamic Risk Manager
        
        Args:
            base_risk_per_trade: Base risk percentage per trade
            max_position_ratio: Maximum position size as % of portfolio
            min_position_size: Minimum position size in monetary terms
            volatility_factor: Weight of volatility in risk adjustment
            trend_factor: Weight of trend in risk adjustment  
            regime_factor: Weight of market regime in risk adjustment
            max_risk_multiplier: Maximum multiplier for risk
            min_risk_multiplier: Minimum multiplier for risk
        """
        self.base_risk_per_trade = base_risk_per_trade
        self.max_position_ratio = max_position_ratio
        self.min_position_size = min_position_size
        self.volatility_factor = volatility_factor
        self.trend_factor = trend_factor
        self.regime_factor = regime_factor
        self.max_risk_multiplier = max_risk_multiplier
        self.min_risk_multiplier = min_risk_multiplier
    
    def calculate_volatility_adjusted_risk(self, data: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """
        Calculate risk adjustment based on volatility
        """
        try:
            # Calculate recent volatility
            returns = data['close'].pct_change().tail(20).dropna()
            if len(returns) == 0:
                return 1.0, {"volatility_multiplier": 1.0, "current_volatility": 0.0}
            
            current_vol = returns.std()
            
            # Calculate historical volatility context
            historical_vols = data['close'].pct_change().rolling(50).std().dropna()
            if len(historical_vols) == 0:
                return 1.0, {"volatility_multiplier": 1.0, "current_volatility": current_vol}
            
            avg_vol = historical_vols.mean()
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
            
            # Calculate volatility percentile
            vol_percentile = 0.5
            if len(historical_vols) > 1:
                vol_percentile = (historical_vols <= current_vol).mean()
            
            # Adjust risk based on volatility
            if vol_ratio > 1.5:  # Much higher than normal
                volatility_multiplier = max(self.min_risk_multiplier, 1.0 / vol_ratio)
            elif vol_ratio < 0.7:  # Much lower than normal
                volatility_multiplier = min(self.max_risk_multiplier, 1.0 + (1 - vol_ratio) * 0.5)
            else:  # Normal range
                volatility_multiplier = 1.0 - (1 - vol_ratio) * 0.2  # Slight adjustment
            
            metrics = {
                "volatility_multiplier": volatility_multiplier,
                "current_volatility": current_vol,
                "average_volatility": avg_vol,
                "volatility_ratio": vol_ratio,
                "volatility_percentile": vol_percentile
            }
            
            return volatility_multiplier, metrics
            
        except Exception as e:
            logger.error(f"Error in volatility adjusted risk calculation: {e}")
            return 1.0, {"volatility_multiplier": 1.0, "error": str(e)}

    def calculate_trend_adjusted_risk(self, data: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """
        Calculate risk adjustment based on trend alignment
        """
        try:
            if len(data) < 50:
                return 1.0, {"trend_multiplier": 1.0, "trend_strength": 0.0}
            
            close = data['close']
            
            # Calculate EMAs for trend analysis
            ema_fast = close.ewm(span=8).mean()
            ema_medium = close.ewm(span=21).mean()
            ema_slow = close.ewm(span=50).mean()
            
            # Calculate trend strength
            fast_over_slow = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / close.iloc[-1]
            position_in_trend = (close.iloc[-1] - ema_slow.iloc[-1]) / close.iloc[-1]
            
            # Trend strength score (0-1)
            trend_strength = abs(fast_over_slow)
            
            # Calculate trend consistency
            recent_trend = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] if len(close) >= 5 else 0
            trend_alignment = np.sign(recent_trend) == np.sign(fast_over_slow)
            
            # Adjust risk based on trend
            if trend_strength > 0.01 and trend_alignment:  # Strong aligned trend
                trend_multiplier = min(self.max_risk_multiplier, 1.0 + trend_strength * 0.5)
            elif trend_strength > 0.01 and not trend_alignment:  # Strong but misaligned
                trend_multiplier = max(self.min_risk_multiplier, 1.0 - trend_strength * 0.3)
            elif trend_strength < 0.005:  # Weak trend (ranging market)
                trend_multiplier = max(self.min_risk_multiplier, 0.8)  # Reduce risk in ranging
            else:  # Moderate trend
                trend_multiplier = 1.0 + (trend_strength - 0.005) * 0.2
            
            metrics = {
                "trend_multiplier": trend_multiplier,
                "trend_strength": trend_strength,
                "fast_over_slow": fast_over_slow,
                "position_in_trend": position_in_trend,
                "trend_aligned": trend_alignment,
                "recent_trend": recent_trend
            }
            
            return trend_multiplier, metrics
            
        except Exception as e:
            logger.error(f"Error in trend adjusted risk calculation: {e}")
            return 1.0, {"trend_multiplier": 1.0, "error": str(e)}

    def calculate_regime_adjusted_risk(self, regime_info: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate risk adjustment based on market regime
        """
        try:
            regime = regime_info.get('final_regime', 'NORMAL')
            confidence = regime_info.get('overall_confidence', 0.5)
            
            # Risk multipliers by regime
            regime_multipliers = {
                'VOLATILE': 0.7,      # Reduce risk in volatile markets
                'TRENDING': 1.2,      # Increase risk in trending markets
                'RANGING': 0.8,       # Reduce risk in ranging markets
                'MOMENTUM': 1.1,      # Slightly increase in momentum
                'NORMAL': 1.0,        # Normal risk
                'STRONG_TREND': 1.3,  # Higher risk in strong trends
                'HIGH_VOLATILITY': 0.6,  # Much lower risk in high volatility
                'TRANSITION': 0.9     # Slightly reduced in transition
            }
            
            base_multiplier = regime_multipliers.get(regime, 1.0)
            
            # Adjust based on confidence
            confidence_adjustment = 1.0 + (confidence - 0.5) * 0.2  # ±10% based on confidence
            regime_multiplier = base_multiplier * confidence_adjustment
            
            # Clamp to allowed range
            regime_multiplier = max(self.min_risk_multiplier, 
                                  min(self.max_risk_multiplier, regime_multiplier))
            
            metrics = {
                "regime_multiplier": regime_multiplier,
                "regime": regime,
                "confidence": confidence,
                "base_multiplier": base_multiplier,
                "confidence_adjustment": confidence_adjustment
            }
            
            return regime_multiplier, metrics
            
        except Exception as e:
            logger.error(f"Error in regime adjusted risk calculation: {e}")
            return 1.0, {"regime_multiplier": 1.0, "error": str(e)}

    def calculate_dynamic_risk(self, 
                             data: pd.DataFrame, 
                             regime_info: Optional[Dict[str, Any]] = None,
                             position_type: str = "LONG") -> Tuple[float, Dict[str, Any]]:
        """
        Calculate overall dynamic risk based on all factors
        
        Args:
            data: Market data
            regime_info: Regime information from MarketRegimeDetector
            position_type: 'LONG' or 'SHORT'
            
        Returns:
            Tuple of (adjusted_risk_percentage, detailed_metrics)
        """
        try:
            # Calculate individual risk adjustments
            vol_multiplier, vol_metrics = self.calculate_volatility_adjusted_risk(data)
            trend_multiplier, trend_metrics = self.calculate_trend_adjusted_risk(data)
            
            # Calculate regime multiplier
            if regime_info:
                regime_multiplier, regime_metrics = self.calculate_regime_adjusted_risk(regime_info)
            else:
                regime_multiplier = 1.0
                regime_metrics = {"regime_multiplier": 1.0, "regime": "UNKNOWN"}
            
            # Combine adjustments using weighted factors
            combined_multiplier = (
                vol_multiplier * self.volatility_factor +
                trend_multiplier * self.trend_factor +
                regime_multiplier * self.regime_factor
            ) / (self.volatility_factor + self.trend_factor + self.regime_factor)
            
            # Clamp to allowed range
            combined_multiplier = max(self.min_risk_multiplier, 
                                   min(self.max_risk_multiplier, combined_multiplier))
            
            # Calculate final risk
            adjusted_risk = self.base_risk_per_trade * combined_multiplier
            
            # Position-specific adjustments (for short positions)
            if position_type == "SHORT":
                # Slightly adjust for short position risk (optional)
                adjusted_risk = adjusted_risk * 0.95  # Small reduction for short risk
            
            metrics = {
                "base_risk": self.base_risk_per_trade,
                "volatility_factor": self.volatility_factor,
                "trend_factor": self.trend_factor,
                "regime_factor": self.regime_factor,
                "volatility_adjusted": vol_multiplier,
                "trend_adjusted": trend_multiplier,
                "regime_adjusted": regime_multiplier,
                "combined_multiplier": combined_multiplier,
                "final_risk_percentage": adjusted_risk,
                "position_type": position_type,
                "volatility_metrics": vol_metrics,
                "trend_metrics": trend_metrics,
                "regime_metrics": regime_metrics
            }
            
            logger.info(f"Dynamic Risk: {adjusted_risk:.4f} ({combined_multiplier:.3f}x) - Regime: {regime_metrics.get('regime', 'UNKNOWN')}")
            
            return adjusted_risk, metrics
            
        except Exception as e:
            logger.error(f"Error in dynamic risk calculation: {e}")
            return self.base_risk_per_trade, {"error": str(e), "final_risk_percentage": self.base_risk_per_trade}

    def calculate_position_size(self, 
                              data: pd.DataFrame,
                              entry_price: float, 
                              stop_loss: float,
                              regime_info: Optional[Dict[str, Any]] = None,
                              position_type: str = "LONG",
                              portfolio_value: float = 10000.0) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate position size based on dynamic risk
        
        Args:
            data: Market data
            entry_price: Entry price for the position
            stop_loss: Stop loss price
            regime_info: Regime information
            position_type: 'LONG' or 'SHORT'
            portfolio_value: Current portfolio value
            
        Returns:
            Tuple of (position_size, detailed_metrics)
        """
        try:
            # Calculate dynamic risk
            risk_percentage, risk_metrics = self.calculate_dynamic_risk(data, regime_info, position_type)
            
            # Calculate risk amount
            risk_amount = portfolio_value * risk_percentage
            
            # Calculate price risk (distance to stop loss)
            if position_type == "LONG":
                price_risk = abs(entry_price - stop_loss)
            else:  # SHORT
                price_risk = abs(stop_loss - entry_price)
            
            # Validate price risk
            if price_risk <= 0 or price_risk > entry_price * 0.1:  # Max 10% stop
                logger.warning(f"Invalid price risk: {price_risk}, using default")
                price_risk = entry_price * 0.008  # Default 0.8% stop
            
            # Calculate position size
            position_size = risk_amount / price_risk if price_risk > 0 else 0
            
            # Apply maximum position size limit
            max_position = portfolio_value * self.max_position_ratio
            position_size = min(position_size, max_position)
            
            # Apply minimum position size
            if position_size < self.min_position_size:
                position_size = 0  # Don't take the trade
            
            # Calculate additional metrics
            expected_return = abs(entry_price - data['close'].iloc[-1])  # Expected movement
            risk_reward_ratio = expected_return / price_risk if price_risk > 0 else 0
            
            metrics = {
                "position_size": position_size,
                "risk_percentage": risk_percentage,
                "risk_amount": risk_amount,
                "price_risk": price_risk,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "max_position_ratio": self.max_position_ratio,
                "max_position_size": max_position,
                "min_position_size": self.min_position_size,
                "risk_reward_ratio": risk_reward_ratio,
                "portfolio_value": portfolio_value,
                "risk_metrics": risk_metrics
            }
            
            logger.info(f"Position size: {position_size:.0f} (Risk: {risk_percentage*100:.2f}%, RR: {risk_reward_ratio:.2f})")
            
            return position_size, metrics
            
        except Exception as e:
            logger.error(f"Error in position size calculation: {e}")
            return 0, {"error": str(e), "position_size": 0}

    def calculate_stop_loss_atr_multiplier(self, 
                                         data: pd.DataFrame, 
                                         regime_info: Optional[Dict[str, Any]] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate dynamic ATR multiplier for stop losses based on market conditions
        """
        try:
            # Calculate ATR
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1] if len(true_range) >= 14 else close.iloc[-1] * 0.01
            
            # Base multiplier
            base_multiplier = 2.0  # Default
            
            # Adjust based on volatility
            returns = data['close'].pct_change().tail(20).dropna()
            current_vol = returns.std() if len(returns) > 0 else 0.01
            
            vol_adjustment = 1.0
            if current_vol > 0.02:  # High volatility
                vol_adjustment = 1.5
            elif current_vol < 0.005:  # Low volatility
                vol_adjustment = 1.2
            
            # Adjust based on regime if provided
            regime_adjustment = 1.0
            if regime_info:
                regime = regime_info.get('final_regime', 'NORMAL')
                regime_multipliers = {
                    'VOLATILE': 1.8,
                    'TRENDING': 1.5,
                    'RANGING': 2.2,
                    'HIGH_VOLATILITY': 2.0,
                    'STRONG_TREND': 1.3
                }
                regime_adjustment = regime_multipliers.get(regime, 1.0)
            
            # Calculate final multiplier
            final_multiplier = base_multiplier * vol_adjustment * regime_adjustment
            final_multiplier = max(1.2, min(3.0, final_multiplier))  # Clamp between 1.2 and 3.0
            
            metrics = {
                "atr": atr,
                "base_multiplier": base_multiplier,
                "volatility_adjustment": vol_adjustment,
                "regime_adjustment": regime_adjustment,
                "final_multiplier": final_multiplier,
                "adjusted_for_regime": regime_info is not None
            }
            
            return final_multiplier, metrics
            
        except Exception as e:
            logger.error(f"Error in ATR multiplier calculation: {e}")
            return 2.0, {"final_multiplier": 2.0, "error": str(e)}

class RiskAdjustmentSystem:
    """
    Comprehensive system for risk-based adjustments
    """
    
    def __init__(self):
        self.risk_manager = DynamicRiskManager()
        self.trade_history = []
    
    def should_take_trade(self, 
                         data: pd.DataFrame,
                         regime_info: Optional[Dict[str, Any]] = None,
                         position_type: str = "LONG") -> Tuple[bool, str, Dict[str, float]]:
        """
        Determine if a trade should be taken based on risk factors
        """
        try:
            # Calculate dynamic risk
            risk_pct, risk_metrics = self.risk_manager.calculate_dynamic_risk(data, regime_info, position_type)
            
            # Calculate ATR multiplier
            atr_mult, atr_metrics = self.risk_manager.calculate_stop_loss_atr_multiplier(data, regime_info)
            
            # Decision factors
            max_risk_threshold = 0.03  # Max 3% risk
            min_risk_threshold = 0.005  # Min 0.5% risk (to avoid extremely low risk scenarios)
            vol_score = risk_metrics.get('volatility_metrics', {}).get('volatility_percentile', 0.5)
            
            # Make decision
            is_suitable = (
                risk_pct <= max_risk_threshold and 
                risk_pct >= min_risk_threshold and 
                vol_score < 0.9  # Not in top 10% of volatility
            )
            
            if not is_suitable:
                reason = []
                if risk_pct > max_risk_threshold:
                    reason.append(f"Risk too high: {risk_pct:.3f}")
                if risk_pct < min_risk_threshold:
                    reason.append(f"Risk too low: {risk_pct:.3f}")
                if vol_score >= 0.9:
                    reason.append(f"Too volatile: {vol_score:.2f}")
                decision_reason = " | ".join(reason)
            else:
                decision_reason = "Risk parameters acceptable"
            
            metrics = {
                "risk_percentage": risk_pct,
                "atr_multiplier": atr_mult,
                "volatility_score": vol_score,
                "is_suitable": is_suitable,
                "reason": decision_reason
            }
            
            return is_suitable, decision_reason, metrics
            
        except Exception as e:
            logger.error(f"Error in trade suitability check: {e}")
            return False, f"Error in risk assessment: {e}", {"error": str(e)}

def create_sample_risk_data() -> pd.DataFrame:
    """Create sample data for testing risk management"""
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    np.random.seed(42)
    
    # Create different market conditions
    prices = []
    trend = 100
    
    for i in range(200):
        if i < 50:  # Stable
            change = np.random.randn() * 0.3
        elif i < 100:  # Trending
            change = 0.5 + np.random.randn() * 0.4
        elif i < 150:  # High volatility
            change = np.random.randn() * 0.8
        else:  # Ranging
            change = np.random.randn() * 0.2
            
        trend += change
        prices.append(max(10, trend))  # Prevent negative prices
    
    data = pd.DataFrame({
        'close': prices,
        'high': [p + abs(np.random.randn() * 0.4) for p in prices],
        'low': [p - abs(np.random.randn() * 0.4) for p in prices],
        'open': [p + np.random.randn() * 0.1 for p in prices],
    }, index=dates)
    
    return data

def test_risk_management():
    """Test the risk management functionality"""
    logger.info("Testing Enhanced Dynamic Risk Management...")
    
    # Create sample data
    data = create_sample_risk_data()
    
    # Initialize risk manager
    risk_manager = DynamicRiskManager(
        base_risk_per_trade=0.015,
        volatility_factor=0.4,
        trend_factor=0.4,
        regime_factor=0.2
    )
    
    # Test with recent data
    recent_data = data.tail(100)
    
    # Calculate dynamic risk
    risk_pct, risk_metrics = risk_manager.calculate_dynamic_risk(recent_data)
    logger.info(f"Dynamic Risk: {risk_pct:.4f}")
    logger.info(f"Risk Metrics: {risk_metrics}")
    
    # Calculate position size
    entry_price = recent_data['close'].iloc[-1]
    stop_loss = entry_price * 0.98  # 2% stop for LONG
    position_size, pos_metrics = risk_manager.calculate_position_size(
        recent_data, entry_price, stop_loss, position_type="LONG"
    )
    logger.info(f"Position Size: {position_size:.0f}")
    logger.info(f"Position Metrics: {pos_metrics}")
    
    # Test ATR multiplier calculation
    atr_mult, atr_metrics = risk_manager.calculate_stop_loss_atr_multiplier(recent_data)
    logger.info(f"ATR Multiplier: {atr_mult:.2f}")
    logger.info(f"ATR Metrics: {atr_metrics}")
    
    # Test risk adjustment system
    risk_system = RiskAdjustmentSystem()
    is_suitable, reason, suitability_metrics = risk_system.should_take_trade(recent_data)
    logger.info(f"Trade Suitable: {is_suitable}, Reason: {reason}")
    logger.info(f"Suitability Metrics: {suitability_metrics}")
    
    # Test with different market regimes
    regime_info = {
        'final_regime': 'VOLATILE',
        'overall_confidence': 0.8
    }
    
    volatile_risk, volatile_metrics = risk_manager.calculate_dynamic_risk(recent_data, regime_info)
    logger.info(f"Volatility Regime Risk: {volatile_risk:.4f}")
    
    regime_info['final_regime'] = 'TRENDING'
    trending_risk, trending_metrics = risk_manager.calculate_dynamic_risk(recent_data, regime_info)
    logger.info(f"Trending Regime Risk: {trending_risk:.4f}")
    
    logger.info("Risk Management testing completed successfully!")

if __name__ == "__main__":
    test_risk_management()