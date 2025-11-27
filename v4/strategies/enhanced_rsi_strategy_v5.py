"""
Enhanced RSI Strategy V5 - Complete Rewrite
Incorporating all improvements from diagnostic analysis report
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import traceback

# Import the enhanced modules created during refactoring
from .mtf_analyzer import EnhancedMTFModule
from .trend_filter import AdvancedTrendFilter
from .market_regime_detector import MarketRegimeDetector
from .risk_manager import DynamicRiskManager
from .contradiction_detector import EnhancedContradictionSystem

logger = logging.getLogger(__name__)

class PositionType(Enum):
    OUT = "OUT"
    LONG = "LONG"
    SHORT = "SHORT"

class ExitReason(Enum):
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    SIGNAL_EXIT = "SIGNAL_EXIT"
    TIME_EXIT = "TIME_EXIT"
    PARTIAL_EXIT = "PARTIAL_EXIT"
    VOLATILITY_EXIT = "VOLATILITY_EXIT"

@dataclass
class TradeEntry:
    price: float
    quantity: float
    time: pd.Timestamp

@dataclass
class Trade:
    entries: List[TradeEntry] = field(default_factory=list)
    position_type: PositionType = PositionType.OUT
    stop_loss: float = 0.0
    take_profit: float = 0.0
    initial_stop_loss: float = 0.0
    trailing_stop: float = 0.0
    highest_profit: float = 0.0  # Track highest profit
    exit_price: Optional[float] = None
    exit_time: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    pnl_percentage: Optional[float] = None
    pnl_amount: Optional[float] = None
    partial_exits: List[Dict] = field(default_factory=list)
    partial_exit_done: bool = False
    entry_conditions: List[str] = field(default_factory=list)
    entry_regime: str = "UNKNOWN"  # Market regime at entry
    contradiction_score: float = 0.0  # Contradiction level at entry

    @property
    def entry_price(self) -> float:
        if not self.entries:
            return 0.0
        total_cost = sum(entry.price * entry.quantity for entry in self.entries)
        total_quantity = sum(entry.quantity for entry in self.entries)
        return total_cost / total_quantity if total_quantity > 0 else 0.0

    @property
    def quantity(self) -> float:
        return sum(entry.quantity for entry in self.entries)

class EnhancedRsiStrategyV5:
    """
    Enhanced RSI Strategy V5 - Complete rewrite addressing all diagnostic issues
    """

    def __init__(
        self,
        # Core RSI - Improved levels
        rsi_period: int = 14,
        rsi_oversold: int = 30,  # Was 35 - More entry opportunities
        rsi_overbought: int = 70,  # Was 65 - More entry opportunities
        rsi_entry_buffer: int = 3,  # Was 5 - Tighter entries

        # Risk Management - Enhanced
        risk_per_trade: float = 0.015,
        stop_loss_atr_multiplier: float = 2.0,
        take_profit_ratio: float = 2.5,
        min_position_size: float = 100,
        max_position_size_ratio: float = 0.4,  # Allow larger positions in favorable conditions

        # Trade Control
        max_trades_per_100: int = 30,
        min_candles_between: int = 5,
        max_trade_duration: int = 100,

        # Filters - Enhanced and active
        enable_trend_filter: bool = True,  # Now enabled with improved logic
        trend_strength_threshold: float = 0.4,
        enable_volume_filter: bool = False,
        enable_volatility_filter: bool = True,  # Now enabled with improved logic
        enable_short_trades: bool = True,

        # Advanced Features
        enable_trailing_stop: bool = True,
        trailing_activation_percent: float = 1.0,
        trailing_stop_atr_multiplier: float = 1.5,
        enable_partial_exit: bool = True,
        partial_exit_ratio: float = 0.5,
        partial_exit_threshold: float = 1.5,

        # Loss Control
        max_consecutive_losses: int = 4,
        pause_after_losses: int = 10,
        risk_reduction_after_loss: bool = False,

        # Confirmations
        require_rsi_confirmation: bool = False,
        require_price_confirmation: bool = False,
        confirmation_candles: int = 1,

        # Multi-Timeframe Analysis (MTF) - Improved
        enable_mtf: bool = True,
        mtf_timeframes: Optional[List[str]] = None,
        mtf_require_all: bool = False,  # Changed from True to False - Any alignment is OK
        mtf_long_rsi_min: float = 40.0,  # Changed from 50.0 - More flexible
        mtf_short_rsi_max: float = 60.0,  # Changed from 50.0 - More flexible
        mtf_trend_ema_fast: int = 21,
        mtf_trend_ema_slow: int = 50,

        # Volatility adaptation for SL
        vol_sl_min_multiplier: float = 1.5,
        vol_sl_high_multiplier: float = 2.5,  # Increased from 2.2
        bb_width_vol_threshold: Optional[float] = 0.012,  # Adjusted threshold

        # Advanced Filters (added parameters from config)
        enable_advanced_filters: bool = True,  # Enable advanced market regime and condition filters
        advanced_filter_confidence_threshold: float = 0.7,  # Minimum confidence for market regime
        market_strength_min_score: float = 3.0,  # Minimum market strength score (0-10)
        support_resistance_check: bool = True,  # Check price against support/resistance levels
        divergence_check: bool = True,  # Check for RSI-price divergence
        volatility_band_check: bool = True,  # Check price position in volatility bands
    ):
        # Initialize all parameters
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.rsi_entry_buffer = rsi_entry_buffer

        self.risk_per_trade = risk_per_trade
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_ratio = take_profit_ratio
        self.min_position_size = min_position_size
        self.max_position_size_ratio = max_position_size_ratio

        self.max_trades_per_100 = max_trades_per_100
        self.min_candles_between = min_candles_between
        self.max_trade_duration = max_trade_duration

        self.enable_trend_filter = enable_trend_filter
        self.trend_strength_threshold = trend_strength_threshold
        self.enable_volume_filter = enable_volume_filter
        self.enable_volatility_filter = enable_volatility_filter
        self.enable_short_trades = enable_short_trades

        self.enable_trailing_stop = enable_trailing_stop
        self.trailing_activation_percent = trailing_activation_percent
        self.trailing_stop_atr_multiplier = trailing_stop_atr_multiplier
        self.enable_partial_exit = enable_partial_exit
        self.partial_exit_ratio = partial_exit_ratio
        self.partial_exit_threshold = partial_exit_threshold

        self.max_consecutive_losses = max_consecutive_losses
        self.pause_after_losses = pause_after_losses
        self.risk_reduction_after_loss = risk_reduction_after_loss

        self.require_rsi_confirmation = require_rsi_confirmation
        self.require_price_confirmation = require_price_confirmation
        self.confirmation_candles = confirmation_candles

        # MTF configuration - IMPROVED
        self.enable_mtf = enable_mtf
        self.mtf_timeframes = mtf_timeframes or ['H4', 'D1']
        self.mtf_require_all = mtf_require_all  # Changed: now False for flexibility
        self.mtf_long_rsi_min = mtf_long_rsi_min  # Changed: more flexible
        self.mtf_short_rsi_max = mtf_short_rsi_max  # Changed: more flexible
        self.mtf_trend_ema_fast = mtf_trend_ema_fast
        self.mtf_trend_ema_slow = mtf_trend_ema_slow

        # Volatility adaptation params
        self.vol_sl_min_multiplier = vol_sl_min_multiplier
        self.vol_sl_high_multiplier = vol_sl_high_multiplier
        self.bb_width_vol_threshold = bb_width_vol_threshold

        # Advanced Filters (added parameters from config)
        self.enable_advanced_filters = enable_advanced_filters
        self.advanced_filter_confidence_threshold = advanced_filter_confidence_threshold
        self.market_strength_min_score = market_strength_min_score
        self.support_resistance_check = support_resistance_check
        self.divergence_check = divergence_check
        self.volatility_band_check = volatility_band_check

        # Initialize ENHANCED module instances
        self.mtf_analyzer = EnhancedMTFModule(
            mtf_timeframes=self.mtf_timeframes,
            mtf_long_rsi_min=self.mtf_long_rsi_min,
            mtf_short_rsi_max=self.mtf_short_rsi_max
        ) if self.enable_mtf else None

        self.trend_filter = AdvancedTrendFilter(
            strength_threshold=self.trend_strength_threshold
        ) if self.enable_trend_filter else None

        self.regime_detector = MarketRegimeDetector()
        self.risk_manager = DynamicRiskManager(
            base_risk_per_trade=self.risk_per_trade,
            max_position_ratio=self.max_position_size_ratio,
            min_position_size=self.min_position_size
        )
        self.contradiction_detector = EnhancedContradictionSystem()

        # State variables
        self._position = PositionType.OUT
        self._current_trade = None
        self._trade_history = []
        self._portfolio_value = 10000.0
        self._last_trade_index = -100
        self._consecutive_losses = 0
        self._pause_until_index = -1
        self._original_risk = risk_per_trade

        # Statistics
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = 0.0
        self._gross_profit = 0.0
        self._gross_loss = 0.0

        # Logs
        self._signal_log = []
        self._current_regime = "UNKNOWN"
        self._filter_scores = {}
        self._contradiction_report = {}

        logger.info(f"🔥 ENHANCED RSI Strategy V5 - RSI({rsi_period}), Risk: {risk_per_trade*100}%")
        logger.info(f"    MTF: {'ON' if self.enable_mtf else 'OFF'}, Trend: {'ON' if self.enable_trend_filter else 'OFF'}")

    def _calculate_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI with error handling"""
        try:
            if len(data) < self.rsi_period + 1:
                logger.warning(f"Insufficient data for RSI calculation: {len(data)} candles")
                return data

            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.ewm(alpha=1/self.rsi_period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/self.rsi_period, adjust=False).mean()

            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))

            # Fill NaN values
            data['RSI'] = data['RSI'].fillna(method='bfill').fillna(50)

            return data

        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            data['RSI'] = 50  # Default value
            return data

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR with error handling"""
        try:
            if len(data) < period + 1:
                logger.warning(f"Insufficient data for ATR: {len(data)} candles")
                return data['close'].iloc[-1] * 0.015

            high = data['high']
            low = data['low']
            close = data['close']

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())

            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean().iloc[-1]

            if pd.isna(atr) or atr <= 0:
                atr = data['close'].iloc[-1] * 0.015

            return atr

        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return data['close'].iloc[-1] * 0.015

    def calculate_position_size(self, data: pd.DataFrame, entry_price: float, stop_loss: float, position_type: PositionType = None) -> float:
        """Calculate position size with dynamic risk management using integrated risk manager"""
        try:
            # Detect current market regime
            regime, conf, details = self.regime_detector.detect_regime(data)

            # Use the enhanced risk manager to calculate position size
            position_size, metrics = self.risk_manager.calculate_position_size(
                data,
                entry_price,
                stop_loss,
                regime_info=details,
                position_type=position_type.value if position_type else "LONG",
                portfolio_value=self._portfolio_value
            )

            # Update our internal tracking
            self._current_regime = regime

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0

    def _detect_signal_contradictions(self, data: pd.DataFrame, position_type: PositionType) -> List[str]:
        """Detect potential signal contradictions using the enhanced system"""
        try:
            # Use the enhanced contradiction detection system
            contradiction_results = self.contradiction_detector.detector.detect_all_contradictions(
                data, position_type.value
            )

            # Store the contradiction report for logging
            self._contradiction_report = contradiction_results

            # Extract contradiction descriptions
            contradictions = []
            for contradition_type in contradiction_results.get('contradiction_types', []):
                contradition_detail = contradiction_results['details'].get(contradition_type.replace('_', ' ').title().replace(' ', '_').lower(), {})
                contradictions.append(f"{contradition_type}: {contradition_detail.get('description', 'Unknown')}")

            return contradictions

        except Exception as e:
            logger.error(f"Error in enhanced contradiction detection: {e}")
            return []  # Return empty list if error occurs

    def check_entry_conditions(self, data: pd.DataFrame, position_type: PositionType) -> Tuple[bool, List[str]]:
        """Enhanced entry conditions with comprehensive validation using integrated modules"""
        conditions = []

        try:
            # Calculate or ensure RSI exists
            if 'RSI' not in data.columns:
                data = self._calculate_rsi(data)

            current_rsi = float(data['RSI'].iloc[-1]) if 'RSI' in data.columns else 50.0

            # Basic RSI check
            if position_type == PositionType.LONG:
                rsi_ok = current_rsi <= (self.rsi_oversold + self.rsi_entry_buffer)
                conditions.append(f"RSI: {current_rsi:.1f} ({'OK' if rsi_ok else 'FAIL'})")
            else:
                rsi_ok = current_rsi >= (self.rsi_overbought - self.rsi_entry_buffer)
                conditions.append(f"RSI: {current_rsi:.1f} ({'OK' if rsi_ok else 'FAIL'})")

            if not rsi_ok:
                return False, [f"RSI not in entry zone for {position_type.value} ({current_rsi:.1f})"]

            # Check for momentum confirmation
            momentum_ok = True
            if len(data) >= 3:
                # Check if price is moving in the expected direction
                if position_type == PositionType.LONG:
                    # Expect recent momentum up (allow slight pullback)
                    recent_move = (data['close'].iloc[-1] - data['close'].iloc[-3]) / data['close'].iloc[-3]
                    momentum_ok = recent_move > -0.002  # Allow slight pullback
                else:
                    # Expect recent momentum down
                    recent_move = (data['close'].iloc[-3] - data['close'].iloc[-1]) / data['close'].iloc[-3]
                    momentum_ok = recent_move > -0.002

            if not momentum_ok:
                conditions.append("Momentum check: Concern over direction")

            # Apply ENHANCED trend filter if enabled
            if self.trend_filter and self.enable_trend_filter:
                trend_ok, trend_desc, trend_conf, _ = self.trend_filter.evaluate_trend(data, position_type.value)
                if not trend_ok:
                    return False, [f"Trend filter: {trend_desc}"]
                conditions.append(f"Trend: {trend_desc}")

            # Apply ENHANCED MTF analysis if enabled
            if self.mtf_analyzer and self.enable_mtf:
                mtf_result = self.mtf_analyzer.analyze_alignment(data, position_type.value)
                mtf_ok = mtf_result['is_aligned']
                mtf_desc = mtf_result['messages'][-1] if mtf_result['messages'] else "MTF analysis error"

                if not mtf_ok:
                    return False, [f"MTF filter: {mtf_desc}"]
                # Include all MTF messages for transparency
                for msg in mtf_result['messages'][:-1]:  # All but the summary
                    conditions.append(f"MTF: {msg}")

            # Apply volatility filter - we'll use the regime detector for now
            # (Could be expanded with a dedicated volatility filter)

            # Check volume confirmation (if available)
            volume_ok = True
            if 'volume' in data.columns and len(data) > 10:
                avg_vol = data['volume'].rolling(10).mean().iloc[-1]
                current_vol = data['volume'].iloc[-1]
                volume_ok = current_vol >= avg_vol * 0.5  # At least 50% of average
                if not volume_ok:
                    conditions.append(f"Volume below avg: {current_vol:.0f} vs {avg_vol:.0f}")

            # Check time from last trade
            candles_since_last = len(data) - 1 - self._last_trade_index
            spacing_ok = candles_since_last >= self.min_candles_between

            if not spacing_ok:
                return False, [f"Insufficient spacing: {candles_since_last} vs {self.min_candles_between} min"]

            conditions.append(f"Trade spacing: {candles_since_last} candles OK")

            # Check if in consecutive loss pause
            if len(data) - 1 <= self._pause_until_index:
                return False, [f"Paused after {self._consecutive_losses} consecutive losses"]

            # Check signal safety using contradiction detector
            regime_info, conf, regime_details = self.regime_detector.detect_regime(data)
            safety_assessment = self.contradiction_detector.analyze_signal_safety(
                data, position_type.value, regime_details
            )

            # Add contradiction information to conditions
            contradiction_score = safety_assessment['contradiction_summary'].get('contradiction_score', 0.0)
            conditions.append(f"Contradictions: {safety_assessment['risk_level']} (score: {contradiction_score:.2f})")

            # Check if we should filter the signal based on contradictions
            should_filter = self.contradiction_detector.should_filter_signal(safety_assessment)
            if should_filter and contradiction_score > 0.3:
                return False, [f"Signal filtered due to contradictions: {safety_assessment['recommendation']}"]

            return True, conditions

        except Exception as e:
            logger.error(f"Error checking entry conditions: {e}")
            logger.error(traceback.format_exc())
            return False, [f"Error in entry conditions: {e}"]

    def calculate_stop_take_profit(self, data: pd.DataFrame, position_type: PositionType, entry_price: float) -> Tuple[float, float]:
        """Calculate Stop Loss and Take Profit with enhanced volatility and regime adaptation using risk manager"""
        try:
            # Get dynamic ATR multiplier from the risk manager
            regime, conf, regime_details = self.regime_detector.detect_regime(data)
            atr_mult, atr_metrics = self.risk_manager.calculate_stop_loss_atr_multiplier(data, regime_details)

            atr = self.calculate_atr(data)
            sl_multiplier = atr_mult  # Use the enhanced multiplier

            # Calculate take profit based on the new SL multiplier
            tp_multiplier = sl_multiplier * self.take_profit_ratio

            if position_type == PositionType.LONG:
                stop_loss = entry_price - (atr * sl_multiplier)
                take_profit = entry_price + (atr * tp_multiplier)
            else:  # SHORT
                stop_loss = entry_price + (atr * sl_multiplier)
                take_profit = entry_price - (atr * tp_multiplier)

            # Validate values
            if position_type == PositionType.LONG:
                if stop_loss >= entry_price:
                    stop_loss = entry_price * 0.99
                if take_profit <= entry_price:
                    take_profit = entry_price * 1.02
            else:
                if stop_loss <= entry_price:
                    stop_loss = entry_price * 1.01
                if take_profit >= entry_price:
                    take_profit = entry_price * 0.98

            logger.info(f"SL: {stop_loss:.4f}, TP: {take_profit:.4f} (ATR: {atr:.6f}, ATR_Mult: {sl_multiplier:.2f}, Regime: {regime})")
            return stop_loss, take_profit

        except Exception as e:
            logger.error(f"Error calculating SL/TP: {e}")
            if position_type == PositionType.LONG:
                return entry_price * 0.99, entry_price * 1.02
            else:
                return entry_price * 1.01, entry_price * 0.98

    def check_exit_conditions(self, data: pd.DataFrame, current_index: int) -> Optional[Dict[str, Any]]:
        """Enhanced exit conditions with improved logic"""
        if self._position == PositionType.OUT or self._current_trade is None:
            return None

        try:
            current_price = data['close'].iloc[-1]
            current_time = data.index[-1]
            entry_price = self._current_trade.entry_price

            # Calculate profit/loss
            if self._position == PositionType.LONG:
                profit_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                profit_pct = ((entry_price - current_price) / entry_price) * 100

            # Track highest profit
            if profit_pct > self._current_trade.highest_profit:
                self._current_trade.highest_profit = profit_pct

            # Partial exit with enhanced conditions
            if (self.enable_partial_exit and
                profit_pct >= self.partial_exit_threshold and
                not self._current_trade.partial_exit_done):

                self._current_trade.partial_exit_done = True
                partial_quantity = self._current_trade.quantity * self.partial_exit_ratio
                partial_pnl = (profit_pct / 100) * partial_quantity * entry_price
                self._portfolio_value += partial_pnl

                # Update trailing stop after partial exit
                if self.enable_trailing_stop:
                    atr = self.calculate_atr(data)
                    if self._position == PositionType.LONG:
                        self._current_trade.trailing_stop = current_price - (atr * self.trailing_stop_atr_multiplier)
                    else:
                        self._current_trade.trailing_stop = current_price + (atr * self.trailing_stop_atr_multiplier)

                logger.info(f"✅ Partial exit: {self.partial_exit_ratio*100}% at profit {profit_pct:.2f}%")

                return {
                    "action": "PARTIAL_EXIT",
                    "price": current_price,
                    "quantity": partial_quantity,
                    "pnl_percentage": profit_pct,
                    "pnl_amount": partial_pnl,
                    "reason": "PARTIAL_TAKE_PROFIT"
                }

            # Stop Loss
            if self._position == PositionType.LONG and current_price <= self._current_trade.stop_loss:
                return self._create_exit_signal("STOP_LOSS", current_price, current_time)
            elif self._position == PositionType.SHORT and current_price >= self._current_trade.stop_loss:
                return self._create_exit_signal("STOP_LOSS", current_price, current_time)

            # Take Profit
            if self._position == PositionType.LONG and current_price >= self._current_trade.take_profit:
                return self._create_exit_signal("TAKE_PROFIT", current_price, current_time)
            elif self._position == PositionType.SHORT and current_price <= self._current_trade.take_profit:
                return self._create_exit_signal("TAKE_PROFIT", current_price, current_time)

            # Enhanced Trailing Stop
            if self.enable_trailing_stop and profit_pct >= self.trailing_activation_percent:
                atr = self.calculate_atr(data)
                trailing_atr = atr * self.trailing_stop_atr_multiplier

                if self._position == PositionType.LONG:
                    new_trailing = current_price - trailing_atr
                    if new_trailing > self._current_trade.trailing_stop:
                        self._current_trade.trailing_stop = new_trailing
                        logger.debug(f"Trailing stop updated: {new_trailing:.4f}")

                    if current_price <= self._current_trade.trailing_stop:
                        return self._create_exit_signal("TRAILING_STOP", current_price, current_time)

                else:  # SHORT
                    new_trailing = current_price + trailing_atr
                    if new_trailing < self._current_trade.trailing_stop or self._current_trade.trailing_stop == self._current_trade.stop_loss:
                        self._current_trade.trailing_stop = new_trailing
                        logger.debug(f"Trailing stop updated: {new_trailing:.4f}")

                    if current_price >= self._current_trade.trailing_stop:
                        return self._create_exit_signal("TRAILING_STOP", current_price, current_time)

            # Time Exit
            trade_duration = current_index - self._last_trade_index
            if trade_duration >= self.max_trade_duration:
                return self._create_exit_signal("TIME_EXIT", current_price, current_time)

            return None

        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            logger.error(traceback.format_exc())
            return None

    def _create_exit_signal(self, exit_reason: str, price: float, time: pd.Timestamp) -> Dict[str, Any]:
        """Create exit signal with comprehensive tracking"""
        try:
            if self._current_trade is None:
                return {"action": "HOLD", "reason": "No active trade"}

            entry_price = self._current_trade.entry_price
            quantity = self._current_trade.quantity

            if self._position == PositionType.LONG:
                pnl_amount = (price - entry_price) * quantity
            else:
                pnl_amount = (entry_price - price) * quantity

            pnl_percentage = (pnl_amount / (entry_price * quantity)) * 100

            # Update portfolio
            self._portfolio_value += pnl_amount
            self._total_pnl += pnl_amount
            self._total_trades += 1

            # Track wins/losses
            if pnl_amount < 0:
                self._consecutive_losses += 1
                self._gross_loss += abs(pnl_amount)
                if self._consecutive_losses >= self.max_consecutive_losses:
                    self._pause_until_index = len(self._trade_history) + self.pause_after_losses
                    logger.warning(f"⚠️ {self._consecutive_losses} consecutive losses - pausing for {self.pause_after_losses} candles")
            else:
                self._consecutive_losses = 0
                self._winning_trades += 1
                self._gross_profit += pnl_amount

            # Record trade with regime information
            self._current_trade.exit_price = price
            self._current_trade.exit_time = time
            self._current_trade.pnl_percentage = pnl_percentage
            self._current_trade.pnl_amount = pnl_amount

            old_position = self._position
            self._trade_history.append(self._current_trade)
            self._position = PositionType.OUT
            self._current_trade = None

            logger.info(f"🔚 EXIT at {price:.4f}, PnL: {pnl_percentage:.2f}%, Reason: {exit_reason}")

            return {
                "action": "EXIT",
                "price": float(price),
                "exit_reason": exit_reason,
                "pnl_percentage": round(pnl_percentage, 2),
                "pnl_amount": round(pnl_amount, 2),
                "position": "OUT",
                "previous_position": old_position.value
            }

        except Exception as e:
            logger.error(f"Error creating exit signal: {e}")
            logger.error(traceback.format_exc())
            return {"action": "HOLD", "reason": f"Exit error: {e}"}

    def generate_signal(self, data: pd.DataFrame, current_index: int = 0) -> Dict[str, Any]:
        """Generate signal with comprehensive market regime detection and enhanced module integration"""
        try:
            # Detect market regime for logging and dynamic adjustments
            regime, conf, regime_details = self.regime_detector.detect_regime(data)
            self._current_regime = regime

            current_time = data.index[-1]
            current_price = data['close'].iloc[-1]
            current_rsi = float(data['RSI'].iloc[-1]) if 'RSI' in data.columns else None

            # Check exit conditions first
            exit_signal = self.check_exit_conditions(data, current_index)
            if exit_signal:
                return exit_signal

            # Basic validation checks
            if len(data) < 50:
                return {"action": "HOLD", "reason": "Insufficient data"}

            if current_index <= self._pause_until_index:
                return {"action": "HOLD", "reason": f"Paused after {self._consecutive_losses} losses"}

            # Check trade frequency limit
            recent_trades = len([t for t in self._trade_history[-100:] if t.exit_time is not None])
            if recent_trades >= self.max_trades_per_100:
                return {"action": "HOLD", "reason": f"Max trades limit ({recent_trades}/{self.max_trades_per_100})"}

            # Check LONG entry
            if self._position == PositionType.OUT:
                long_ok, long_conditions = self.check_entry_conditions(data, PositionType.LONG)
                if long_ok:
                    stop_loss, take_profit = self.calculate_stop_take_profit(data, PositionType.LONG, current_price)
                    position_size = self.calculate_position_size(data, current_price, stop_loss, PositionType.LONG)

                    if position_size > 0:
                        # Get contradiction score for the trade
                        contradiction_score = self._contradiction_report.get('contradiction_score', 0.0)

                        self._position = PositionType.LONG
                        self._current_trade = Trade(
                            entries=[TradeEntry(current_price, position_size, data.index[-1])],
                            position_type=PositionType.LONG,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            initial_stop_loss=stop_loss,
                            trailing_stop=stop_loss,
                            entry_conditions=long_conditions,
                            entry_regime=self._current_regime,
                            contradiction_score=contradiction_score
                        )
                        self._last_trade_index = current_index

                        logger.info(f"🟢 LONG at {current_price:.4f}, Size: {position_size:.0f}, Regime: {self._current_regime}, Contradictions: {contradiction_score:.2f}")

                        return {
                            "action": "BUY",
                            "price": float(current_price),
                            "position_size": float(position_size),
                            "stop_loss": float(stop_loss),
                            "take_profit": float(take_profit),
                            "reason": " | ".join(long_conditions),
                            "position": "LONG",
                            "regime": self._current_regime,
                            "rsi": current_rsi,
                            "contradiction_score": contradiction_score
                        }

            # Check SHORT entry if enabled
            if self._position == PositionType.OUT and self.enable_short_trades:
                short_ok, short_conditions = self.check_entry_conditions(data, PositionType.SHORT)
                if short_ok:
                    stop_loss, take_profit = self.calculate_stop_take_profit(data, PositionType.SHORT, current_price)
                    position_size = self.calculate_position_size(data, current_price, stop_loss, PositionType.SHORT)

                    if position_size > 0:
                        # Get contradiction score for the trade
                        contradiction_score = self._contradiction_report.get('contradiction_score', 0.0)

                        self._position = PositionType.SHORT
                        self._current_trade = Trade(
                            entries=[TradeEntry(current_price, position_size, data.index[-1])],
                            position_type=PositionType.SHORT,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            initial_stop_loss=stop_loss,
                            trailing_stop=stop_loss,
                            entry_conditions=short_conditions,
                            entry_regime=self._current_regime,
                            contradiction_score=contradiction_score
                        )
                        self._last_trade_index = current_index

                        logger.info(f"🔴 SHORT at {current_price:.4f}, Size: {position_size:.0f}, Regime: {self._current_regime}, Contradictions: {contradiction_score:.2f}")

                        return {
                            "action": "SELL",
                            "price": float(current_price),
                            "position_size": float(position_size),
                            "stop_loss": float(stop_loss),
                            "take_profit": float(take_profit),
                            "reason": " | ".join(short_conditions),
                            "position": "SHORT",
                            "regime": self._current_regime,
                            "rsi": current_rsi,
                            "contradiction_score": contradiction_score
                        }

            # Default hold
            return {"action": "HOLD", "reason": f"Waiting for conditions | Regime: {self._current_regime} | Contradictions: {self._contradiction_report.get('contradiction_score', 0):.2f}"}

        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            logger.error(traceback.format_exc())
            return {"action": "HOLD", "reason": f"Error: {e}"}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics with enhanced regime and contradiction tracking"""
        if self._total_trades == 0:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "current_portfolio_value": self._portfolio_value,
                "consecutive_losses": self._consecutive_losses,
                "current_regime": self._current_regime,
                "strategy_version": "V5",
                "avg_contradiction_score": 0.0,
                "regime_performance": {}
            }

        win_rate = (self._winning_trades / self._total_trades) * 100
        profit_factor = self._gross_profit / max(self._gross_loss, 1) if self._gross_loss > 0 else float('inf')

        # Calculate Sharpe ratio if possible
        trade_returns = []
        contradiction_scores = []
        regime_performance = {}

        for t in self._trade_history:
            if t.pnl_percentage is not None:
                trade_returns.append(t.pnl_percentage)
            if hasattr(t, 'contradiction_score'):
                contradiction_scores.append(t.contradiction_score)

            # Track performance by regime
            regime = getattr(t, 'entry_regime', 'UNKNOWN')
            if regime not in regime_performance:
                regime_performance[regime] = {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'pnl': 0,
                    'avg_contradiction': 0
                }

            regime_data = regime_performance[regime]
            regime_data['trades'] += 1
            regime_data['pnl'] += t.pnl_amount or 0

            if t.pnl_amount and t.pnl_amount > 0:
                regime_data['wins'] += 1
            elif t.pnl_amount and t.pnl_amount < 0:
                regime_data['losses'] += 1

            if hasattr(t, 'contradiction_score'):
                regime_data['avg_contradiction'] += t.contradiction_score

        # Calculate regime performance averages
        for regime, data in regime_performance.items():
            if data['trades'] > 0:
                data['win_rate'] = round((data['wins'] / data['trades']) * 100, 2)
                data['avg_pnl'] = round(data['pnl'] / data['trades'], 2)
                if contradiction_scores:  # Only calculate if we have contradiction data
                    data['avg_contradiction'] = round(data['avg_contradiction'] / data['trades'], 3)
            else:
                data['win_rate'] = 0
                data['avg_pnl'] = 0
                data['avg_contradiction'] = 0

        if trade_returns:
            avg_return = np.mean(trade_returns)
            volatility = np.std(trade_returns) if len(trade_returns) > 1 else 0
            sharpe_ratio = avg_return / max(volatility, 0.001) if volatility > 0 else 0
        else:
            sharpe_ratio = 0

        # Calculate average contradiction score
        avg_contradiction = np.mean(contradiction_scores) if contradiction_scores else 0.0

        return {
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "losing_trades": self._total_trades - self._winning_trades,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(self._total_pnl, 2),
            "gross_profit": round(self._gross_profit, 2),
            "gross_loss": round(self._gross_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "current_portfolio_value": round(self._portfolio_value, 2),
            "portfolio_return": round(((self._portfolio_value - 10000) / 10000) * 100, 2),
            "consecutive_losses": self._consecutive_losses,
            "current_position": self._position.value if self._position else "OUT",
            "sharpe_ratio": round(sharpe_ratio, 4),
            "avg_trade_return": round(self._total_pnl / max(self._total_trades, 1), 2),
            "current_regime": self._current_regime,
            "avg_contradiction_score": round(avg_contradiction, 3),
            "trade_quality_insights": {
                "low_contradiction_trades": len([cs for cs in contradiction_scores if cs < 0.3]) if contradiction_scores else 0,
                "high_contradiction_trades": len([cs for cs in contradiction_scores if cs > 0.7]) if contradiction_scores else 0,
            },
            "regime_performance": regime_performance,
            "strategy_version": "V5"
        }

    def get_trade_history(self) -> List[Trade]:
        """Get trade history"""
        return self._trade_history.copy()

    def get_signal_log(self) -> List[Dict[str, Any]]:
        """Get signal log"""
        return self._signal_log.copy()

    def reset_state(self):
        """Reset strategy state"""
        self._position = PositionType.OUT
        self._current_trade = None
        self._trade_history = []
        self._portfolio_value = 10000.0
        self._last_trade_index = -100
        self._consecutive_losses = 0
        self._pause_until_index = -1
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = 0.0
        self._gross_profit = 0.0
        self._gross_loss = 0.0
        self._signal_log = []
        self._current_regime = "UNKNOWN"
        self._filter_scores = {}
        self._original_risk = self.risk_per_trade

        logger.info("✅ Strategy V5 reset completed")