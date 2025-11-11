# strategies/ensemble_strategy_v4.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime

# Reuse core constructs from EnhancedRsiStrategyV4 to keep behavior consistent
from .enhanced_rsi_strategy_v4 import (
    TradeEntry,
    Trade,
    PositionType,
    ExitReason,
)

# Import AdvancedMarketFilters
from .advanced_filters import AdvancedMarketFilters

logger = logging.getLogger(__name__)


class EnsembleRsiStrategyV4:
    """
    Ensemble strategy for M5/M15 profitability:
    - Combines multiple micro-algorithms (signals) and votes using weighted scores
    - Components:
      1) Mean Reversion (RSI + Bollinger touch, low ADX)
      2) Trend Pullback (EMA21/EMA50 alignment + RSI momentum)
      3) Breakout (Donchian/Keltner/BB expansion + ADX rising)
      4) MACD + RSI momentum (Histogram turn aligned with RSI midline)
    - Signal selection: weighted sum for LONG vs SHORT; if one exceeds threshold and dominates, enter.
    """

    def __init__(
        self,
        # Core RSI baseline for filters
        rsi_period: int = 14,
        rsi_oversold: int = 35,
        rsi_overbought: int = 65,
        rsi_entry_buffer: int = 5,

        # Risk Management
        risk_per_trade: float = 0.015,
        stop_loss_atr_multiplier: float = 1.6,
        take_profit_ratio: float = 1.8,
        min_position_size: float = 100.0,
        max_position_size_ratio: float = 0.35,

        # Trade Control
        max_trades_per_100: int = 60,   # scalping-friendly
        min_candles_between: int = 3,   # allow more frequent entries on M5/M15
        max_trade_duration: int = 50,   # faster cycles for intraday

        # Advanced exits
        enable_trailing_stop: bool = True,
        trailing_activation_percent: float = 0.6,
        trailing_stop_atr_multiplier: float = 1.2,
        enable_partial_exit: bool = True,
        partial_exit_ratio: float = 0.5,
        partial_exit_threshold: float = 1.0,

        # Loss control
        max_consecutive_losses: int = 5,
        pause_after_losses: int = 10,
        risk_reduction_after_loss: bool = False,

        # Short trades
        enable_short_trades: bool = True,

        # Ensemble Weights
        w_mean_reversion: float = 0.9,
        w_trend_pullback: float = 1.0,
        w_breakout: float = 1.1,
        w_macd_rsi: float = 0.8,

        # Entry thresholding
        entry_threshold: float = 1.6,  # weighted score threshold
        dominance_ratio: float = 1.15, # winning side must exceed the other by 15%

        # Scalping/time filter (optional)
        session_filter_enabled: bool = True,
        session_hours: Optional[List[Tuple[int, int]]] = None,  # e.g., [(8,12),(13,18)] UTC or data tz
        session_timezone_offset: int = 0,  # adjust hours to data timezone

        # Optional guards
        bb_width_min: Optional[float] = 0.001,  # avoid dead markets
        bb_width_max: Optional[float] = 0.06,   # avoid extreme spikes,

        # Volatility adaptation (new)
        vol_sl_min_multiplier: float = 1.5,     # enforce at least 1.5x ATR for SL
        vol_sl_high_multiplier: float = 2.2,    # widen SL in volatile regimes
        bb_width_vol_threshold: Optional[float] = 0.015,  # BB width threshold to detect volatility

        # Advanced Filters (new addition)
        enable_advanced_filters: bool = True,
        advanced_filter_confidence_threshold: float = 0.7,
        market_strength_min_score: float = 3.0,
        support_resistance_check: bool = True,
        divergence_check: bool = True,
        volatility_band_check: bool = True,
    ):
        # Parameters
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

        self.enable_trailing_stop = enable_trailing_stop
        self.trailing_activation_percent = trailing_activation_percent
        self.trailing_stop_atr_multiplier = trailing_stop_atr_multiplier
        self.enable_partial_exit = enable_partial_exit
        self.partial_exit_ratio = partial_exit_ratio
        self.partial_exit_threshold = partial_exit_threshold

        self.max_consecutive_losses = max_consecutive_losses
        self.pause_after_losses = pause_after_losses
        self.risk_reduction_after_loss = risk_reduction_after_loss

        self.enable_short_trades = enable_short_trades

        self.w_mean_reversion = w_mean_reversion
        self.w_trend_pullback = w_trend_pullback
        self.w_breakout = w_breakout
        self.w_macd_rsi = w_macd_rsi

        self.entry_threshold = entry_threshold
        self.dominance_ratio = dominance_ratio

        self.session_filter_enabled = session_filter_enabled
        self.session_hours = session_hours or [(7, 12), (13, 20)]  # Default ranges around London+NY overlap
        self.session_timezone_offset = session_timezone_offset

        self.bb_width_min = bb_width_min
        self.bb_width_max = bb_width_max
        self.vol_sl_min_multiplier = vol_sl_min_multiplier
        self.vol_sl_high_multiplier = vol_sl_high_multiplier
        self.bb_width_vol_threshold = bb_width_vol_threshold

        # Advanced Filter params
        self.enable_advanced_filters = enable_advanced_filters
        self.advanced_filter_confidence_threshold = advanced_filter_confidence_threshold
        self.market_strength_min_score = market_strength_min_score
        self.support_resistance_check = support_resistance_check
        self.divergence_check = divergence_check
        self.volatility_band_check = volatility_band_check

        # State
        self._position = PositionType.OUT
        self._current_trade: Optional[Trade] = None
        self._trade_history: List[Trade] = []
        self._portfolio_value: float = 10000.0
        self._last_trade_index: int = -100
        self._consecutive_losses: int = 0
        self._pause_until_index: int = -1
        self._original_risk: float = risk_per_trade

        # Stats
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = 0.0
        self._gross_profit = 0.0
        self._gross_loss = 0.0

        # Logs
        self._signal_log: List[Dict[str, Any]] = []

        logger.info("🧠 EnsembleRsiStrategyV4 initialized (scalping-ready)")

    # ------------------------------ Utilities ------------------------------

    def _ensure_min_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure minimal indicators exist. Compute locally if absent (safe for backtest/live)."""
        df = data.copy()

        # RSI
        if 'RSI' not in df.columns:
            df = self._calculate_rsi(df)

        # EMAs
        if 'EMA_21' not in df.columns:
            df['EMA_21'] = df['close'].ewm(span=21).mean()
        if 'EMA_50' not in df.columns:
            df['EMA_50'] = df['close'].ewm(span=50).mean()

        # ATR
        if 'ATR' not in df.columns:
            df['ATR'] = self._calculate_atr_series(df)

        # Bollinger + width
        need_bb = any(c not in df.columns for c in ['BB_Middle', 'BB_Upper', 'BB_Lower'])
        if need_bb:
            df = self._calculate_bollinger(df)
        if 'BB_Width' not in df.columns:
            mid = df['BB_Middle'].replace(0, np.nan)
            df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / mid).replace([np.inf, -np.inf], np.nan).fillna(0)

        # MACD
        need_macd = any(c not in df.columns for c in ['MACD', 'MACD_Signal', 'MACD_Histogram'])
        if need_macd:
            df = self._calculate_macd(df)

        # ADX/DI
        need_adx = any(c not in df.columns for c in ['ADX', 'PLUS_DI', 'MINUS_DI'])
        if need_adx:
            try:
                df = self._calculate_adx(df)
            except Exception:
                df['ADX'] = 0.0
                df['PLUS_DI'] = 0.0
                df['MINUS_DI'] = 0.0

        # Keltner
        need_kc = any(c not in df.columns for c in ['KC_Middle', 'KC_Upper', 'KC_Lower'])
        if need_kc:
            try:
                df = self._calculate_keltner(df)
            except Exception:
                df['KC_Middle'] = df['close'].ewm(span=20).mean()
                df['KC_Upper'] = df['KC_Middle']
                df['KC_Lower'] = df['KC_Middle']

        # Donchian
        need_dc = any(c not in df.columns for c in ['DC_Upper', 'DC_Lower'])
        if need_dc:
            try:
                df = self._calculate_donchian(df)
            except Exception:
                df['DC_Upper'] = df['high'].rolling(20).max()
                df['DC_Lower'] = df['low'].rolling(20).min()

        return df

    def _calculate_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            if len(data) < self.rsi_period + 1:
                data['RSI'] = 50.0
                return data
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            data['RSI'] = 100 - (100 / (1 + rs))
            data['RSI'] = data['RSI'].fillna(method='bfill').fillna(50.0)
            return data
        except Exception as e:
            logger.warning(f"RSI calc fallback: {e}")
            data['RSI'] = 50.0
            return data

    def _calculate_atr_series(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean().fillna(method='bfill')
            return atr
        except Exception:
            return pd.Series(index=data.index, data=data['close'] * 0.01)

    def _calculate_bollinger(self, data: pd.DataFrame, period: int = 20, std: int = 2) -> pd.DataFrame:
        try:
            mid = data['close'].rolling(period).mean()
            s = data['close'].rolling(period).std()
            data['BB_Middle'] = mid
            data['BB_Upper'] = mid + (s * std)
            data['BB_Lower'] = mid - (s * std)
            return data
        except Exception:
            data['BB_Middle'] = data['close']
            data['BB_Upper'] = data['close']
            data['BB_Lower'] = data['close']
            return data

    def _calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        try:
            exp1 = data['close'].ewm(span=fast).mean()
            exp2 = data['close'].ewm(span=slow).mean()
            macd = exp1 - exp2
            data['MACD'] = macd
            data['MACD_Signal'] = macd.ewm(span=signal).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            return data
        except Exception:
            data['MACD'] = 0.0
            data['MACD_Signal'] = 0.0
            data['MACD_Histogram'] = 0.0
            return data

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high = data['high']
        low = data['low']
        close = data['close']
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr1 = (high - low)
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
        plus_dm_sm = pd.Series(plus_dm, index=data.index).ewm(alpha=1 / period, adjust=False).mean()
        minus_dm_sm = pd.Series(minus_dm, index=data.index).ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100 * (plus_dm_sm / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm_sm / atr.replace(0, np.nan))
        dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()
        data['PLUS_DI'] = plus_di.fillna(0.0)
        data['MINUS_DI'] = minus_di.fillna(0.0)
        data['ADX'] = adx.fillna(0.0)
        return data

    def _calculate_keltner(self, data: pd.DataFrame, period: int = 20, mult: float = 2.0) -> pd.DataFrame:
        middle = data['close'].ewm(span=period).mean()
        atr = data['ATR'] if 'ATR' in data.columns else self._calculate_atr_series(data, period)
        data['KC_Middle'] = middle
        data['KC_Upper'] = middle + atr * mult
        data['KC_Lower'] = middle - atr * mult
        return data

    def _calculate_donchian(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        data['DC_Upper'] = data['high'].rolling(period).max()
        data['DC_Lower'] = data['low'].rolling(period).min()
        return data

    # ------------------------------ Scorers ------------------------------

    def _score_mean_reversion(self, df: pd.DataFrame) -> Tuple[float, float, List[str]]:
        msgs: List[str] = []
        c = df['close'].iloc[-1]
        rsi = float(df['RSI'].iloc[-1])
        bb_u = df['BB_Upper'].iloc[-1]
        bb_l = df['BB_Lower'].iloc[-1]
        bb_w = float(df['BB_Width'].iloc[-1]) if 'BB_Width' in df.columns else 0.0
        adx = float(df['ADX'].iloc[-1]) if 'ADX' in df.columns else 0.0

        long_score = 0.0
        short_score = 0.0

        # Bollinger touch preference
        if c <= bb_l and rsi <= (self.rsi_oversold + self.rsi_entry_buffer):
            long_score += 1.0
            msgs.append("MR: BB_L touch + RSI low")
        if c >= bb_u and rsi >= (self.rsi_overbought - self.rsi_entry_buffer):
            short_score += 1.0
            msgs.append("MR: BB_U touch + RSI high")

        # Prefer ranging markets for MR
        if adx < 18:
            long_score += 0.3
            short_score += 0.3
            msgs.append("MR: Low ADX boost")

        # BB width gating
        if self.bb_width_min is not None and bb_w < self.bb_width_min:
            msgs.append("MR: Rejected - BB width too small")
            return 0.0, 0.0, msgs
        if self.bb_width_max is not None and bb_w > self.bb_width_max:
            msgs.append("MR: Rejected - BB width too large")
            return 0.0, 0.0, msgs

        return long_score, short_score, msgs

    def _score_trend_pullback(self, df: pd.DataFrame) -> Tuple[float, float, List[str]]:
        msgs: List[str] = []
        c = df['close'].iloc[-1]
        ema21 = float(df['EMA_21'].iloc[-1])
        ema50 = float(df['EMA_50'].iloc[-1])
        rsi = float(df['RSI'].iloc[-1])
        adx = float(df['ADX'].iloc[-1]) if 'ADX' in df.columns else 0.0

        long_score = 0.0
        short_score = 0.0

        # Uptrend alignment + pullback near EMA21
        if ema21 >= ema50 and (c >= ema50 * 0.996) and (c <= ema21 * 1.004) and rsi >= 45.0:
            long_score += 1.0
            msgs.append("TP: Uptrend EMA + near EMA21, RSI>=45")

        # Downtrend alignment + pullback near EMA21
        if ema21 <= ema50 and (c <= ema50 * 1.004) and (c >= ema21 * 0.996) and rsi <= 55.0:
            short_score += 1.0
            msgs.append("TP: Downtrend EMA + near EMA21, RSI<=55")

        # Trend strength via ADX
        if adx > 20:
            if long_score > 0:
                long_score += 0.2
            if short_score > 0:
                short_score += 0.2
            msgs.append("TP: ADX support")

        return long_score, short_score, msgs

    def _score_breakout(self, df: pd.DataFrame) -> Tuple[float, float, List[str]]:
        msgs: List[str] = []
        c = df['close'].iloc[-1]
        dc_u = df['DC_Upper'].iloc[-1] if 'DC_Upper' in df.columns else np.nan
        dc_l = df['DC_Lower'].iloc[-1] if 'DC_Lower' in df.columns else np.nan
        kc_u = df['KC_Upper'].iloc[-1] if 'KC_Upper' in df.columns else np.nan
        kc_l = df['KC_Lower'].iloc[-1] if 'KC_Lower' in df.columns else np.nan
        adx = float(df['ADX'].iloc[-1]) if 'ADX' in df.columns else 0.0
        bb_w = float(df['BB_Width'].iloc[-1]) if 'BB_Width' in df.columns else 0.0

        long_score = 0.0
        short_score = 0.0

        # Expansion regime
        if bb_w > (self.bb_width_min or 0.0) * 1.5:
            if not np.isnan(dc_u) and c > dc_u:
                long_score += 1.0
                msgs.append("BO: Donchian breakout up")
            if not np.isnan(dc_l) and c < dc_l:
                short_score += 1.0
                msgs.append("BO: Donchian breakout down")

            # Keltner channel poke
            if not np.isnan(kc_u) and c > kc_u:
                long_score += 0.3
                msgs.append("BO: Keltner upper breach")
            if not np.isnan(kc_l) and c < kc_l:
                short_score += 0.3
                msgs.append("BO: Keltner lower breach")

            # ADX support
            if adx > 22:
                long_score += 0.2
                short_score += 0.2
                msgs.append("BO: ADX boost")

        return long_score, short_score, msgs

    def _score_macd_rsi(self, df: pd.DataFrame) -> Tuple[float, float, List[str]]:
        msgs: List[str] = []
        rsi = float(df['RSI'].iloc[-1])
        hist = float(df['MACD_Histogram'].iloc[-1])
        hist_prev = float(df['MACD_Histogram'].iloc[-2]) if len(df) > 1 and not pd.isna(df['MACD_Histogram'].iloc[-2]) else hist

        long_score = 0.0
        short_score = 0.0

        # Histogram turning up through zero region with RSI > 50
        if hist >= 0 and hist_prev < hist and rsi > 50:
            long_score += 0.8
            msgs.append("MACD: Upturn + RSI>50")

        # Histogram turning down with RSI < 50
        if hist <= 0 and hist_prev > hist and rsi < 50:
            short_score += 0.8
            msgs.append("MACD: Downturn + RSI<50")

        return long_score, short_score, msgs

    def _aggregate_scores(self, df: pd.DataFrame) -> Tuple[float, float, List[str]]:
        """
        Returns (long_score, short_score, messages)
        """
        l1, s1, m1 = self._score_mean_reversion(df)
        l2, s2, m2 = self._score_trend_pullback(df)
        l3, s3, m3 = self._score_breakout(df)
        l4, s4, m4 = self._score_macd_rsi(df)

        long_score = (
            self.w_mean_reversion * l1
            + self.w_trend_pullback * l2
            + self.w_breakout * l3
            + self.w_macd_rsi * l4
        )
        short_score = (
            self.w_mean_reversion * s1
            + self.w_trend_pullback * s2
            + self.w_breakout * s3
            + self.w_macd_rsi * s4
        )

        msgs = []
        msgs += [f"MR:{l1:.2f}/{s1:.2f}"] + m1
        msgs += [f"TP:{l2:.2f}/{s2:.2f}"] + m2
        msgs += [f"BO:{l3:.2f}/{s3:.2f}"] + m3
        msgs += [f"MD:{l4:.2f}/{s4:.2f}"] + m4
        msgs.append(f"Σ LONG={long_score:.2f}, SHORT={short_score:.2f}")

        return long_score, short_score, msgs

    def check_entry_conditions(self, df: pd.DataFrame, position_type: PositionType) -> Tuple[bool, List[str]]:
        """
        Gate entries for LONG/SHORT using ensemble scores plus basic session/spacing throttle + advanced filters.
        Returns (eligible: bool, messages: List[str])
        """
        try:
            # Ensure minimal features exist
            df = self._ensure_min_features(df)

            # Basic pre-checks to avoid noise entries
            if len(df) < 50:
                return False, ["Insufficient data"]

            now = df.index[-1]
            if not self._in_active_session(now):
                return False, ["Outside active session"]

            candles_since_last = len(df) - 1 - self._last_trade_index
            if candles_since_last < self.min_candles_between:
                return False, [f"Spacing {candles_since_last}<{self.min_candles_between}"]

            recent_trades = len([t for t in self._trade_history[-100:] if t.exit_time])
            if recent_trades >= self.max_trades_per_100:
                return False, [f"Throttle {recent_trades}/{self.max_trades_per_100}"]

            # Advanced market filters (new integration)
            advanced_filter_ok, advanced_conditions = self._evaluate_advanced_filters(df, position_type)
            if not advanced_filter_ok:
                return False, advanced_conditions

            # Aggregate ensemble scores
            long_score, short_score, msgs = self._aggregate_scores(df)

            # Directional eligibility with threshold + dominance ratio
            if position_type == PositionType.LONG:
                eligible = (long_score >= self.entry_threshold) and (long_score >= short_score * self.dominance_ratio)
            else:
                eligible = (
                    self.enable_short_trades
                    and (short_score >= self.entry_threshold)
                    and (short_score >= long_score * self.dominance_ratio)
                )

            # If eligible based on ensemble but not meeting advanced filter requirements, be more selective
            if eligible:
                # Combine ensemble and advanced filter conditions
                return True, msgs + advanced_conditions
            else:
                return False, msgs

        except Exception as e:
            logger.error(f"check_entry_conditions error: {e}")
            return False, [f"Error: {e}"]
    # ------------------------------ Risk/Position helpers ------------------------------

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        series = data['ATR'] if 'ATR' in data.columns else self._calculate_atr_series(data, period)
        atr = float(series.iloc[-1]) if len(series) else float(data['close'].iloc[-1]) * 0.01
        if atr <= 0 or np.isnan(atr):
            atr = float(data['close'].iloc[-1]) * 0.015
        return atr

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        try:
            current_risk = self.risk_per_trade
            if self.risk_reduction_after_loss and self._consecutive_losses > 0:
                risk_reduction = max(0.7, 1.0 - (self._consecutive_losses * 0.05))
                current_risk = self._original_risk * risk_reduction

            risk_amount = self._portfolio_value * current_risk
            price_risk = (entry_price - stop_loss) if stop_loss < entry_price else (stop_loss - entry_price)
            if price_risk <= 0 or price_risk > entry_price * 0.1:
                return 0.0

            position_size = risk_amount / price_risk
            max_position = self._portfolio_value * self.max_position_size_ratio
            position_size = min(position_size, max_position)

            if position_size < self.min_position_size:
                return 0.0

            return position_size
        except Exception:
            return 0.0

    def calculate_stop_take_profit(self, data: pd.DataFrame, position_type: PositionType, entry_price: float) -> Tuple[float, float]:
        # Base ATR
        atr = self.calculate_atr(data)

        # Detect volatility regime using BB width and ADX if available
        bb_w = float(data['BB_Width'].iloc[-1]) if 'BB_Width' in data.columns and not pd.isna(data['BB_Width'].iloc[-1]) else 0.0
        adx = float(data['ADX'].iloc[-1]) if 'ADX' in data.columns and not pd.isna(data['ADX'].iloc[-1]) else 0.0

        # Enforce minimum SL multiplier and widen in volatile regimes
        sl_mult = max(self.stop_loss_atr_multiplier, getattr(self, 'vol_sl_min_multiplier', 1.5))
        if (self.bb_width_vol_threshold is not None and bb_w >= self.bb_width_vol_threshold) or adx >= 22.0:
            sl_mult = max(sl_mult, getattr(self, 'vol_sl_high_multiplier', 2.2))

        # Floor ATR to avoid ultra-tight stops on very low TFs
        atr_floor = max(atr, entry_price * 0.002)  # 0.2% of price
        tp_mult = sl_mult * self.take_profit_ratio

        if position_type == PositionType.LONG:
            sl = entry_price - (atr_floor * sl_mult)
            tp = entry_price + (atr_floor * tp_mult)
            if sl >= entry_price:
                sl = entry_price * 0.99
            if tp <= entry_price:
                tp = entry_price * 1.02
        else:
            sl = entry_price + (atr_floor * sl_mult)
            tp = entry_price - (atr_floor * tp_mult)
            if sl <= entry_price:
                sl = entry_price * 1.01
            if tp >= entry_price:
                tp = entry_price * 0.98

        return float(sl), float(tp)

    # ------------------------------ Exit logic ------------------------------

    def check_exit_conditions(self, data: pd.DataFrame, current_index: int) -> Optional[Dict[str, Any]]:
        if self._position == PositionType.OUT or self._current_trade is None:
            return None

        c = float(data['close'].iloc[-1])
        entry_price = float(self._current_trade.entry_price)

        # Profit %
        if self._position == PositionType.LONG:
            profit_pct = ((c - entry_price) / entry_price) * 100.0
        else:
            profit_pct = ((entry_price - c) / entry_price) * 100.0

        # Partial exit
        if (self.enable_partial_exit
                and profit_pct >= self.partial_exit_threshold
                and not self._current_trade.partial_exit_done):
            self._current_trade.partial_exit_done = True
            partial_qty = self._current_trade.quantity * self.partial_exit_ratio
            partial_pnl = (profit_pct / 100.0) * partial_qty * entry_price
            self._portfolio_value += partial_pnl

            # update trailing after partial
            if self.enable_trailing_stop:
                atr = self.calculate_atr(data)
                if self._position == PositionType.LONG:
                    self._current_trade.trailing_stop = c - (atr * self.trailing_stop_atr_multiplier)
                else:
                    self._current_trade.trailing_stop = c + (atr * self.trailing_stop_atr_multiplier)

            return {
                "action": "PARTIAL_EXIT",
                "price": c,
                "quantity": partial_qty,
                "pnl_percentage": profit_pct,
                "pnl_amount": partial_pnl,
                "reason": "PARTIAL_TAKE_PROFIT"
            }

        # Hard SL / TP
        if self._position == PositionType.LONG:
            if c <= self._current_trade.stop_loss:
                return self._create_exit_signal("STOP_LOSS", c, data.index[-1])
            if c >= self._current_trade.take_profit:
                return self._create_exit_signal("TAKE_PROFIT", c, data.index[-1])
        else:
            if c >= self._current_trade.stop_loss:
                return self._create_exit_signal("STOP_LOSS", c, data.index[-1])
            if c <= self._current_trade.take_profit:
                return self._create_exit_signal("TAKE_PROFIT", c, data.index[-1])

        # Trailing stop
        if self.enable_trailing_stop and abs(profit_pct) >= self.trailing_activation_percent:
            atr = self.calculate_atr(data)
            trail_atr = atr * self.trailing_stop_atr_multiplier
            if self._position == PositionType.LONG:
                new_trail = c - trail_atr
                if new_trail > self._current_trade.trailing_stop:
                    self._current_trade.trailing_stop = new_trail
                if c <= self._current_trade.trailing_stop:
                    return self._create_exit_signal("TRAILING_STOP", c, data.index[-1])
            else:
                new_trail = c + trail_atr
                if self._current_trade.trailing_stop == self._current_trade.stop_loss or new_trail < self._current_trade.trailing_stop:
                    self._current_trade.trailing_stop = new_trail
                if c >= self._current_trade.trailing_stop:
                    return self._create_exit_signal("TRAILING_STOP", c, data.index[-1])

        # Time exit
        duration = current_index - self._last_trade_index
        if duration >= self.max_trade_duration:
            return self._create_exit_signal("TIME_EXIT", c, data.index[-1])

        return None

    def _create_exit_signal(self, exit_reason: str, price: float, time: pd.Timestamp) -> Dict[str, Any]:
        if self._current_trade is None:
            return {"action": "HOLD", "reason": "No active trade"}
        entry_price = self._current_trade.entry_price
        qty = self._current_trade.quantity
        pnl_amount = (price - entry_price) * qty if self._position == PositionType.LONG else (entry_price - price) * qty
        pnl_pct = (pnl_amount / (entry_price * qty)) * 100.0 if entry_price and qty else 0.0

        # Portfolio update
        self._portfolio_value += pnl_amount
        self._total_pnl += pnl_amount
        self._total_trades += 1

        # Loss streak handling
        if pnl_amount < 0:
            self._consecutive_losses += 1
            self._gross_loss += abs(pnl_amount)
            if self._consecutive_losses >= self.max_consecutive_losses:
                self._pause_until_index = len(self._trade_history) + self.pause_after_losses
                logger.warning(f"Pause after {self._consecutive_losses} losses for {self.pause_after_losses} candles")
        else:
            self._consecutive_losses = 0
            self._winning_trades += 1
            self._gross_profit += pnl_amount

        # Record trade
        self._current_trade.exit_price = price
        self._current_trade.exit_time = time
        self._current_trade.pnl_percentage = pnl_pct
        self._current_trade.pnl_amount = pnl_amount
        self._current_trade.exit_reason = ExitReason(exit_reason)

        old_pos = self._position
        self._trade_history.append(self._current_trade)
        self._position = PositionType.OUT
        self._current_trade = None

        logger.info(f"Exit {exit_reason}: price={price:.5f} pnl%={pnl_pct:.2f}")
        return {
            "action": "EXIT",
            "price": float(price),
            "exit_reason": exit_reason,
            "pnl_percentage": round(pnl_pct, 2),
            "pnl_amount": round(pnl_amount, 2),
            "position": "OUT",
            "previous_position": old_pos.value,
        }

    # ------------------------------ Sessions ------------------------------

    def _in_active_session(self, ts: pd.Timestamp) -> bool:
        if not self.session_filter_enabled:
            return True
        try:
            hour = (ts.hour + int(self.session_timezone_offset)) % 24
            for lo, hi in self.session_hours:
                if lo <= hour <= hi:
                    return True
            return False
        except Exception:
            return True

    # ------------------------------ Signal generation ------------------------------

    def generate_signal(self, data: pd.DataFrame, current_index: int = 0) -> Dict[str, Any]:
        try:
            df = self._ensure_min_features(data)

            now = df.index[-1]
            price = float(df['close'].iloc[-1])
            rsi = float(df['RSI'].iloc[-1]) if 'RSI' in df.columns else None

            # EXIT first
            exit_sig = self.check_exit_conditions(df, current_index)
            if exit_sig:
                self._signal_log.append({
                    "time": now, "action": exit_sig.get("action", "EXIT"), "price": price,
                    "position": self._position.value if hasattr(self, "_position") else "OUT",
                    "reason": exit_sig.get("reason") or exit_sig.get("exit_reason", ""),
                    "pnl_percentage": exit_sig.get("pnl_percentage"), "rsi": rsi
                })
                return exit_sig

            # Minimum data / pause / session
            if len(df) < 50:
                hold = {"action": "HOLD", "reason": "Insufficient data"}
                self._signal_log.append({"time": now, "action": "HOLD", "price": price, "position": self._position.value, "reason": hold["reason"], "rsi": rsi})
                return hold

            if current_index <= self._pause_until_index:
                hold = {"action": "HOLD", "reason": "Paused after losses"}
                self._signal_log.append({"time": now, "action": "HOLD", "price": price, "position": self._position.value, "reason": hold["reason"], "rsi": rsi})
                return hold

            if not self._in_active_session(now):
                hold = {"action": "HOLD", "reason": "Outside active session"}
                self._signal_log.append({"time": now, "action": "HOLD", "price": price, "position": self._position.value, "reason": hold["reason"], "rsi": rsi})
                return hold

            # Spacing/throttle
            candles_since_last = len(df) - 1 - self._last_trade_index
            if candles_since_last < self.min_candles_between:
                hold = {"action": "HOLD", "reason": f"Too close to last trade ({candles_since_last} candles)"}
                self._signal_log.append({"time": now, "action": "HOLD", "price": price, "position": self._position.value, "reason": hold["reason"], "rsi": rsi})
                return hold

            recent_trades = len([t for t in self._trade_history[-100:] if t.exit_time])
            if recent_trades >= self.max_trades_per_100:
                hold = {"action": "HOLD", "reason": f"Trade throttle hit ({recent_trades}/{self.max_trades_per_100})"}
                self._signal_log.append({"time": now, "action": "HOLD", "price": price, "position": self._position.value, "reason": hold["reason"], "rsi": rsi})
                return hold

            # Try entries only when OUT
            if self._position == PositionType.OUT:
                long_score, short_score, msgs = self._aggregate_scores(df)

                # Decide direction with dominance
                action = "HOLD"
                direction: Optional[PositionType] = None
                reason = " | ".join(msgs)

                if long_score >= self.entry_threshold and long_score >= short_score * self.dominance_ratio:
                    action = "BUY"
                    direction = PositionType.LONG
                elif (self.enable_short_trades and
                      short_score >= self.entry_threshold and short_score >= long_score * self.dominance_ratio):
                    action = "SELL"
                    direction = PositionType.SHORT

                if direction is not None:
                    sl, tp = self.calculate_stop_take_profit(df, direction, price)
                    size = self.calculate_position_size(price, sl)
                    if size > 0:
                        self._position = direction
                        self._current_trade = Trade(
                            entries=[TradeEntry(price, size, df.index[-1])],
                            position_type=direction,
                            stop_loss=sl,
                            take_profit=tp,
                            initial_stop_loss=sl,
                            trailing_stop=sl,
                            entry_conditions=msgs
                        )
                        self._last_trade_index = current_index
                        logger.info(f"Ensemble ENTRY {action} @ {price:.5f} size={size:.0f}")

                        sig = {
                            "action": action,
                            "price": float(price),
                            "position_size": float(size),
                            "stop_loss": float(sl),
                            "take_profit": float(tp),
                            "reason": reason,
                            "position": direction.value
                        }
                        self._signal_log.append({
                            "time": now, "action": action, "price": price, "position": direction.value,
                            "stop_loss": float(sl), "take_profit": float(tp), "rsi": rsi, "reason": reason
                        })
                        return sig

                # otherwise hold
                hold = {"action": "HOLD", "reason": reason}
                self._signal_log.append({"time": now, "action": "HOLD", "price": price, "position": self._position.value, "reason": reason, "rsi": rsi})
                return hold

            # Default HOLD when in a position and no exit triggered
            hold = {"action": "HOLD", "reason": "Managing open trade"}
            self._signal_log.append({"time": now, "action": "HOLD", "price": price, "position": self._position.value, "reason": hold["reason"], "rsi": rsi})
            return hold

        except Exception as e:
            logger.error(f"Ensemble generate_signal error: {e}")
            return {"action": "HOLD", "reason": f"Error: {e}"}

    # ------------------------------ Public metrics API ------------------------------

    def get_performance_metrics(self) -> Dict[str, Any]:
        if self._total_trades == 0:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "current_portfolio_value": round(self._portfolio_value, 2),
                "consecutive_losses": self._consecutive_losses,
                "current_position": self._position.value,
            }
        win_rate = (self._winning_trades / self._total_trades) * 100.0
        profit_factor = self._gross_profit / max(self._gross_loss, 1e-6)
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
            "portfolio_return": round(((self._portfolio_value - 10000) / 10000) * 100.0, 2),
            "consecutive_losses": self._consecutive_losses,
            "current_position": self._position.value,
        }

    def get_trade_history(self) -> List[Trade]:
        return self._trade_history.copy()

    def get_signal_log(self) -> List[Dict[str, Any]]:
        return self._signal_log.copy()

    def reset_state(self):
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
        self._original_risk = self.risk_per_trade
        logger.info("Ensemble strategy state reset complete")

    def _evaluate_advanced_filters(self, df: pd.DataFrame, position_type: PositionType) -> Tuple[bool, List[str]]:
        """Evaluate advanced market filters to enhance entry decisions in ensemble strategy"""
        conditions = []
        
        # Check if advanced filters are enabled
        if not self.enable_advanced_filters:
            return True, ["Advanced filters disabled"]
        
        try:
            # Market regime detection
            regime_info = AdvancedMarketFilters.detect_market_regime(df)
            conditions.append(f"Regime: {regime_info['regime']} (Conf: {regime_info['confidence']})")
            
            # Only trade in favorable market conditions based on position type
            if position_type == PositionType.LONG:
                # For long trades, prefer bullish or neutral regimes
                if (regime_info['regime'] == 'BEARISH' and 
                    regime_info['confidence'] > self.advanced_filter_confidence_threshold):
                    return False, [f"Regime filter: Avoiding long in bearish market ({regime_info['confidence']}>{self.advanced_filter_confidence_threshold})"]
            elif position_type == PositionType.SHORT:
                # For short trades, prefer bearish or neutral regimes
                if (regime_info['regime'] == 'BULLISH' and 
                    regime_info['confidence'] > self.advanced_filter_confidence_threshold):
                    return False, [f"Regime filter: Avoiding short in bullish market ({regime_info['confidence']}>{self.advanced_filter_confidence_threshold})"]
            
            # Trend strength check
            trend_info = AdvancedMarketFilters.calculate_trend_strength(df)
            conditions.append(f"Trend: {trend_info['direction']} (Str: {trend_info['strength']})")
            
            # Support/Resistance levels (only if enabled)
            if self.support_resistance_check:
                support, resistance = AdvancedMarketFilters.calculate_support_resistance(df)
                current_price = df['close'].iloc[-1]
                
                if position_type == PositionType.LONG and current_price > resistance:
                    conditions.append(f"Price above resistance: {resistance:.4f}")
                elif position_type == PositionType.SHORT and current_price < support:
                    conditions.append(f"Price below support: {support:.4f}")
            
            # Volatility bands position (only if enabled)
            if self.volatility_band_check:
                vol_bands = AdvancedMarketFilters.calculate_volatility_bands(df)
                position_in_band = vol_bands['price_position']
                
                # For LONG positions, prefer when price is not too high in the band
                if position_type == PositionType.LONG and position_in_band > 0.8:
                    conditions.append(f"Price too high in volatility band ({position_in_band:.2f})")
                # For SHORT positions, prefer when price is not too low in the band
                elif position_type == PositionType.SHORT and position_in_band < 0.2:
                    conditions.append(f"Price too low in volatility band ({position_in_band:.2f})")
            
            # Divergence detection (only if enabled)
            if self.divergence_check:
                divergence_info = AdvancedMarketFilters.detect_divergence(df)
                if divergence_info['divergence'] != 'NONE':
                    conditions.append(f"Divergence: {divergence_info['divergence']} ({divergence_info.get('type', 'N/A')})")
            
            # Market strength score
            strength_score = AdvancedMarketFilters.get_market_strength_score(df)
            conditions.append(f"Market strength: {strength_score}/10")
            
            # Only allow trades in moderate to strong market conditions
            if strength_score < self.market_strength_min_score:
                return False, [f"Market strength too low ({strength_score}<{self.market_strength_min_score})"]
            
            return True, conditions

        except Exception as e:
            logger.error(f"Error in ensemble advanced filters evaluation: {e}")
            return True, [f"Advanced filters error: {e} (Proceeding with basic conditions)"]