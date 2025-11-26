"""
Comprehensive diagnostic analysis system for algorithmic trading strategies.
Captures detailed market snapshots, indicator values, decision processes, and trade outcomes.
All data is stored in MongoDB for later evaluation and analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from utils.logger import get_mongo_collection
from strategies.enhanced_rsi_strategy_v4 import EnhancedRsiStrategyV4
from backtest.enhanced_rsi_backtest_v4 import EnhancedRSIBacktestV4
from config.parameters import OPTIMIZED_PARAMS_V4

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classification"""
    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    RANGE_BOUND = "RANGE_BOUND"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    CHOPPY = "CHOPPY"

@dataclass
class MarketSnapshot:
    """Complete market snapshot at a specific moment"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    volatility_metrics: Dict[str, float]  # ATR, std_dev, bb_width, etc.
    trend_state: Dict[str, Any]  # trend direction, strength, etc.
    market_regime: MarketRegime
    additional_metrics: Dict[str, float]  # any other relevant metrics

@dataclass
class IndicatorSnapshot:
    """Complete indicator snapshot"""
    rsi: float
    ema_fast: float
    ema_slow: float
    atr: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    macd: float
    macd_signal: float
    adx: float
    plus_di: float
    minus_di: float
    custom_signals: Dict[str, Any]

@dataclass
class DecisionProcess:
    """Detailed decision-making process"""
    decision_type: str  # "ENTRY", "EXIT", "SKIP", "PARTIAL_EXIT"
    action_taken: str  # "BUY", "SELL", "EXIT", "HOLD", etc.
    decision_reason: str  # Detailed explanation
    filters_passed: List[str]
    filters_failed: List[str]
    conditions_checked: List[Dict[str, Any]]  # List of condition checks with results
    entry_logic: str  # What led to entry decision
    exit_logic: str  # What led to exit decision (if applicable)

@dataclass
class TradeDetails:
    """Detailed trade information"""
    entry_type: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: Optional[float] = None
    entry_timestamp: Optional[datetime] = None
    exit_timestamp: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    pnl_amount: Optional[float] = None
    pnl_percentage: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    trade_duration: Optional[int] = None  # in candles
    outcome: Optional[str] = None  # "WIN", "LOSS", "BREAKEVEN"

@dataclass
class DiagnosticLog:
    """Complete diagnostic log entry"""
    test_id: str
    timestamp: datetime
    market_state: MarketSnapshot
    indicators: IndicatorSnapshot
    decision: DecisionProcess
    trade_info: Optional[TradeDetails]
    result: Dict[str, Any]

class DiagnosticSystem:
    """Main diagnostic system that captures comprehensive strategy behavior"""
    
    def __init__(self):
        self.test_id = str(uuid.uuid4())
        self.logger = logging.getLogger(__name__)
        
        # Setup MongoDB collections
        self.backtest_logs_collection = get_mongo_collection("backtest_logs")
        self.trade_results_collection = get_mongo_collection("trade_results")
        self.market_snapshots_collection = get_mongo_collection("market_snapshots")
        self.test_metadata_collection = get_mongo_collection("test_metadata")
        self.performance_metrics_collection = get_mongo_collection("performance_metrics")
        
        if not all([self.backtest_logs_collection is not None, self.trade_results_collection is not None,
                   self.market_snapshots_collection is not None, self.test_metadata_collection is not None,
                   self.performance_metrics_collection is not None]):
            self.logger.warning("⚠️ MongoDB collections not available. Ensure MongoDB is running.")
        else:
            self.logger.info("✅ Diagnostic system initialized with MongoDB integration")

    def _create_simulated_data(self, days_back: int, timeframe: str) -> pd.DataFrame:
        """Create simulated market data when MT5 is not available"""
        import numpy as np
        from datetime import datetime, timedelta

        # Calculate number of candles based on timeframe
        timeframe_minutes = {
            "M1": 1, "M5": 5, "M15": 15, "M30": 30,
            "H1": 60, "H4": 240, "D1": 1440
        }

        minutes_per_day = 24 * 60  # Minutes in a day
        total_minutes = days_back * minutes_per_day

        # Calculate number of candles needed
        tf_minutes = timeframe_minutes.get(timeframe, 60)  # Default to H1
        num_candles = int(total_minutes / tf_minutes)

        # Limit to reasonable size for testing
        num_candles = min(num_candles, 5000)  # Max 5000 candles for performance

        self.logger.info(f"📊 Creating simulated data: {num_candles} candles for {days_back} days at {timeframe}")

        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=num_candles * tf_minutes)

        timestamps = pd.date_range(start=start_time, end=end_time, periods=num_candles + 1)
        timestamps = timestamps[1:]  # Skip the first timestamp to align properly

        # Generate realistic price data using a random walk
        np.random.seed(42)  # For reproducibility in testing

        # Start with a base price
        base_price = 1.2000  # Starting price for forex pair

        # Generate returns (small random changes) - we need num_candles values
        returns = np.random.normal(0, 0.001, num_candles)  # 0.1% daily volatility

        # Calculate price series - we need exactly num_candles+1 prices to get num_candles bars
        prices = [base_price]
        for ret in returns:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)

        # Make sure we have the right number of elements
        # open should have num_candles elements (prices[0] to prices[num_candles-1])
        # close should have num_candles elements (prices[1] to prices[num_candles])
        open_prices = prices[:-1]  # All prices except the last one
        close_prices = prices[1:]  # All prices except the first one

        # Create high, low, and volume arrays with the same length
        high_prices = []
        low_prices = []
        volumes = []

        for i in range(len(open_prices)):
            # Generate high and low prices that are realistic relative to open/close
            op = open_prices[i]
            cl = close_prices[i]
            typical = (op + cl) / 2
            volatility = abs(op - cl) + 0.0005  # Add some base volatility

            high_val = max(op, cl) + abs(np.random.normal(0, volatility/2))
            low_val = min(op, cl) - abs(np.random.normal(0, volatility/2))

            # Ensure low is not below 0 and high is reasonable
            low_val = max(low_val, typical * 0.995)
            high_val = min(high_val, typical * 1.005)

            high_prices.append(high_val)
            low_prices.append(low_val)
            volumes.append(np.random.randint(1000, 10000))

        # Create the DataFrame
        data = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=timestamps[:num_candles])

        # Ensure high >= open,close and low <= open,close
        for i in range(len(data)):
            data.iloc[i, data.columns.get_loc('high')] = max(
                data.iloc[i, data.columns.get_loc('open')],
                data.iloc[i, data.columns.get_loc('close')],
                data.iloc[i, data.columns.get_loc('high')]
            )
            data.iloc[i, data.columns.get_loc('low')] = min(
                data.iloc[i, data.columns.get_loc('open')],
                data.iloc[i, data.columns.get_loc('close')],
                data.iloc[i, data.columns.get_loc('low')]
            )

        self.logger.info(f"✅ Simulated data created: {len(data)} candles from {data.index[0]} to {data.index[-1]}")
        return data

    def calculate_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Classify current market regime"""
        try:
            if len(data) < 30:
                return MarketRegime.RANGE_BOUND
            
            # Calculate regime indicators
            # Trend strength using EMA alignment
            ema_fast = float(data['EMA_21'].iloc[-1]) if 'EMA_21' in data.columns else 0
            ema_slow = float(data['EMA_50'].iloc[-1]) if 'EMA_50' in data.columns else 0
            trend_strength = abs(ema_fast - ema_slow) / data['close'].iloc[-1] if data['close'].iloc[-1] != 0 else 0
            
            # Volatility calculation
            price_std = data['close'].tail(30).std() / data['close'].iloc[-1]
            atr = float(data['ATR'].iloc[-1]) if 'ATR' in data.columns else data['close'].iloc[-1] * 0.01
            
            # Directional movement
            recent_trend = (data['close'].iloc[-1] - data['close'].iloc[-10]) / data['close'].iloc[-10] if len(data) >= 10 else 0
            
            # Range detection using ATR and price movement
            range_threshold = atr * 0.5
            price_movement = abs(data['close'].iloc[-1] - data['close'].iloc[-10]) if len(data) >= 10 else 0
            
            if abs(recent_trend) > 0.05 and trend_strength > 0.005:  # Strong directional move
                if recent_trend > 0:
                    return MarketRegime.BULL_TREND
                else:
                    return MarketRegime.BEAR_TREND
            elif price_std > 0.02:  # High volatility
                return MarketRegime.HIGH_VOLATILITY
            elif price_std < 0.005:  # Very low volatility
                return MarketRegime.LOW_VOLATILITY
            elif price_movement < range_threshold:  # Low movement despite ATR
                return MarketRegime.RANGE_BOUND
            else:
                return MarketRegime.CHOPPY
                
        except Exception as e:
            self.logger.error(f"Error calculating market regime: {e}")
            return MarketRegime.BULL_TREND  # Default fallback
    
    def calculate_volatility_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility-related metrics"""
        try:
            metrics = {}
            
            # ATR
            if 'ATR' in data.columns:
                metrics['atr'] = float(data['ATR'].iloc[-1])
            else:
                # Calculate ATR if not available
                high_low = data['high'] - data['low']
                high_close = (data['high'] - data['close'].shift()).abs()
                low_close = (data['low'] - data['close'].shift()).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = tr.rolling(window=14).mean().iloc[-1]
                metrics['atr'] = float(atr) if not pd.isna(atr) else data['close'].iloc[-1] * 0.01
            
            # Standard deviation
            metrics['std_dev'] = float(data['close'].tail(20).std()) if len(data) >= 20 else 0
            
            # Bollinger Bands width
            if 'BB_Width' in data.columns:
                metrics['bb_width'] = float(data['BB_Width'].iloc[-1])
            else:
                middle = data['close'].rolling(20).mean().iloc[-1]
                std = data['close'].rolling(20).std().iloc[-1]
                metrics['bb_width'] = (std * 2) / middle if middle != 0 else 0
            
            # Price volatility (rolling)
            if len(data) >= 10:
                returns = data['close'].pct_change().tail(10).dropna()
                metrics['price_volatility'] = float(returns.std()) if not returns.empty else 0
            else:
                metrics['price_volatility'] = 0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility metrics: {e}")
            return {
                'atr': data['close'].iloc[-1] * 0.01,
                'std_dev': 0,
                'bb_width': 0,
                'price_volatility': 0
            }
    
    def capture_market_snapshot(self, data: pd.DataFrame, current_index: int) -> MarketSnapshot:
        """Capture comprehensive market snapshot"""
        current_row = data.iloc[current_index]
        
        volatility_metrics = self.calculate_volatility_metrics(data)
        
        # Trend indicators
        trend_state = {}
        try:
            if 'EMA_21' in data.columns and 'EMA_50' in data.columns:
                ema21 = float(data['EMA_21'].iloc[current_index])
                ema50 = float(data['EMA_50'].iloc[current_index])
                
                if ema21 > ema50:
                    trend_state['direction'] = 'BULLISH'
                    trend_state['strength'] = abs(ema21 - ema50) / current_row['close']
                else:
                    trend_state['direction'] = 'BEARISH' 
                    trend_state['strength'] = abs(ema50 - ema21) / current_row['close']
            else:
                trend_state['direction'] = 'NEUTRAL'
                trend_state['strength'] = 0
                
        except Exception as e:
            self.logger.error(f"Error calculating trend state: {e}")
            trend_state = {'direction': 'NEUTRAL', 'strength': 0}
        
        market_regime = self.calculate_market_regime(data.iloc[:current_index+1])
        
        return MarketSnapshot(
            timestamp=pd.to_datetime(data.index[current_index]).to_pydatetime(),
            open=float(current_row['open']),
            high=float(current_row['high']),
            low=float(current_row['low']),
            close=float(current_row['close']),
            volume=float(current_row['volume']) if 'volume' in current_row else 0,
            volatility_metrics=volatility_metrics,
            trend_state=trend_state,
            market_regime=market_regime,
            additional_metrics={}
        )
    
    def capture_indicator_snapshot(self, data: pd.DataFrame, current_index: int) -> IndicatorSnapshot:
        """Capture all indicator values"""
        current_row = data.iloc[current_index]
        
        return IndicatorSnapshot(
            rsi=float(current_row['RSI']) if 'RSI' in current_row else 50,
            ema_fast=float(current_row['EMA_21']) if 'EMA_21' in current_row else 0,
            ema_slow=float(current_row['EMA_50']) if 'EMA_50' in current_row else 0,
            atr=float(current_row['ATR']) if 'ATR' in current_row else 0,
            bb_upper=float(current_row['BB_Upper']) if 'BB_Upper' in current_row else 0,
            bb_middle=float(current_row['BB_Middle']) if 'BB_Middle' in current_row else 0,
            bb_lower=float(current_row['BB_Lower']) if 'BB_Lower' in current_row else 0,
            macd=float(current_row['MACD']) if 'MACD' in current_row else 0,
            macd_signal=float(current_row['MACD_Signal']) if 'MACD_Signal' in current_row else 0,
            adx=float(current_row['ADX']) if 'ADX' in current_row else 0,
            plus_di=float(current_row['PLUS_DI']) if 'PLUS_DI' in current_row else 0,
            minus_di=float(current_row['MINUS_DI']) if 'MINUS_DI' in current_row else 0,
            custom_signals={}
        )
    
    def analyze_decision_process(self, signal: Dict[str, Any], data: pd.DataFrame, current_index: int) -> DecisionProcess:
        """Analyze and document the decision-making process"""
        action = signal.get('action', 'HOLD')
        
        # Determine decision type
        if action in ['BUY', 'SELL']:
            decision_type = 'ENTRY'
        elif action == 'EXIT':
            decision_type = 'EXIT'
        elif action == 'PARTIAL_EXIT':
            decision_type = 'PARTIAL_EXIT'
        else:
            decision_type = 'SKIP'
        
        # Analyze filters and conditions
        filters_passed = []
        filters_failed = []
        conditions_checked = []
        
        # This would be more detailed in a real implementation
        # For now, we'll extract what's available from the signal
        reason = signal.get('reason', 'No reason provided')
        
        # Parse conditions from the reason string (this is a simplified approach)
        if reason:
            if 'RSI' in reason:
                conditions_checked.append({'condition': 'RSI_threshold', 'passed': 'suitable' in reason.lower(), 'value': signal.get('rsi')})
            if 'Trend' in reason or 'trend' in reason:
                conditions_checked.append({'condition': 'Trend_filter', 'passed': 'passed' in reason.lower(), 'value': 'trend_aligned' if 'passed' in reason.lower() else 'trend_misaligned'})
        
        return DecisionProcess(
            decision_type=decision_type,
            action_taken=action,
            decision_reason=reason,
            filters_passed=filters_passed,
            filters_failed=filters_failed,
            conditions_checked=conditions_checked,
            entry_logic=reason if decision_type == 'ENTRY' else '',
            exit_logic=reason if decision_type == 'EXIT' else ''
        )
    
    def capture_trade_details(self, signal: Dict[str, Any], current_index: int) -> Optional[TradeDetails]:
        """Capture detailed trade information"""
        if signal.get('action') in ['BUY', 'SELL', 'EXIT', 'PARTIAL_EXIT']:
            if signal['action'] in ['BUY', 'SELL']:
                return TradeDetails(
                    entry_type='LONG' if signal['action'] == 'BUY' else 'SHORT',
                    entry_price=float(signal.get('price', 0)),
                    entry_timestamp=pd.to_datetime(signal.get('timestamp')).to_pydatetime() if 'timestamp' in signal else None,
                    stop_loss=float(signal.get('stop_loss', 0)) if signal.get('stop_loss') else None,
                    take_profit=float(signal.get('take_profit', 0)) if signal.get('take_profit') else None,
                    position_size=float(signal.get('position_size', 0)) if signal.get('position_size') else None
                )
            elif signal['action'] == 'EXIT':
                return TradeDetails(
                    entry_type='',  # Not applicable for exit
                    entry_price=0,  # Not stored here
                    exit_price=float(signal.get('price', 0)),
                    exit_timestamp=pd.to_datetime(signal.get('timestamp')).to_pydatetime() if 'timestamp' in signal else None,
                    pnl_amount=float(signal.get('pnl_amount', 0)) if signal.get('pnl_amount') else None,
                    pnl_percentage=float(signal.get('pnl_percentage', 0)) if signal.get('pnl_percentage') else None,
                    outcome='WIN' if signal.get('pnl_percentage', 0) > 0 else 'LOSS'
                )
        return None
    
    def log_diagnostic_entry(self, data: pd.DataFrame, current_index: int, signal: Dict[str, Any], strategy_state: Any) -> None:
        """Log complete diagnostic information for a single candle/signal"""
        try:
            # Skip if not enough data
            if current_index < 50:
                return
            
            # Create market snapshot
            market_snapshot = self.capture_market_snapshot(data, current_index)
            
            # Create indicator snapshot
            indicator_snapshot = self.capture_indicator_snapshot(data, current_index)
            
            # Analyze decision process
            decision_process = self.analyze_decision_process(signal, data, current_index)
            
            # Capture trade details if applicable
            trade_details = self.capture_trade_details(signal, current_index)
            
            # Create result dict (can be extended based on strategy state)
            result = {
                'portfolio_value': getattr(strategy_state, '_portfolio_value', 10000.0),
                'current_position': getattr(strategy_state, '_position', 'OUT').value if hasattr(strategy_state, '_position') else 'OUT',
                'total_trades': getattr(strategy_state, '_total_trades', 0)
            }
            
            # Create diagnostic log
            diagnostic_log = DiagnosticLog(
                test_id=self.test_id,
                timestamp=pd.to_datetime(data.index[current_index]).to_pydatetime(),
                market_state=market_snapshot,
                indicators=indicator_snapshot,
                decision=decision_process,
                trade_info=trade_details,
                result=result
            )
            
            # Convert to dictionary for MongoDB storage
            log_dict = {
                'test_id': diagnostic_log.test_id,
                'timestamp': diagnostic_log.timestamp,
                'market_state': asdict(diagnostic_log.market_state),
                'indicators': asdict(diagnostic_log.indicators),
                'decision': asdict(diagnostic_log.decision),
                'result': diagnostic_log.result
            }
            
            # Add trade info if available
            if diagnostic_log.trade_info:
                log_dict['trade_info'] = asdict(diagnostic_log.trade_info)
            
            # Store in MongoDB - convert enum values to strings first
            if self.backtest_logs_collection is not None:
                # Convert any enum values to their string values
                def convert_enums_to_str(obj):
                    if isinstance(obj, dict):
                        return {k: convert_enums_to_str(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_enums_to_str(v) for v in obj]
                    elif hasattr(obj, 'value'):  # enum values
                        return obj.value
                    else:
                        return obj

                serializable_log_dict = convert_enums_to_str(log_dict)
                self.backtest_logs_collection.insert_one(serializable_log_dict)
            
            # Store market snapshot separately for easier querying
            market_snap_dict = {
                'test_id': self.test_id,
                'timestamp': diagnostic_log.timestamp,
                'market_state': asdict(diagnostic_log.market_state),
                'symbol': getattr(strategy_state, '_current_symbol', 'UNKNOWN'),
                'timeframe': getattr(strategy_state, '_current_timeframe', 'UNKNOWN')
            }
            
            if self.market_snapshots_collection is not None:
                # Convert any enum values to their string values
                def convert_enums_to_str(obj):
                    if isinstance(obj, dict):
                        return {k: convert_enums_to_str(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_enums_to_str(v) for v in obj]
                    elif hasattr(obj, 'value'):  # enum values
                        return obj.value
                    else:
                        return obj

                serializable_market_snap_dict = convert_enums_to_str(market_snap_dict)
                self.market_snapshots_collection.insert_one(serializable_market_snap_dict)
                
        except Exception as e:
            self.logger.error(f"Error logging diagnostic entry: {e}")
    
    def run_comprehensive_diagnostic(self,
                                   symbol: str,
                                   timeframe: str,
                                   days_back: int,
                                   strategy_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run comprehensive diagnostic backtest"""
        try:
            self.logger.info(f"🚀 Starting comprehensive diagnostic for {symbol} ({timeframe})")

            # Setup backtest engine and strategy
            backtest_engine = EnhancedRSIBacktestV4()
            strategy_params = strategy_params or OPTIMIZED_PARAMS_V4

            # Fetch data
            try:
                data = backtest_engine.fetch_real_data_from_mt5(symbol, timeframe, days_back)
            except Exception as e:
                self.logger.warning(f"⚠️ MT5 connection failed: {e}. Creating simulated data...")
                # Create simulated data for testing purposes
                data = self._create_simulated_data(days_back, timeframe)

                # Calculate technical indicators for simulated data
                data = backtest_engine._calculate_technical_indicators(data)
            
            # Initialize strategy
            strategy = EnhancedRsiStrategyV4(**strategy_params)
            
            # Set symbol and timeframe in strategy for logging
            strategy._current_symbol = symbol
            strategy._current_timeframe = timeframe
            
            # Store test metadata
            test_metadata = {
                'test_id': self.test_id,
                'symbol': symbol,
                'timeframe': timeframe,
                'days_back': days_back,
                'strategy_params': strategy_params,
                'start_date': data.index[0],
                'end_date': data.index[-1],
                'total_candles': len(data),
                'created_at': datetime.now()
            }
            
            if self.test_metadata_collection is not None:
                self.test_metadata_collection.insert_one(test_metadata)
            
            # Run backtest with diagnostic logging
            self.logger.info(f"📊 Processing {len(data)} candles for diagnostic analysis")
            
            for i in range(len(data)):
                if i < 50:  # Skip initial candles that don't have enough historical data
                    continue
                
                current_data = data.iloc[:i+1].copy()
                
                # Generate signal
                signal = strategy.generate_signal(current_data, i)
                
                # Log diagnostic information
                self.log_diagnostic_entry(current_data, i, signal, strategy)
                
                # Show progress
                if (i + 1) % 500 == 0:
                    progress = (i + 1) / len(data) * 100
                    self.logger.info(f"📈 Diagnostic progress: {progress:.1f}% ({i + 1}/{len(data)})")
            
            # Calculate and store performance metrics after backtest
            performance_metrics = self.calculate_comprehensive_metrics(strategy, data)
            self.store_performance_metrics(performance_metrics)
            
            self.logger.info(f"✅ Diagnostic completed for {symbol} ({timeframe}) - Test ID: {self.test_id}")
            
            return {
                'test_id': self.test_id,
                'symbol': symbol,
                'timeframe': timeframe,
                'total_candles': len(data),
                'performance_metrics': performance_metrics
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in comprehensive diagnostic: {e}")
            raise
    
    def calculate_comprehensive_metrics(self, strategy, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        try:
            # Get strategy metrics
            strategy_metrics = strategy.get_performance_metrics()
            
            # Calculate additional metrics
            trade_history = strategy.get_trade_history()
            
            total_trades = len(trade_history)
            winning_trades = len([t for t in trade_history if t.pnl_amount and t.pnl_amount > 0])
            losing_trades = len([t for t in trade_history if t.pnl_amount and t.pnl_amount < 0])
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate profit/loss ratios
            winning_pnls = [t.pnl_amount for t in trade_history if t.pnl_amount and t.pnl_amount > 0]
            losing_pnls = [t.pnl_amount for t in trade_history if t.pnl_amount and t.pnl_amount < 0]
            
            avg_win = np.mean(winning_pnls) if winning_pnls else 0
            avg_loss = np.mean(losing_pnls) if losing_pnls else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            # Calculate drawdown
            equity_values = [10000.0]  # Start with initial capital
            current_capital = 10000.0
            for trade in trade_history:
                if trade.pnl_amount:
                    current_capital += trade.pnl_amount
                    equity_values.append(current_capital)
            
            if len(equity_values) > 1:
                equity_series = pd.Series(equity_values)
                rolling_max = equity_series.expanding().max()
                drawdowns = (equity_series - rolling_max) / rolling_max * 100
                max_drawdown = drawdowns.min()
                
                # Calculate daily returns for Sharpe ratio
                daily_returns = equity_series.pct_change().dropna()
                if len(daily_returns) > 0:
                    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() != 0 else 0
                else:
                    sharpe_ratio = 0
            else:
                max_drawdown = 0
                sharpe_ratio = 0
            
            # Calculate additional metrics
            total_pnl = sum([t.pnl_amount for t in trade_history if t.pnl_amount is not None])
            expectancy = total_pnl / total_trades if total_trades > 0 else 0
            
            # Win/loss streaks
            streaks = []
            current_streak = 0
            streak_type = None  # 'W' for win, 'L' for loss
            
            for trade in trade_history:
                if trade.pnl_amount and trade.pnl_amount > 0:
                    if streak_type == 'W':
                        current_streak += 1
                    else:
                        if current_streak != 0:
                            streaks.append((streak_type, abs(current_streak)))
                        current_streak = 1
                        streak_type = 'W'
                elif trade.pnl_amount and trade.pnl_amount < 0:
                    if streak_type == 'L':
                        current_streak += 1
                    else:
                        if current_streak != 0:
                            streaks.append((streak_type, abs(current_streak)))
                        current_streak = 1
                        streak_type = 'L'
            
            if current_streak != 0:
                streaks.append((streak_type, abs(current_streak)))
            
            max_win_streak = max([s[1] for s in streaks if s[0] == 'W'], default=0)
            max_loss_streak = max([s[1] for s in streaks if s[0] == 'L'], default=0)
            
            return {
                'test_id': self.test_id,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': round(win_rate, 2),
                'total_pnl': round(total_pnl, 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'profit_factor': round(profit_factor, 2),
                'max_drawdown': round(max_drawdown, 2),
                'sharpe_ratio': round(sharpe_ratio, 4),
                'expectancy': round(expectancy, 2),
                'max_win_streak': max_win_streak,
                'max_loss_streak': max_loss_streak,
                'strategy_metrics': strategy_metrics,
                'calculated_at': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating comprehensive metrics: {e}")
            return {
                'test_id': self.test_id,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'expectancy': 0,
                'max_win_streak': 0,
                'max_loss_streak': 0,
                'strategy_metrics': {},
                'calculated_at': datetime.now()
            }
    
    def store_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """Store performance metrics in MongoDB"""
        try:
            if self.performance_metrics_collection is not None:
                self.performance_metrics_collection.insert_one(metrics)
                self.logger.info(f"📊 Performance metrics stored for test {metrics.get('test_id')}")
            else:
                self.logger.warning("⚠️ Performance metrics collection not available")
        except Exception as e:
            self.logger.error(f"Error storing performance metrics: {e}")
    
    def run_multiple_timeframe_diagnostic(self,
                                        symbol: str,
                                        timeframes: List[str],
                                        days_back: int) -> Dict[str, Any]:
        """Run diagnostic across multiple timeframes"""
        self.logger.info(f"🔄 Running multi-timeframe diagnostic for {symbol}")

        results = {}
        for tf in timeframes:
            self.logger.info(f"⏱️ Processing timeframe: {tf}")
            try:
                # Create a new test ID for each timeframe
                self.test_id = str(uuid.uuid4())
                result = self.run_comprehensive_diagnostic(symbol, tf, days_back)
                results[tf] = result
            except Exception as e:
                self.logger.error(f"❌ Error processing timeframe {tf}: {e}")
                results[tf] = {'error': str(e), 'timeframe': tf}

        total_results = {
            'symbol': symbol,
            'timeframes': timeframes,
            'combined_results': results,
            'run_at': datetime.now()
        }

        return total_results
    
    def run_multiple_asset_diagnostic(self,
                                    symbols: List[str],
                                    timeframe: str,
                                    days_back: int) -> Dict[str, Any]:
        """Run diagnostic across multiple assets"""
        self.logger.info(f"🔄 Running multi-asset diagnostic for {len(symbols)} symbols")

        results = {}
        for symbol in symbols:
            self.logger.info(f"📈 Processing asset: {symbol}")
            try:
                # Create a new test ID for each symbol
                self.test_id = str(uuid.uuid4())
                result = self.run_comprehensive_diagnostic(symbol, timeframe, days_back)
                results[symbol] = result
            except Exception as e:
                self.logger.error(f"❌ Error processing asset {symbol}: {e}")
                results[symbol] = {'error': str(e), 'symbol': symbol}

        total_results = {
            'symbols': symbols,
            'timeframe': timeframe,
            'combined_results': results,
            'run_at': datetime.now()
        }

        return total_results