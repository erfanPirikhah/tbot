# main.py

import pandas as pd
import numpy as np
import logging
import sys
import os
from datetime import datetime, timedelta
import warnings
from typing import Dict, Any, Optional, List
import inspect

# Add project paths
v4_dir = os.path.dirname(os.path.abspath(__file__))
if v4_dir not in sys.path:
    sys.path.insert(0, v4_dir)

# Initial settings
warnings.filterwarnings('ignore')

# Import project modules
from utils.logger import setup_logger, get_trade_logger, get_performance_logger, get_mongo_collection
from data.data_fetcher import DataFetcher
from strategies.enhanced_rsi_strategy_v4 import EnhancedRsiStrategyV4, PositionType
from strategies.ensemble_strategy_v4 import EnsembleRsiStrategyV4
from diagnostic.diagnostic_enhanced_rsi_strategy import DiagnosticEnhancedRsiStrategy
from backtest.enhanced_rsi_backtest_v4 import EnhancedRSIBacktestV4
from config.parameters import OPTIMIZED_PARAMS_V4, MARKET_CONDITION_PARAMS, get_best_params_for_timeframe
from config.market_config import SYMBOL_MAPPING, TIMEFRAME_MAPPING, DEFAULT_CONFIG

class TradingBotV4:
    """Trading Bot Version 4 - Complete Integration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        # Default settings
        self.config = config or DEFAULT_CONFIG.copy()
        
        # Setup logger
        output_dir = self.config.get('output_dir', os.path.join("logs", "backtests"))
        self.output_dir = output_dir
        self.logger = setup_logger(
            "trading_bot_v4",
            logging.INFO,
            log_to_file=False,
            log_to_console=True,
            log_dir=output_dir,
            log_to_mongo=True,
            mongo_uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            mongo_db=os.getenv("TRADING_LOGS_DB", "trading_logs")
        )
        self.trade_logger = get_trade_logger(log_dir=output_dir)
        self.performance_logger = get_performance_logger(log_dir=output_dir)

        # Live output logging routed to MongoDB (no files)
        self.live_output_dir = os.path.join("logs", "live")  # kept for backward compat, not used
        self.live_log_path = None
        # Session and live collections
        self.session_id = None
        self.live_timeframe = None
        self.live_data_source = None
        self.live_symbols: List[str] = []
        self.live_collection = None  # legacy single collection (unused)
        self.live_col_entries = None
        self.live_col_exits = None
        self.live_col_partials = None
        self.live_col_candidates = None
        self.live_col_status = None
        
        # Create instances
        self.data_fetcher = DataFetcher()
        self.strategy = None
        self.backtest_engine = None
        
        # Bot status
        self.is_running = False
        self.current_position = "OUT"
        self.portfolio_value = self.config.get('initial_capital', 10000.0)
        
        self.logger.info("🤖 Trading Bot Version 4 initialized")
        self.logger.info(f"💰 Initial capital: ${self.portfolio_value:,.2f}")

    def _filter_strategy_params(self, strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out parameters that don't exist in the selected strategy class."""
        try:
            cls_name = (strategy_params or {}).get('strategy_class', 'EnhancedRsiStrategyV4')
            target_cls = EnhancedRsiStrategyV4
            if cls_name == 'EnsembleRsiStrategyV4':
                target_cls = EnsembleRsiStrategyV4
            elif cls_name == 'DiagnosticEnhancedRsiStrategy':
                target_cls = DiagnosticEnhancedRsiStrategy

            # Build valid parameter set from selected class
            sig = inspect.signature(target_cls.__init__)
            valid_params = set(sig.parameters.keys())
            valid_params.discard('self')
            # Remove diagnostic_system parameter if it's not needed for regular strategies
            if cls_name != 'DiagnosticEnhancedRsiStrategy':
                valid_params.discard('diagnostic_system')

            # Filter map to selected class signature
            filtered_params = {k: v for k, v in (strategy_params or {}).items() if k in valid_params}

            # Report removals excluding meta keys like 'strategy_class'
            removed_params = set((strategy_params or {}).keys()) - set(filtered_params.keys())
            removed_params_without_meta = removed_params - {'strategy_class'}
            if removed_params_without_meta:
                self.logger.warning(f"⚠️ Removed incompatible parameters: {removed_params_without_meta}")

            return filtered_params

        except Exception as e:
            self.logger.error(f"Error filtering strategy parameters: {e}")
            return strategy_params or {}

    def _infer_profile_name_from_timeframe(self, timeframe: str) -> str:
        """Infer profile name used by get_best_params_for_timeframe for logging/diagnostics."""
        tf = (timeframe or '').upper()
        if tf in ['M1', 'M5']:
            return 'ENSEMBLE_SCALPING_M5'
        elif tf in ['M15', 'M30']:
            return 'ENSEMBLE_INTRADAY_M15'
        elif tf == 'H1':
            return 'ENHANCED_INTRADAY_H1'
        elif tf == 'H4':
            return 'OPTIMIZED_PARAMS_V4'
        else:
            return 'CONSERVATIVE_PARAMS'

    def initialize_strategy(self, strategy_params: Dict[str, Any] = None, use_diagnostic: bool = False):
        """Initialize strategy (supports Ensemble via 'strategy_class')."""
        try:
            if strategy_params is None:
                strategy_params = OPTIMIZED_PARAMS_V4

            cls_name = (strategy_params or {}).get('strategy_class', 'EnhancedRsiStrategyV4')

            # If specifically told to use diagnostic or if the class name requires it
            if use_diagnostic or cls_name == 'DiagnosticEnhancedRsiStrategy':
                from diagnostic.diagnostic_system import DiagnosticSystem
                diagnostic_system = DiagnosticSystem()
                # Filter params to remove diagnostic_system if it's already in filtered_params
                filtered_params = self._filter_strategy_params(strategy_params)
                filtered_params['diagnostic_system'] = diagnostic_system
                self.strategy = DiagnosticEnhancedRsiStrategy(**filtered_params)
                self.logger.info("✅ Diagnostic Enhanced RSI Strategy V4 initialized")
            else:
                filtered_params = self._filter_strategy_params(strategy_params)

                if cls_name == 'EnsembleRsiStrategyV4':
                    self.strategy = EnsembleRsiStrategyV4(**filtered_params)
                    self.logger.info("✅ Ensemble RSI Strategy V4 initialized")
                else:
                    self.strategy = EnhancedRsiStrategyV4(**filtered_params)
                    self.logger.info("✅ RSI Strategy Version 4 initialized")

            # Common param logging if available
            rsi_period = filtered_params.get('rsi_period')
            risk = filtered_params.get('risk_per_trade')
            if rsi_period is not None and risk is not None:
                self.logger.info(f"📊 Parameters: RSI({rsi_period}), Risk: {risk*100}%")

            return True

        except Exception as e:
            self.logger.error(f"❌ Error initializing strategy: {e}")
            return False

    def initialize_backtest(self, backtest_params: Dict[str, Any] = None):
        """Initialize backtest engine"""
        try:
            backtest_config = {
                'initial_capital': self.config.get('initial_capital', 10000.0),
                'commission': self.config.get('commission', 0.0003),
                'slippage': self.config.get('slippage', 0.0001),
                'enable_plotting': True,
                'detailed_logging': True,
                'save_trade_logs': True,
                'output_dir': self.config.get('output_dir', os.path.join("logs", "backtests"))
            }
            
            if backtest_params:
                backtest_config.update(backtest_params)
            
            self.backtest_engine = EnhancedRSIBacktestV4(**backtest_config)
            self.logger.info("✅ Backtest engine initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing backtest: {e}")
            return False

    def test_connections(self) -> bool:
        """Test connections to data sources"""
        self.logger.info("🔌 Testing data source connections...")
        
        try:
            # Test MT5
            from data.mt5_data import MT5_AVAILABLE
            if MT5_AVAILABLE:
                mt5_ok = self.data_fetcher.test_connection("MT5")
                if mt5_ok:
                    self.logger.info("✅ MT5 connection: Successful")
                else:
                    self.logger.warning("⚠️ MT5 connection: Failed")
            
            # Test CryptoCompare
            crypto_ok = self.data_fetcher.test_connection("CRYPTOCOMPARE")
            if crypto_ok:
                self.logger.info("✅ CryptoCompare connection: Successful")
            else:
                self.logger.warning("⚠️ CryptoCompare connection: Failed")
            
            return mt5_ok or crypto_ok
            
        except Exception as e:
            self.logger.error(f"❌ Connection test error: {e}")
            return False

    def get_market_data(
        self, 
        symbol: str, 
        timeframe: str, 
        limit: int = 100,
        data_source: str = "AUTO"
    ) -> pd.DataFrame:
        """Get market data"""
        try:
            self.logger.info(f"📥 Getting data for {symbol} (Timeframe: {timeframe})")
            
            data = self.data_fetcher.fetch_market_data(
                symbol=symbol,
                interval=timeframe,
                limit=limit,
                data_source=data_source
            )
            
            if data.empty:
                raise ValueError("No data received")
            
            self.logger.info(f"✅ Received {len(data)} candles for {symbol}")
            self.logger.info(f"💰 Price range: {data['close'].min():.4f} - {data['close'].max():.4f}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Error getting data: {e}")
            raise

    def run_backtest(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "H1",
        days_back: int = 90,
        strategy_params: Dict[str, Any] = None,
        use_diagnostic: bool = False
    ) -> Dict[str, Any]:
        """Run complete backtest"""
        try:
            self.logger.info("🎯 Starting complete backtest")
            if use_diagnostic:
                self.logger.info("🔬 Running in diagnostic mode with comprehensive logging")

            # Auto-select params by timeframe when not provided (enables Ensemble on M5/M15)
            if strategy_params is None:
                try:
                    strategy_params = get_best_params_for_timeframe(timeframe)
                    profile_name = self._infer_profile_name_from_timeframe(timeframe)
                    self.logger.info(f"🧭 Auto-selected profile for {timeframe}: {profile_name} | strategy_class={strategy_params.get('strategy_class','EnhancedRsiStrategyV4')}")
                except Exception as e:
                    self.logger.warning(f"Param auto-selection failed, falling back to defaults: {e}")
                    strategy_params = OPTIMIZED_PARAMS_V4

            # Initialize strategy with diagnostic support if requested
            if use_diagnostic:
                strategy_params = strategy_params or OPTIMIZED_PARAMS_V4
                # Force diagnostic strategy if requested
                strategy_params = strategy_params.copy()
                strategy_params['strategy_class'] = 'DiagnosticEnhancedRsiStrategy'
            if not self.initialize_strategy(strategy_params, use_diagnostic=use_diagnostic):
                raise Exception("Error initializing strategy")

            # Initialize backtest
            if not self.initialize_backtest():
                raise Exception("Error initializing backtest engine")

            # Run backtest
            results = self.backtest_engine.run_backtest(
                symbol=symbol,
                timeframe=timeframe,
                days_back=days_back,
                strategy_params=strategy_params
            )

            # Display results
            report = self.backtest_engine.generate_comprehensive_report()
            print("\n" + "="*80)
            print("Backtest Results")
            print("="*80)
            print(report)

            # Save results
            self._save_backtest_results(results, symbol, timeframe)

            return results

        except Exception as e:
            self.logger.error(f"❌ Error running backtest: {e}")
            raise

    def run_live_simulation(
        self,
        symbol: str,
        timeframe: str,
        duration_hours: int = 24
    ):
        """Live simulation"""
        try:
            self.logger.info(f"🔄 Starting live simulation for {symbol}")
            
            if not self.initialize_strategy():
                raise Exception("Strategy not initialized")
            
            end_time = datetime.now() + timedelta(hours=duration_hours)
            iteration = 0
            # Initialize live log collections (single-symbol session)
            self._init_live_trade_log(timeframe, [symbol], data_source="AUTO")
            
            while datetime.now() < end_time and self.is_running:
                iteration += 1
                
                try:
                    # Get new data
                    data = self.get_market_data(symbol, timeframe, limit=100)
                    
                    # Generate signal
                    signal = self.strategy.generate_signal(data, iteration)
                    
                    # Process signal
                    self._process_signal(signal, data)

                    # Calculate and display real-time PnL if position is open
                    pnl_data = self._calculate_real_time_pnl(data)
                    if pnl_data:
                        self._display_real_time_pnl(pnl_data)

                    # Persist live entry/exit to dedicated collections
                    try:
                        action = signal.get('action', 'HOLD')
                        rec = None
                        if action in ('BUY', 'SELL'):
                            rec = {
                                "timestamp": data.index[-1],
                                "type": "ENTRY",
                                "symbol": symbol,
                                "side": "LONG" if action == "BUY" else "SHORT",
                                "price": float(signal.get('price', data['close'].iloc[-1])),
                                "position_size": signal.get('position_size'),
                                "stop_loss": signal.get('stop_loss'),
                                "take_profit": signal.get('take_profit'),
                                "reason": signal.get('reason', ''),
                                "conditions": [],
                            }
                        elif action in ('EXIT', 'PARTIAL_EXIT'):
                            rec = {
                                "timestamp": data.index[-1],
                                "type": "EXIT" if action == "EXIT" else "PARTIAL_EXIT",
                                "symbol": symbol,
                                "price": float(signal.get('price', data['close'].iloc[-1])),
                                "pnl_percentage": signal.get('pnl_percentage'),
                                "pnl_amount": signal.get('pnl_amount'),
                                "exit_reason": signal.get('exit_reason', signal.get('reason', '')),
                                "outcome": ("WIN" if (signal.get('pnl_percentage') or 0) >= 0 else "LOSS")
                            }
                        if rec:
                            self._append_live_trade_log(rec)
                    except Exception:
                        pass
                    
                    # Display status
                    if iteration % 10 == 0:
                        self._display_status(iteration)

                    # Persist status snapshot (single-symbol)
                    try:
                        open_positions = 0 if self.current_position == "OUT" else 1
                        self._log_live_status(iteration, open_positions, 0)
                    except Exception:
                        pass
                    
                    # Wait for next candle
                    self._wait_for_next_candle(timeframe)
                    
                except Exception as e:
                    self.logger.error(f"Error in iteration {iteration}: {e}")
                    continue
            
            self.logger.info("✅ Live simulation completed")
            self.logger.info("📄 Live data saved to MongoDB collections: live_entries, live_exits, live_partial_exits, live_status")
            
        except Exception as e:
            self.logger.error(f"❌ Live simulation error: {e}")
            raise

    def _process_signal(self, signal: Dict[str, Any], data: pd.DataFrame):
        """Process received signal"""
        try:
            action = signal.get('action', 'HOLD')
            current_price = data['close'].iloc[-1]
            
            if action == 'BUY':
                self.trade_logger.info(f"🎯 BUY signal at {current_price:.4f}")
                self.current_position = "LONG"
                
            elif action == 'SELL':
                self.trade_logger.info(f"🎯 SELL signal at {current_price:.4f}")
                self.current_position = "SHORT"
                
            elif action == 'EXIT':
                pnl = signal.get('pnl_percentage', 0)
                reason = signal.get('exit_reason', '')
                self.trade_logger.info(f"🔚 Trade exit - PnL: {pnl:.2f}% - Reason: {reason}")
                self.current_position = "OUT"
                
            elif action == 'PARTIAL_EXIT':
                pnl = signal.get('pnl_percentage', 0)
                self.trade_logger.info(f"📦 Partial exit - PnL: {pnl:.2f}%")
                
            # Update portfolio value
            if 'portfolio_value' in signal:
                self.portfolio_value = signal['portfolio_value']
                
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")

    def _calculate_strategy_real_time_pnl(self, strategy, data: pd.DataFrame, symbol: str) -> Optional[Dict[str, Any]]:
        """Calculate real-time PnL for a single strategy"""
        try:
            if (getattr(strategy, "_position", None) and 
                strategy._position != PositionType.OUT and 
                getattr(strategy, "_current_trade", None)):
                
                current_price = data['close'].iloc[-1]
                entry_price = strategy._current_trade.entry_price
                position_type = strategy._position

                if position_type == PositionType.LONG:
                    pnl_percentage = ((current_price - entry_price) / entry_price) * 100
                else:  # SHORT
                    pnl_percentage = ((entry_price - current_price) / entry_price) * 100

                # Calculate dollar PnL based on position size
                position_size = strategy._current_trade.quantity
                pnl_amount = pnl_percentage * (position_size * entry_price) / 100

                return {
                    'symbol': symbol,
                    'current_price': current_price,
                    'entry_price': entry_price,
                    'position_type': position_type.value,
                    'pnl_percentage': pnl_percentage,
                    'pnl_amount': pnl_amount,
                    'stop_loss': strategy._current_trade.stop_loss,
                    'take_profit': strategy._current_trade.take_profit,
                    'time_in_position': len(data) - strategy._last_trade_index
                }
        except Exception as e:
            self.logger.error(f"Error calculating strategy real-time PnL: {e}")
            return None

    def _display_strategy_real_time_pnl(self, pnl_data: Dict[str, Any]):
        """Display real-time PnL information for a single strategy"""
        try:
            symbol = pnl_data['symbol']
            position_type = pnl_data['position_type']
            pnl_percentage = pnl_data['pnl_percentage']
            pnl_amount = pnl_data['pnl_amount']
            current_price = pnl_data['current_price']
            entry_price = pnl_data['entry_price']
            stop_loss = pnl_data['stop_loss']
            take_profit = pnl_data['take_profit']
            time_in_pos = pnl_data['time_in_position']

            pnl_sign = "🟢" if pnl_percentage >= 0 else "🔴"
            pnl_status = f"{pnl_sign} PnL: {pnl_percentage:+.2f}% (${pnl_amount:+.2f})"
            
            self.logger.info(f"📈 {symbol} {position_type} | {pnl_status} | Price: {current_price:.4f} | Entry: {entry_price:.4f} | SL: {stop_loss:.4f} | TP: {take_profit:.4f} | Bars: {time_in_pos}")
        except Exception as e:
            self.logger.error(f"Error displaying strategy real-time PnL: {e}")

    def _calculate_real_time_pnl(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Calculate real-time PnL for open positions"""
        try:
            if self.current_position == "OUT" or self.strategy is None or self.strategy._current_trade is None:
                return None

            current_price = data['close'].iloc[-1]
            entry_price = self.strategy._current_trade.entry_price
            position_type = self.strategy._position

            if position_type == PositionType.LONG:
                pnl_percentage = ((current_price - entry_price) / entry_price) * 100
            else:  # SHORT
                pnl_percentage = ((entry_price - current_price) / entry_price) * 100

            # Calculate dollar PnL based on position size
            position_size = self.strategy._current_trade.quantity
            pnl_amount = pnl_percentage * (position_size * entry_price) / 100

            return {
                'current_price': current_price,
                'entry_price': entry_price,
                'position_type': position_type.value,
                'pnl_percentage': pnl_percentage,
                'pnl_amount': pnl_amount,
                'stop_loss': self.strategy._current_trade.stop_loss,
                'take_profit': self.strategy._current_trade.take_profit,
                'time_in_position': len(data) - self.strategy._last_trade_index
            }
        except Exception as e:
            self.logger.error(f"Error calculating real-time PnL: {e}")
            return None

    def _display_real_time_pnl(self, pnl_data: Dict[str, Any]):
        """Display real-time PnL information"""
        try:
            position_type = pnl_data['position_type']
            pnl_percentage = pnl_data['pnl_percentage']
            pnl_amount = pnl_data['pnl_amount']
            current_price = pnl_data['current_price']
            entry_price = pnl_data['entry_price']
            stop_loss = pnl_data['stop_loss']
            take_profit = pnl_data['take_profit']
            time_in_pos = pnl_data['time_in_position']

            pnl_sign = "🟢" if pnl_percentage >= 0 else "🔴"
            pnl_status = f"{pnl_sign} PnL: {pnl_percentage:+.2f}% (${pnl_amount:+.2f})"
            
            self.logger.info(f"📈 {position_type} Position | {pnl_status} | Price: {current_price:.4f} | Entry: {entry_price:.4f} | SL: {stop_loss:.4f} | TP: {take_profit:.4f} | Bars: {time_in_pos}")
        except Exception as e:
            self.logger.error(f"Error displaying real-time PnL: {e}")

    def _display_status(self, iteration: int):
        """Display current status"""
        try:
            metrics = self.strategy.get_performance_metrics()
            
            status = f"""
📊 System Status - Iteration {iteration}
├── Current Position: {self.current_position}
├── Portfolio Value: ${self.portfolio_value:,.2f}
├── Total Trades: {metrics.get('total_trades', 0)}
├── Win Rate: {metrics.get('win_rate', 0):.1f}%
└── Total PnL: ${metrics.get('total_pnl', 0):,.2f}
            """
            
            self.logger.info(status)
            
        except Exception as e:
            self.logger.error(f"Error displaying status: {e}")

    def _wait_for_next_candle(self, timeframe: str):
        """Wait for next candle (realistic sleep for live modes)"""
        try:
            # Calculate wait time based on timeframe
            wait_times = {
                'M1': 60, '1M': 60, '1m': 60,
                'M5': 300, '5M': 300, '5m': 300,
                'M15': 900, '15m': 900, 'M30': 1800,
                'H1': 3600, '1h': 3600, 'H4': 14400,
                'D1': 86400
            }
            import time
            wait_seconds = wait_times.get(timeframe, 60)
            time.sleep(wait_seconds)
        except Exception as e:
            self.logger.error(f"Error waiting for candle: {e}")

    def _init_live_trade_log(self, timeframe: str, symbols: List[str], data_source: str = "AUTO") -> str:
        """Initialize separate MongoDB collections for live logging and set session metadata."""
        try:
            # Session metadata
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.live_timeframe = timeframe
            self.live_data_source = data_source
            self.live_symbols = list(symbols or [])

            # Collections
            self.live_col_entries = get_mongo_collection("live_entries")
            self.live_col_exits = get_mongo_collection("live_exits")
            self.live_col_partials = get_mongo_collection("live_partial_exits")
            self.live_col_candidates = get_mongo_collection("live_candidates")
            self.live_col_status = get_mongo_collection("live_status")

            self.live_log_path = None  # no file fallback by requirement

            self.logger.info(
                f"📝 Live logging session initialized | session_id={self.session_id} timeframe={timeframe} "
                f"collections=[live_entries, live_exits, live_partial_exits, live_candidates, live_status]"
            )
            return "OK"
        except Exception as e:
            self.logger.error(f"Error initializing live log collections: {e}")
            # Null out collections on failure
            self.live_col_entries = None
            self.live_col_exits = None
            self.live_col_partials = None
            self.live_col_candidates = None
            self.live_col_status = None
            return ""

    def _append_live_trade_log(self, record: Dict[str, Any]) -> None:
        """Route a single live record to its dedicated MongoDB collection."""
        try:
            rec_type = (record or {}).get("type", "").upper()
            # Ensure collections initialized
            if any(getattr(self, attr, None) is None for attr in [
                "live_col_entries", "live_col_exits", "live_col_partials", "live_col_candidates", "live_col_status"
            ]):
                # attempt lazy init with last known metadata
                self._init_live_trade_log(self.live_timeframe or "M1", getattr(self, "live_symbols", []), self.live_data_source or "AUTO")

            col = None
            if rec_type == "ENTRY":
                col = self.live_col_entries
            elif rec_type == "EXIT":
                col = self.live_col_exits
            elif rec_type == "PARTIAL_EXIT":
                col = self.live_col_partials
            elif rec_type == "CANDIDATES":
                col = self.live_col_candidates
            elif rec_type == "STATUS":
                col = self.live_col_status

            if col is None:
                return

            doc = self._convert_to_serializable(record)
            # Enrich with session metadata
            doc["session_id"] = getattr(self, "session_id", None)
            doc["timeframe"] = getattr(self, "live_timeframe", None)
            doc["data_source"] = getattr(self, "live_data_source", None)
            doc["ts"] = datetime.now().isoformat()
            col.insert_one(doc)
        except Exception as e:
            self.logger.error(f"Error writing live record to MongoDB: {e}")

    def _log_live_candidates(self, candidates: List[Dict[str, Any]], iteration: int) -> None:
        """Persist live candidates snapshot for this iteration into 'live_candidates' collection."""
        try:
            payload = {
                "type": "CANDIDATES",
                "iteration": iteration,
                "session_id": getattr(self, "session_id", None),
                "timeframe": getattr(self, "live_timeframe", None),
                "data_source": getattr(self, "live_data_source", None),
                "total": len(candidates or []),
                "top": sorted(
                    [c for c in candidates or []],
                    key=lambda x: x.get("strength", 0.0),
                    reverse=True
                )[:10],
            }
            self._append_live_trade_log(payload)
        except Exception:
            pass

    def _log_live_status(self, iteration: int, open_positions: int, candidates_count: int) -> None:
        """Persist status snapshot into 'live_status' collection."""
        try:
            payload = {
                "type": "STATUS",
                "iteration": iteration,
                "open_positions": int(open_positions),
                "candidates_count": int(candidates_count),
            }
            self._append_live_trade_log(payload)
        except Exception:
            pass

    def _fetch_with_fallback(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """Fetch market data with timeframe fallback across sources (M1<->1m, M5<->5m)."""
        try:
            return self.get_market_data(symbol, timeframe, limit=limit)
        except Exception:
            try:
                tf_map = {'M1': '1m', 'M5': '5m', '1m': 'M1', '5m': 'M5'}
                alt_tf = tf_map.get(timeframe, timeframe)
                return self.get_market_data(symbol, alt_tf, limit=limit)
            except Exception as e2:
                self.logger.error(f"❌ Data fetch failed for {symbol} ({timeframe}): {e2}")
                return pd.DataFrame()

    def _check_entry_conditions(self, strat, df: pd.DataFrame, position_type: PositionType):
        """
        Safe wrapper for strategy entry gating. Uses strategy.check_entry_conditions when present,
        otherwise falls back to ensemble-style score aggregation and gating.
        Returns (eligible: bool, messages: List[str])
        """
        try:
            # Preferred path if strategy exposes the method
            if hasattr(strat, "check_entry_conditions"):
                return strat.check_entry_conditions(df, position_type)

            # Fallback for Ensemble strategy: compute features and aggregate scores
            if hasattr(strat, "_ensure_min_features"):
                try:
                    df = strat._ensure_min_features(df)
                except Exception:
                    pass

            # Basic pre-checks (session and spacing)
            try:
                now = df.index[-1]
                if hasattr(strat, "_in_active_session") and not strat._in_active_session(now):
                    return False, ["Outside active session"]
            except Exception:
                pass

            candles_since_last = len(df) - 1 - getattr(strat, "_last_trade_index", -100)
            min_between = getattr(strat, "min_candles_between", 3)
            if candles_since_last < min_between:
                return False, [f"Spacing {candles_since_last}<{min_between}"]

            # Aggregate scores if available
            if hasattr(strat, "_aggregate_scores"):
                long_score, short_score, msgs = strat._aggregate_scores(df)
            else:
                return False, ["No gating available"]

            entry_threshold = getattr(strat, "entry_threshold", 1.6)
            dominance_ratio = getattr(strat, "dominance_ratio", 1.15)
            enable_short = getattr(strat, "enable_short_trades", True)

            if position_type == PositionType.LONG:
                eligible = (long_score >= entry_threshold) and (long_score >= short_score * dominance_ratio)
            else:
                eligible = enable_short and (short_score >= entry_threshold) and (short_score >= long_score * dominance_ratio)

            return bool(eligible), msgs

        except Exception as e:
            self.logger.error(f"_check_entry_conditions fallback error: {e}")
            return False, [f"Error: {e}"]

    def run_multi_symbol_live_scan(
        self,
        symbols: List[str],
        timeframe: str = "M1",
        duration_hours: int = 1,
        data_source: str = "AUTO",
        max_open_positions: int = 1
    ):
        """
        Live scan across multiple symbols (M1/M5) for 1-2 hours, pick best entries, and log trades.
        - Scans at least 5-10 symbols.
        - Opens position on the symbol with best conditions (by RSI gate strength).
        - Logs all entries/exits to a separate JSONL file, including PnL and outcome.
        """
        try:
            if not symbols or len(symbols) < 1:
                raise ValueError("No symbols provided")

            # Clamp duration to 1..2 hours as requested
            duration_hours = max(1, min(2, int(duration_hours or 1)))

            # Select parameters per timeframe (M1/M5 -> Ensemble by design)
            try:
                base_params = get_best_params_for_timeframe(timeframe)
            except Exception:
                # fallback to M5 ensemble if unknown
                base_params = get_best_params_for_timeframe("M5")

            # Strategy per symbol (isolated state per market)
            strategies: Dict[str, Any] = {}
            for sym in symbols:
                params = dict(base_params)  # copy
                cls_name = params.get('strategy_class', 'EnhancedRsiStrategyV4')
                filtered = self._filter_strategy_params(params)
                if cls_name == 'EnsembleRsiStrategyV4':
                    strategies[sym] = EnsembleRsiStrategyV4(**filtered)
                else:
                    strategies[sym] = EnhancedRsiStrategyV4(**filtered)
                self.logger.info(f"✅ Strategy initialized for {sym} | {cls_name}")

            # Init live log collections
            self._init_live_trade_log(timeframe, symbols, data_source)

            self.is_running = True
            end_time = datetime.now() + timedelta(hours=duration_hours)
            iteration = 0

            # Simple cache for last fetched data to avoid double-fetch on execution
            data_cache: Dict[str, pd.DataFrame] = {}

            while datetime.now() < end_time and self.is_running:
                iteration += 1
                candidates: List[Dict[str, Any]] = []

                # 1) Process exits for active positions; 2) Build entry candidates for flat symbols
                for sym, strat in strategies.items():
                    try:
                        df = self._fetch_with_fallback(sym, timeframe, limit=200)
                        if df is None or df.empty:
                            continue
                        data_cache[sym] = df

                        # Ensure RSI for evaluation
                        try:
                            if 'RSI' not in df.columns:
                                # Access strategy's RSI calc if needed
                                df = strat._calculate_rsi(df)  # type: ignore
                        except Exception:
                            pass

                        # If in position -> let strategy manage exits
                        if getattr(strat, "_position", None) and strat._position != PositionType.OUT:
                            signal = strat.generate_signal(df, iteration)
                            action = signal.get('action')
                            if action in ('EXIT', 'PARTIAL_EXIT'):
                                # Log exit (with PnL)
                                record = {
                                    "timestamp": df.index[-1],
                                    "type": "EXIT" if action == "EXIT" else "PARTIAL_EXIT",
                                    "symbol": sym,
                                    "price": float(signal.get('price', df['close'].iloc[-1])),
                                    "pnl_percentage": signal.get('pnl_percentage'),
                                    "pnl_amount": signal.get('pnl_amount'),
                                    "exit_reason": signal.get('exit_reason', signal.get('reason', '')),
                                    "outcome": ("WIN" if (signal.get('pnl_percentage') or 0) >= 0 else "LOSS")
                                }
                                self._append_live_trade_log(record)
                            continue

                        # If flat -> evaluate entry strength for LONG/SHORT
                        current_rsi = float(df['RSI'].iloc[-1]) if 'RSI' in df.columns else 50.0

                        # Check LONG
                        ok_long, cond_long = self._check_entry_conditions(strat, df, PositionType.LONG)
                        if ok_long:
                            strength_long = (getattr(strat, 'rsi_oversold', 35) + getattr(strat, 'rsi_entry_buffer', 5)) - current_rsi
                            candidates.append({
                                "symbol": sym,
                                "side": "LONG",
                                "strength": float(max(0.0, strength_long)),
                                "conditions": cond_long
                            })

                        # Check SHORT (if enabled)
                        if getattr(strat, 'enable_short_trades', True):
                            ok_short, cond_short = self._check_entry_conditions(strat, df, PositionType.SHORT)
                            if ok_short:
                                strength_short = current_rsi - (getattr(strat, 'rsi_overbought', 65) - getattr(strat, 'rsi_entry_buffer', 5))
                                candidates.append({
                                    "symbol": sym,
                                    "side": "SHORT",
                                    "strength": float(max(0.0, strength_short)),
                                    "conditions": cond_short
                                })

                    except Exception as e:
                        self.logger.error(f"Scan error for {sym}: {e}")
                        continue

                # Persist candidates for this iteration
                try:
                    self._log_live_candidates(candidates, iteration)
                except Exception:
                    pass

                # Determine capacity
                open_positions = sum(1 for s in strategies.values() if getattr(s, "_position", None) != PositionType.OUT)
                capacity = max(0, max_open_positions - open_positions)

                # Pick best candidates by strength
                if capacity > 0 and candidates:
                    candidates.sort(key=lambda x: x['strength'], reverse=True)
                    selected = candidates[:capacity]

                    for pick in selected:
                        sym = pick['symbol']
                        df = data_cache.get(sym)
                        if df is None or df.empty:
                            df = self._fetch_with_fallback(sym, timeframe, limit=200)
                        if df is None or df.empty:
                            continue
                        strat = strategies[sym]
                        signal = strat.generate_signal(df, iteration)
                        action = signal.get('action')
                        if action in ('BUY', 'SELL'):
                            # Log entry
                            record = {
                                "timestamp": df.index[-1],
                                "type": "ENTRY",
                                "symbol": sym,
                                "side": "LONG" if action == "BUY" else "SHORT",
                                "price": float(signal.get('price', df['close'].iloc[-1])),
                                "position_size": signal.get('position_size'),
                                "stop_loss": signal.get('stop_loss'),
                                "take_profit": signal.get('take_profit'),
                                "reason": signal.get('reason', ''),
                                "conditions": pick.get('conditions', []),
                            }
                            self._append_live_trade_log(record)
                            self.trade_logger.info(f"🎯 {action} {sym} @ {record['price']:.5f}")

                # Calculate and display real-time PnL for each open position
                for sym, strat in strategies.items():
                    df = data_cache.get(sym)
                    if df is not None and not df.empty:
                        pnl_data = self._calculate_strategy_real_time_pnl(strat, df, sym)
                        if pnl_data:
                            self._display_strategy_real_time_pnl(pnl_data)

                # Status every iteration
                total_open = sum(1 for s in strategies.values() if getattr(s, "_position", None) != PositionType.OUT)
                self.logger.info(f"⏱️ Iteration {iteration} | Open positions: {total_open} | Candidates: {len(candidates)}")

                # Persist status snapshot
                try:
                    self._log_live_status(iteration, total_open, len(candidates))
                except Exception:
                    pass

                # Wait for next candle based on timeframe
                self._wait_for_next_candle(timeframe)

            self.logger.info("✅ Multi-symbol live scan completed")
            self.logger.info("📄 Live data saved to MongoDB collections: live_entries, live_exits, live_partial_exits, live_candidates, live_status")

        except Exception as e:
            self.logger.error(f"❌ Multi-symbol live scan error: {e}")
        finally:
            self.is_running = False

    def _convert_to_serializable(self, obj):
        """Convert non-serializable objects to serializable formats"""
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif hasattr(obj, 'dtype'):  # Handle other numpy types
            return float(obj)
        else:
            return obj

    def _save_backtest_results(self, results: Dict[str, Any], symbol: str, timeframe: str):
        """Save backtest results with proper serialization"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_root = getattr(self.backtest_engine, 'output_dir', os.path.join("logs", "backtests"))
            results_dir = os.path.join(output_root, "results")
            os.makedirs(results_dir, exist_ok=True)
            filename = os.path.join(results_dir, f"backtest_results_{symbol}_{timeframe}_{timestamp}.json")
            
            # Convert to savable format with proper serialization
            save_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': timestamp,
                'performance_metrics': self._convert_to_serializable(results.get('performance_metrics', {})),
                'data_info': self._convert_to_serializable(results.get('data_info', {})),
                'strategy_metrics': self._convert_to_serializable(results.get('strategy_metrics', {}))
            }
            
            import json
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"💾 Results saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

    def optimize_h1_parameters(self, symbol: str = "EURUSD", days_back: int = 60) -> Dict[str, Any]:
        """Run a small optimization grid for H1 to improve win rate/return, then backtest best."""
        try:
            # Ensure backtest engine
            if not self.initialize_backtest():
                raise Exception("Error initializing backtest engine")

            # Fetch data via backtest engine for consistency
            data = self.backtest_engine.fetch_real_data_from_mt5(symbol, "H1", days_back)

            # Focused H1 grid (keep MTF disabled to avoid over-filtering)
            param_grid = {
                'rsi_entry_buffer': [3, 4, 5],
                'stop_loss_atr_multiplier': [1.4, 1.6, 1.8],
                'take_profit_ratio': [1.6, 1.8, 2.0],
                'trailing_activation_percent': [1.0, 1.2, 1.5],
                'min_candles_between': [3, 4, 5],
                'enable_mtf': [False],
                'enable_trend_filter': [True],
                'risk_per_trade': [0.01]
            }

            best_params = self.backtest_engine.optimize_parameters(data, param_grid)
            self.logger.info(f"🔧 Best H1 params: {best_params}")

            # Confirmatory backtest with best params
            results = self.backtest_engine.run_backtest(
                symbol=symbol,
                timeframe="H1",
                days_back=days_back,
                strategy_params=best_params
            )

            report = self.backtest_engine.generate_comprehensive_report()
            print("\n" + "="*80)
            print("Optimized H1 Backtest Results")
            print("="*80)
            print(report)

            # Persist results summary
            self._save_backtest_results(results, symbol, "H1")
            return best_params

        except Exception as e:
            self.logger.error(f"❌ H1 optimization error: {e}")
            return {}

    def get_available_symbols(self, data_source: str = "MT5") -> list:
        """Get available symbols list"""
        try:
            symbols = self.data_fetcher.get_available_symbols(data_source)
            self.logger.info(f"📋 Number of {data_source} symbols: {len(symbols)}")
            return symbols
            
        except Exception as e:
            self.logger.error(f"Error getting symbols list: {e}")
            return []

    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get strategy performance"""
        if self.strategy:
            return self.strategy.get_performance_metrics()
        return {}

    def stop(self):
        """Stop bot"""
        self.is_running = False
        self.logger.info("🛑 Bot stopped")

def main():
    """Main function"""
    print("🤖 Trading Bot Version 4 - Advanced RSI System")
    print("=" * 60)
    
    # Create bot
    bot = TradingBotV4()
    
    try:
        # Test connections
        print("\n🔌 Testing connections...")
        bot.test_connections()
        
        while True:
            print("\n" + "=" * 60)
            print("Main Menu:")
            print("1. Quick Backtest (EURUSD - H1)")
            print("2. Custom Backtest")
            print("3. Live Simulation")
            print("4. Show Available Symbols")
            print("5. Test Data Fetching")
            print("6. Run Comprehensive Diagnostic Analysis")
            print("7. Multi-Symbol Live Scan (M1/M5)")
            print("8. Exit")
            
            choice = input("\nPlease select an option: ").strip()
            
            if choice == "1":
                # Quick backtest
                print("\n🎯 Running quick backtest...")
                bot.run_backtest(symbol="EURUSD", timeframe="H1", days_back=60)
                
            elif choice == "2":
                # Custom backtest
                print("\n🎯 Running custom backtest...")
                symbol = input("Symbol (default: EURUSD): ").strip() or "EURUSD"
                timeframe = input("Timeframe (default: H1): ").strip() or "H1"

                days = input("Days back (default: 90): ").strip()
                days_back = int(days) if days.isdigit() else 90

                diagnostic_choice = input("Run in diagnostic mode? (y/N): ").strip().lower()
                use_diagnostic = diagnostic_choice == 'y'

                bot.run_backtest(symbol=symbol, timeframe=timeframe, days_back=days_back, use_diagnostic=use_diagnostic)
                
            elif choice == "3":
                # Live simulation
                print("\n🔄 Starting live simulation...")
                symbol = input("Symbol (default: EURUSD): ").strip() or "EURUSD"
                timeframe = input("Timeframe (default: H1): ").strip() or "H1"
                
                bot.is_running = True
                bot.run_live_simulation(symbol=symbol, timeframe=timeframe, duration_hours=2)
                
            elif choice == "4":
                # Show symbols
                print("\n📋 Available symbols:")
                mt5_symbols = bot.get_available_symbols("MT5")
                crypto_symbols = bot.get_available_symbols("CRYPTOCOMPARE")
                
                print(f"MT5: {', '.join(mt5_symbols[:10])}{'...' if len(mt5_symbols) > 10 else ''}")
                print(f"Crypto: {', '.join(crypto_symbols[:10])}{'...' if len(crypto_symbols) > 10 else ''}")
                
            elif choice == "5":
                # Test data fetching
                print("\n📊 Testing data fetching...")
                symbol = input("Symbol (default: EURUSD): ").strip() or "EURUSD"
                timeframe = input("Timeframe (default: H1): ").strip() or "H1"
                
                try:
                    data = bot.get_market_data(symbol, timeframe, limit=10)
                    print(f"✅ Data received: {len(data)} candles")
                    print(f"Last price: {data['close'].iloc[-1]:.4f}")
                    print(f"RSI: {data['RSI'].iloc[-1]:.1f}" if 'RSI' in data.columns else "RSI: Not calculated")
                except Exception as e:
                    print(f"❌ Error: {e}")
            
            elif choice == "6":
                # Comprehensive Diagnostic Analysis
                print("\n🔬 Running comprehensive diagnostic analysis...")
                from diagnostic.main_diagnostic import run_complete_diagnostic_analysis
                try:
                    results = run_complete_diagnostic_analysis()
                    print("Diagnostic analysis completed successfully!")
                except Exception as e:
                    print(f"Error in diagnostic analysis: {e}")

            elif choice == "7":
                # Multi-symbol live scan (M1/M5)
                print("\n🔄 Starting multi-symbol live scan...")
                symbols_input = input("Symbols (comma separated, default: BTC,ETH,ADA,XRP,SOL): ").strip() or "BTC,ETH,ADA,XRP,SOL"
                symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
                timeframe = (input("Timeframe (M1 or M5, default: M1): ").strip() or "M1").upper()
                try:
                    hours = input("Duration hours (1-2, default: 1): ").strip()
                    duration_hours = int(hours) if hours.isdigit() else 1
                    duration_hours = max(1, min(2, duration_hours))
                except Exception:
                    duration_hours = 1
                bot.is_running = True
                bot.run_multi_symbol_live_scan(symbols=symbols, timeframe=timeframe, duration_hours=duration_hours)
                print("\n📄 Live data stored in MongoDB collections: live_entries, live_exits, live_partial_exits, live_candidates, live_status")

            elif choice == "8":
                # Exit
                print("\n👋 Goodbye!")
                bot.stop()
                break
                
            else:
                print("❌ Invalid option!")
                
    except KeyboardInterrupt:
        print("\n\n🛑 Program stopped by user")
        bot.stop()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        bot.stop()

if __name__ == "__main__":
    main()