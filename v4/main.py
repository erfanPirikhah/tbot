# main.py

import pandas as pd
import numpy as np
import logging
import sys
import os
from datetime import datetime, timedelta
import warnings
from typing import Dict, Any, Optional
import inspect

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initial settings
warnings.filterwarnings('ignore')

# Import project modules
from utils.logger import setup_logger, get_trade_logger, get_performance_logger
from data.data_fetcher import DataFetcher
from strategies.enhanced_rsi_strategy_v4 import EnhancedRsiStrategyV4
from backtest.enhanced_rsi_backtest_v4 import EnhancedRSIBacktestV4
from config.parameters import OPTIMIZED_PARAMS_V4, MARKET_CONDITION_PARAMS
from config.market_config import SYMBOL_MAPPING, TIMEFRAME_MAPPING, DEFAULT_CONFIG

class TradingBotV4:
    """Trading Bot Version 4 - Complete Integration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        # Default settings
        self.config = config or DEFAULT_CONFIG.copy()
        
        # Setup logger
        output_dir = self.config.get('output_dir', os.path.join("logs", "backtests"))
        self.output_dir = output_dir
        self.logger = setup_logger("trading_bot_v4", logging.INFO, log_to_file=True, log_to_console=True, log_dir=output_dir)
        self.trade_logger = get_trade_logger(log_dir=output_dir)
        self.performance_logger = get_performance_logger(log_dir=output_dir)
        
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
        """Filter out parameters that don't exist in strategy class"""
        try:
            # Get valid parameters from strategy class
            sig = inspect.signature(EnhancedRsiStrategyV4.__init__)
            valid_params = set(sig.parameters.keys())
            
            # Remove 'self' from valid params
            valid_params.discard('self')
            
            # Filter parameters
            filtered_params = {k: v for k, v in strategy_params.items() if k in valid_params}
            
            # Log removed parameters
            removed_params = set(strategy_params.keys()) - set(filtered_params.keys())
            if removed_params:
                self.logger.warning(f"⚠️ Removed incompatible parameters: {removed_params}")
            
            return filtered_params
            
        except Exception as e:
            self.logger.error(f"Error filtering strategy parameters: {e}")
            return strategy_params

    def initialize_strategy(self, strategy_params: Dict[str, Any] = None):
        """Initialize strategy"""
        try:
            if strategy_params is None:
                strategy_params = OPTIMIZED_PARAMS_V4
            
            # Filter out incompatible parameters
            filtered_params = self._filter_strategy_params(strategy_params)
            
            self.strategy = EnhancedRsiStrategyV4(**filtered_params)
            self.logger.info("✅ RSI Strategy Version 4 initialized")
            self.logger.info(f"📊 Parameters: RSI({filtered_params['rsi_period']}), Risk: {filtered_params['risk_per_trade']*100}%")
            
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
        strategy_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Run complete backtest"""
        try:
            self.logger.info("🎯 Starting complete backtest")
            
            # Initialize strategy
            if not self.initialize_strategy(strategy_params):
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
            
            while datetime.now() < end_time and self.is_running:
                iteration += 1
                
                try:
                    # Get new data
                    data = self.get_market_data(symbol, timeframe, limit=100)
                    
                    # Generate signal
                    signal = self.strategy.generate_signal(data, iteration)
                    
                    # Process signal
                    self._process_signal(signal, data)
                    
                    # Display status
                    if iteration % 10 == 0:
                        self._display_status(iteration)
                    
                    # Wait for next candle
                    self._wait_for_next_candle(timeframe)
                    
                except Exception as e:
                    self.logger.error(f"Error in iteration {iteration}: {e}")
                    continue
            
            self.logger.info("✅ Live simulation completed")
            
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
        """Wait for next candle"""
        try:
            # Calculate wait time based on timeframe
            wait_times = {
                'M1': 60, 'M5': 300, 'M15': 900, 'M30': 1800,
                'H1': 3600, 'H4': 14400, 'D1': 86400
            }
            
            wait_seconds = wait_times.get(timeframe, 3600)
            
            # In simulation, we only wait one second
            import time
            time.sleep(1)
            
        except Exception as e:
            self.logger.error(f"Error waiting for candle: {e}")

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
            print("6. Exit")
            
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
                
                bot.run_backtest(symbol=symbol, timeframe=timeframe, days_back=days_back)
                
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