# test_position_lifecycle.py
"""
Position Lifecycle Test File - TradeBot Pro
Version: 1.0.1 - Fixed timeframe and symbol mapping
Developer: Market Analysis Team
"""

import sys
import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.improved_advanced_rsi_strategy import ImprovedAdvancedRsiStrategy, PositionType, ExitReason
from data.data_fetcher import fetch_market_data, get_current_price
from indicators.rsi import calculate_rsi
from config import MT5_SYMBOL_MAP, CRYPTOCOMPARE_SYMBOL_MAP, TEST_STRATEGY_PARAMS, MT5_INTERVAL_MAP, CRYPTOCOMPARE_INTERVAL_MAP

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('position_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class PositionLifecycleTester:
    """Complete position lifecycle test class"""
    
    def __init__(self, data_source="MT5", symbol_display="Gold (XAUUSD)", interval="1h"):
        self.data_source = data_source
        self.symbol_display = symbol_display
        self.interval = interval
        self.symbol_code = self.get_symbol_code(symbol_display, data_source)
        
        if not self.symbol_code:
            raise ValueError(f"Symbol code not found for {symbol_display} in {data_source}")
        
        # Use test parameters that are more sensitive
        self.strategy = ImprovedAdvancedRsiStrategy(**TEST_STRATEGY_PARAMS)
        
        # Reduce trade duration for quick testing
        self.strategy.max_trade_duration = 5  # 5 minutes for testing
        
        self.test_results = {
            'position_opened': False,
            'position_closed': False,
            'entry_time': None,
            'exit_time': None,
            'entry_price': 0,
            'exit_price': 0,
            'pnl_percentage': 0,
            'exit_reason': None,
            'trade_duration_minutes': 0,
            'signal_strength': None,
            'errors': []
        }
        
        logger.info(f"🚀 Position tester initialized for {symbol_display} ({self.symbol_code}) from {data_source}")
    
    def get_symbol_code(self, symbol_display, data_source):
        """Get symbol code with proper mapping"""
        # Map English display names to Persian for lookup
        symbol_mapping = {
            "Gold (XAUUSD)": "طلا (XAUUSD)",
            "Bitcoin (BTC)": "بیت‌کوین (BTC)",
            "EUR/USD (EURUSD)": "یورو/دلار (EURUSD)",
            "Silver (XAGUSD)": "نقره (XAGUSD)",
            "Oil (XTIUSD)": "نفت (XTIUSD)"
        }
        
        # Convert English display name to Persian for lookup
        persian_symbol_display = symbol_mapping.get(symbol_display, symbol_display)
        
        if data_source == "MT5":
            return MT5_SYMBOL_MAP.get(persian_symbol_display)
        else:
            return CRYPTOCOMPARE_SYMBOL_MAP.get(persian_symbol_display)
    
    def get_interval_code(self, interval, data_source):
        """Get interval code with proper mapping"""
        # Map English interval names to Persian for lookup
        interval_mapping = {
            "1 minute": "۱ دقیقه",
            "5 minutes": "۵ دقیقه", 
            "15 minutes": "۱۵ دقیقه",
            "30 minutes": "۳۰ دقیقه",
            "1 hour": "۱ ساعت",
            "4 hours": "۴ ساعت", 
            "1 day": "۱ روز",
            "1 week": "۱ هفته"
        }
        
        # Convert English interval to Persian for lookup
        persian_interval = interval_mapping.get(interval, interval)
        
        if data_source == "MT5":
            return MT5_INTERVAL_MAP.get(persian_interval, "H1")
        else:
            return CRYPTOCOMPARE_INTERVAL_MAP.get(persian_interval, "1h")
    
    def create_test_data_with_signal(self, base_price=1800, trend="uptrend", rsi_level=25):
        """
        Create test data with specific signal
        
        Args:
            base_price: Base price
            trend: Market trend ('uptrend', 'downtrend', 'sideways')
            rsi_level: RSI level (25 for buy, 75 for sell)
        """
        dates = pd.date_range(start=datetime.now() - timedelta(days=10), end=datetime.now(), freq='1H')
        
        prices = [base_price]
        for i in range(1, len(dates)):
            if trend == "uptrend":
                change = np.random.normal(0.1, 0.2)  # Mild uptrend
            elif trend == "downtrend":
                change = np.random.normal(-0.1, 0.2)  # Mild downtrend
            else:  # sideways
                change = np.random.normal(0, 0.1)  # Neutral trend
            
            new_price = prices[-1] * (1 + change/100)
            prices.append(max(new_price, base_price * 0.8))  # Prevent negative price
        
        # Create artificial RSI at desired level
        df = pd.DataFrame({
            'open_time': dates,
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices],
            'close': prices,
            'volume': np.random.normal(1000, 200, len(dates))
        })
        
        # Calculate real RSI
        df = calculate_rsi(df)
        
        # Set last candle RSI to desired level
        df.loc[df.index[-1], 'RSI'] = rsi_level
        
        logger.info(f"📊 Test data created: {len(df)} candles, Last RSI: {rsi_level}")
        return df
    
    def wait_for_position_opening(self, timeout_minutes=10):
        """
        Wait for position opening
        
        Args:
            timeout_minutes: Maximum wait time (minutes)
        """
        logger.info("⏳ Waiting for buy signal...")
        
        start_time = datetime.now()
        timeout = timedelta(minutes=timeout_minutes)
        
        while datetime.now() - start_time < timeout:
            try:
                # Get proper interval code
                interval_code = self.get_interval_code(self.interval, self.data_source)
                
                # Get real data
                data = fetch_market_data(
                    self.symbol_code, 
                    interval_code, 
                    100, 
                    "MT5" if self.data_source == "MT5" else "CRYPTOCOMPARE"
                )
                
                if data.empty:
                    logger.warning("⚠️ No data received, retrying...")
                    time.sleep(30)
                    continue
                
                # Calculate RSI
                data = calculate_rsi(data)
                
                # Generate signal
                signal = self.strategy.generate_signal(data)
                
                if signal['action'] == 'BUY':
                    logger.info("🎯 Buy signal received!")
                    current_price = get_current_price(self.symbol_code, 
                                                    "MT5" if self.data_source == "MT5" else "CRYPTOCOMPARE")
                    
                    self.test_results.update({
                        'position_opened': True,
                        'entry_time': datetime.now(),
                        'entry_price': current_price,
                        'signal_strength': signal.get('signal_strength', 'N/A')
                    })
                    return True
                
                logger.info(f"📊 Current status: {signal['action']} - RSI: {signal.get('rsi', 0):.1f}")
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                error_msg = f"Error waiting for position: {str(e)}"
                logger.error(error_msg)
                self.test_results['errors'].append(error_msg)
                time.sleep(30)
        
        logger.warning("⏰ Wait time for buy signal expired")
        return False
    
    def monitor_position(self, max_duration_minutes=10):
        """
        Monitor open position
        
        Args:
            max_duration_minutes: Maximum monitoring time (minutes)
        """
        if not self.test_results['position_opened']:
            logger.error("❌ No open position to monitor")
            return False
        
        logger.info("🔍 Starting open position monitoring...")
        
        start_time = datetime.now()
        max_duration = timedelta(minutes=max_duration_minutes)
        
        while datetime.now() - start_time < max_duration:
            try:
                # Get proper interval code
                interval_code = self.get_interval_code(self.interval, self.data_source)
                
                # Get real data
                data = fetch_market_data(
                    self.symbol_code, 
                    interval_code, 
                    100, 
                    "MT5" if self.data_source == "MT5" else "CRYPTOCOMPARE"
                )
                
                if data.empty:
                    logger.warning("⚠️ No data received, retrying...")
                    time.sleep(15)
                    continue
                
                # Calculate RSI
                data = calculate_rsi(data)
                
                # Check exit conditions
                exit_signal = self.strategy.check_exit_conditions(data)
                
                if exit_signal and exit_signal['action'] == 'SELL':
                    logger.info(f"🎯 Exit signal received: {exit_signal['reason']}")
                    
                    current_price = get_current_price(self.symbol_code, 
                                                    "MT5" if self.data_source == "MT5" else "CRYPTOCOMPARE")
                    self.test_results.update({
                        'position_closed': True,
                        'exit_time': datetime.now(),
                        'exit_price': current_price,
                        'exit_reason': exit_signal.get('exit_reason', 'UNKNOWN'),
                        'pnl_percentage': exit_signal.get('pnl_percentage', 0),
                        'trade_duration_minutes': (datetime.now() - self.test_results['entry_time']).total_seconds() / 60
                    })
                    
                    # Update performance
                    performance = self.strategy.get_performance_metrics()
                    logger.info(f"📊 Final performance: {performance}")
                    
                    return True
                
                # Display current status
                current_trade = self.strategy.current_trade
                if current_trade:
                    current_price = get_current_price(self.symbol_code, 
                                                    "MT5" if self.data_source == "MT5" else "CRYPTOCOMPARE")
                    unrealized_pnl = ((current_price - current_trade.entry_price) / current_trade.entry_price) * 100
                    
                    logger.info(f"📈 Open position: {current_trade.position_type.value}")
                    logger.info(f"💰 Entry price: {current_trade.entry_price:.2f}")
                    logger.info(f"💰 Current price: {current_price:.2f}")
                    logger.info(f"📊 Current PnL: {unrealized_pnl:+.2f}%")
                    logger.info(f"🛑 Stop loss: {current_trade.stop_loss:.2f}")
                    logger.info(f"🎯 Take profit: {current_trade.take_profit:.2f}")
                
                time.sleep(15)  # Check every 15 seconds
                
            except Exception as e:
                error_msg = f"Error monitoring position: {str(e)}"
                logger.error(error_msg)
                self.test_results['errors'].append(error_msg)
                time.sleep(15)
        
        logger.warning("⏰ Monitoring time expired")
        return False
    
    def run_simulation_test(self):
        """Run simulation test"""
        logger.info("🎲 Starting simulation test")
        
        try:
            # Create test data with buy conditions
            test_data = self.create_test_data_with_signal(rsi_level=25)  # Low RSI for buy
            
            # Reset strategy
            self.strategy.reset_state()
            
            # Generate signal
            signal = self.strategy.generate_signal(test_data)
            
            if signal['action'] == 'BUY':
                logger.info("✅ Simulation test: Buy signal generated")
                
                # Simulate time passage and create new data for exit
                time.sleep(2)
                
                # Create new data with exit conditions (e.g., high RSI)
                exit_data = self.create_test_data_with_signal(rsi_level=75)
                
                # Check exit
                exit_signal = self.strategy.check_exit_conditions(exit_data)
                
                if exit_signal and exit_signal['action'] == 'SELL':
                    logger.info("✅ Simulation test: Position closed successfully")
                    return True
                else:
                    logger.warning("⚠️ Simulation test: Position not closed")
                    return False
            else:
                logger.warning(f"⚠️ Simulation test: Buy signal not generated (signal: {signal['action']})")
                return False
                
        except Exception as e:
            error_msg = f"Simulation test error: {str(e)}"
            logger.error(error_msg)
            self.test_results['errors'].append(error_msg)
            return False
    
    def run_live_test(self, wait_timeout=10, monitor_timeout=10):
        """Run live test with real data"""
        logger.info("🌐 Starting live test with real data")
        
        try:
            # Step 1: Wait for position opening
            position_opened = self.wait_for_position_opening(wait_timeout)
            
            if not position_opened:
                logger.error("❌ Live test: No position opened")
                return False
            
            logger.info("✅ Position opened successfully")
            
            # Step 2: Monitor position
            position_closed = self.monitor_position(monitor_timeout)
            
            if position_closed:
                logger.info("✅ Live test: Position closed successfully")
                return True
            else:
                logger.warning("⚠️ Live test: Position not closed within specified time")
                return False
                
        except Exception as e:
            error_msg = f"Live test error: {str(e)}"
            logger.error(error_msg)
            self.test_results['errors'].append(error_msg)
            return False
    
    def print_test_report(self):
        """Print complete test report"""
        print("\n" + "="*80)
        print("📊 COMPLETE POSITION LIFECYCLE TEST REPORT")
        print("="*80)
        
        print(f"📅 Test time: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
        print(f"💎 Symbol: {self.symbol_display} ({self.symbol_code})")
        print(f"🌐 Data source: {self.data_source}")
        print(f"⏰ Timeframe: {self.interval}")
        
        print("\n🔍 Test Results:")
        print(f"   ✅ Position opened: {'Yes' if self.test_results['position_opened'] else 'No'}")
        print(f"   ✅ Position closed: {'Yes' if self.test_results['position_closed'] else 'No'}")
        
        if self.test_results['position_opened']:
            print(f"   🕒 Entry time: {self.test_results['entry_time']}")
            print(f"   💰 Entry price: {self.test_results.get('entry_price', 0):.2f}$")
            print(f"   💪 Signal strength: {self.test_results.get('signal_strength', 'N/A')}")
        
        if self.test_results['position_closed']:
            print(f"   🕒 Exit time: {self.test_results['exit_time']}")
            print(f"   💰 Exit price: {self.test_results.get('exit_price', 0):.2f}$")
            print(f"   📊 PnL: {self.test_results.get('pnl_percentage', 0):+.2f}%")
            print(f"   🕓 Trade duration: {self.test_results.get('trade_duration_minutes', 0):.1f} minutes")
            print(f"   🎯 Exit reason: {self.test_results.get('exit_reason', 'N/A')}")
        
        # Display errors
        if self.test_results['errors']:
            print(f"\n❌ Errors occurred:")
            for i, error in enumerate(self.test_results['errors'], 1):
                print(f"   {i}. {error}")
        
        # Display strategy performance
        performance = self.strategy.get_performance_metrics()
        print(f"\n📈 Strategy Performance:")
        print(f"   📋 Total trades: {performance.get('total_trades', 0)}")
        print(f"   🎯 Win rate: {performance.get('win_rate', 0):.1f}%")
        print(f"   💰 Total PnL: {performance.get('total_pnl', 0):+.2f}$")
        print(f"   📊 Profit factor: {performance.get('profit_factor', 0):.2f}")
        
        print("\n" + "="*80)
        print("✅ Test completed")
        print("="*80)

def get_available_intervals(data_source):
    """Get available intervals for the data source"""
    if data_source == "MT5":
        return ["1 minute", "5 minutes", "15 minutes", "30 minutes", "1 hour", "4 hours", "1 day", "1 week"]
    else:
        return ["1 hour", "1 day", "1 week"]

def get_available_symbols(data_source):
    """Get available symbols for the data source"""
    if data_source == "MT5":
        return [
            "Gold (XAUUSD)", "Silver (XAGUSD)", "EUR/USD (EURUSD)", 
            "GBP/USD (GBPUSD)", "USD/JPY (USDJPY)", "Oil (XTIUSD)"
        ]
    else:
        return [
            "Bitcoin (BTC)", "Ethereum (ETH)", "Binance Coin (BNB)",
            "Solana (SOL)", "Ripple (XRP)", "Cardano (ADA)"
        ]

def test_multiple_symbols():
    """Test on multiple different symbols"""
    symbols_to_test = [
        ("Gold (XAUUSD)", "MT5", "1 hour"),
        ("Bitcoin (BTC)", "CRYPTOCOMPARE", "1 hour"),
        ("EUR/USD (EURUSD)", "MT5", "1 hour")
    ]
    
    results = {}
    
    for symbol_display, data_source, interval in symbols_to_test:
        print(f"\n🧪 Testing symbol: {symbol_display} from {data_source}")
        print("-" * 50)
        
        try:
            tester = PositionLifecycleTester(data_source, symbol_display, interval)
            
            # Run simulation test
            sim_success = tester.run_simulation_test()
            
            # Run live test (short)
            live_success = tester.run_live_test(wait_timeout=2, monitor_timeout=2)
            
            results[symbol_display] = {
                'simulation_success': sim_success,
                'live_success': live_success,
                'errors': tester.test_results['errors']
            }
            
            tester.print_test_report()
        except Exception as e:
            print(f"❌ Failed to test {symbol_display}: {e}")
            results[symbol_display] = {
                'simulation_success': False,
                'live_success': False,
                'errors': [str(e)]
            }
    
    return results

def main():
    """Main test execution function"""
    print("🚀 TradeBot Pro - Complete Position Lifecycle Test")
    print("Version 1.0.1 - Fixed timeframe and symbol mapping")
    print("Developed by Market Analysis Team")
    print("=" * 60)
    
    # Select type of test
    print("\n🎯 Please select test type:")
    print("1. Simulation test (fast)")
    print("2. Live test with real data (Gold - MT5)")
    print("3. Test on multiple symbols")
    print("4. Custom test")
    
    choice = input("\n📍 Your choice (1-4): ").strip()
    
    if choice == "1":
        # Simulation test
        tester = PositionLifecycleTester()
        success = tester.run_simulation_test()
        tester.print_test_report()
        
    elif choice == "2":
        # Live test with Gold
        tester = PositionLifecycleTester("MT5", "Gold (XAUUSD)", "1 hour")
        success = tester.run_live_test(wait_timeout=5, monitor_timeout=5)
        tester.print_test_report()
        
    elif choice == "3":
        # Multiple symbols test
        results = test_multiple_symbols()
        
        print("\n📋 Multiple symbols test summary:")
        for symbol, result in results.items():
            status = "✅ Success" if result['simulation_success'] or result['live_success'] else "❌ Failed"
            print(f"   {symbol}: {status}")
            
    elif choice == "4":
        # Custom test
        print("\n🔧 Custom Test Configuration")
        
        # Data source selection
        data_source = input("🌐 Data source (MT5/CRYPTOCOMPARE): ").strip().upper()
        if data_source not in ["MT5", "CRYPTOCOMPARE"]:
            print("❌ Invalid data source. Using MT5 as default.")
            data_source = "MT5"
        
        # Symbol selection
        available_symbols = get_available_symbols(data_source)
        print(f"\n💎 Available symbols for {data_source}:")
        for i, symbol in enumerate(available_symbols, 1):
            print(f"   {i}. {symbol}")
        
        try:
            symbol_choice = int(input(f"\n📍 Select symbol (1-{len(available_symbols)}): ")) - 1
            symbol_display = available_symbols[symbol_choice]
        except:
            symbol_display = available_symbols[0]
            print(f"⚠️ Invalid choice. Using default: {symbol_display}")
        
        # Interval selection
        available_intervals = get_available_intervals(data_source)
        print(f"\n⏰ Available intervals for {data_source}:")
        for i, interval in enumerate(available_intervals, 1):
            print(f"   {i}. {interval}")
        
        try:
            interval_choice = int(input(f"\n📍 Select interval (1-{len(available_intervals)}): ")) - 1
            interval = available_intervals[interval_choice]
        except:
            interval = available_intervals[0]
            print(f"⚠️ Invalid choice. Using default: {interval}")
        
        wait_timeout = int(input("⏳ Wait time for signal (minutes): ") or "3")
        monitor_timeout = int(input("🔍 Monitoring time (minutes): ") or "3")
        
        tester = PositionLifecycleTester(data_source, symbol_display, interval)
        success = tester.run_live_test(wait_timeout, monitor_timeout)
        tester.print_test_report()
        
    else:
        print("❌ Invalid choice!")
        return
    
    print(f"\n🎉 Test completed {'successfully' if success else 'with errors'}!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Test stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")