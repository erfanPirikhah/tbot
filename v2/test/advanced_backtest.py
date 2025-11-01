# tests/advanced_backtest_fixed.py

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover
except ImportError:
    print("❌ پکیج backtesting نصب نیست. لطفاً اجرا کنید: pip install backtesting")
    sys.exit(1)

import matplotlib.pyplot as plt
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    print("⚠️ scipy نصب نیست. برخی معیارهای پیشرفته غیرفعال خواهند بود.")
    SCIPY_AVAILABLE = False
    stats = None

# Import project modules
try:
    from strategies.adaptive_elite_rsi_strategy import ProfessionalAdvancedRsiStrategy, PositionType
    from data.data_fetcher import fetch_market_data
    from indicators.rsi import calculate_rsi
    from config import PROFESSIONAL_STRATEGY_PARAMS
except ImportError as e:
    print(f"❌ خطا در ایمپورت ماژول‌های پروژه: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_backtest.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProfessionalRSIStrategyBacktest(Strategy):
    """
    کلاس استراتژی برای بکتست با backtesting.py - نسخه کاملاً تصحیح شده
    """
    
    # تعریف پارامترهای استراتژی به صورت کلاس-ول
    rsi_period = 14
    rsi_base_oversold = 30
    rsi_base_overbought = 70
    risk_per_trade = 0.02
    base_stop_atr_multiplier = 2.0
    base_take_profit_ratio = 2.5
    max_trade_duration = 168
    enable_short_trades = True
    
    def init(self):
        """آماده‌سازی اندیکاتورها"""
        try:
            # محاسبه RSI
            self.rsi = self.I(self.calculate_rsi_indicator)
            
            # ایجاد نمونه استراتژی اصلی با پارامترهای سازگار
            self.main_strategy = ProfessionalAdvancedRsiStrategy(
                rsi_period=self.rsi_period,
                rsi_base_oversold=self.rsi_base_oversold,
                rsi_base_overbought=self.rsi_base_overbought,
                risk_per_trade=self.risk_per_trade,
                base_stop_atr_multiplier=self.base_stop_atr_multiplier,
                base_take_profit_ratio=self.base_take_profit_ratio,
                max_trade_duration=self.max_trade_duration,
                enable_short_trades=self.enable_short_trades,
                use_adaptive_rsi=True,
                use_adaptive_adx=True,
                use_rsi_momentum=True,
                use_price_roc=True,
                use_volatility_regimes=True,
                use_dynamic_trailing=True,
                use_mtf_confirmation=False,
                # کاهش سطوح سخت‌گیرانه
                max_trades_per_100=50,  # افزایش از 30 به 50
                min_candles_between=2,  # کاهش از 3 به 2
                adx_base_threshold=15.0  # کاهش از 20 به 15
            )
            
            self.trade_history = []
            self.current_signal = None
            self.initialized = True
            self.signal_count = 0
            
            logger.info("✅ Strategy initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error in strategy initialization: {e}")
            self.initialized = False
        
    def calculate_rsi_indicator(self):
        """محاسبه RSI برای داده‌های فعلی"""
        try:
            closes = pd.Series(self.data.Close)
            # ایجاد DataFrame موقت برای محاسبه RSI
            temp_df = pd.DataFrame({'close': closes})
            temp_df = calculate_rsi(temp_df, period=self.rsi_period)
            return temp_df['RSI'].fillna(50).values
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return np.full(len(self.data.Close), 50)
    
    def next(self):
        """اجرای استراتژی در هر مرحله"""
        if not hasattr(self, 'initialized') or not self.initialized:
            return
            
        try:
            # اگر داده کافی نداریم، صبر کن
            if len(self.data) < max(self.rsi_period, 100):  # افزایش به 100 برای اطمینان
                return
            
            # ایجاد DataFrame از داده‌های فعلی
            current_data = self.create_current_dataframe()
            
            if current_data.empty or 'RSI' not in current_data.columns:
                return
            
            # بررسی اینکه RSI دارای مقادیر معتبر است
            current_rsi = current_data['RSI'].iloc[-1]
            if pd.isna(current_rsi) or current_rsi == 0:
                return
            
            # تولید سیگنال با استراتژی اصلی
            current_index = len(self.data) - 1
            signal_info = self.main_strategy.generate_signal(current_data, current_index)
            self.current_signal = signal_info
            
            # لاگ سیگنال برای دیباگ
            self.signal_count += 1
            if self.signal_count % 100 == 0:  # لاگ هر 100 کندل
                action = signal_info.get('action', 'HOLD')
                reason = signal_info.get('reason', '')
                logger.info(f"📡 Signal #{self.signal_count}: {action} - {reason[:100]}...")
            
            # اجرای معامله بر اساس سیگنال
            self.execute_trade(signal_info, current_data)
            
        except Exception as e:
            logger.error(f"Error in strategy execution: {e}")
    
    def create_current_dataframe(self) -> pd.DataFrame:
        """ایجاد DataFrame از داده‌های فعلی برای استراتژی"""
        try:
            # ایجاد DataFrame با ساختار مورد نیاز استراتژی
            current_idx = len(self.data.Close)
            
            df = pd.DataFrame({
                'open': self.data.Open[-current_idx:],
                'high': self.data.High[-current_idx:],
                'low': self.data.Low[-current_idx:],
                'close': self.data.Close[-current_idx:],
                'volume': self.data.Volume[-current_idx:] if hasattr(self.data, 'Volume') else np.ones(current_idx)
            })
            
            # ایجاد index زمانی مصنوعی برای سازگاری
            df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
            
            # محاسبه RSI
            df = calculate_rsi(df, period=self.rsi_period)
            
            return df.iloc[-1000:]  # محدود کردن به 1000 کندل آخر برای عملکرد بهتر
            
        except Exception as e:
            logger.error(f"Error creating dataframe: {e}")
            # بازگرداندن DataFrame خالی در صورت خطا
            return pd.DataFrame()
    
    def execute_trade(self, signal_info: Dict, data: pd.DataFrame):
        """اجرای معامله بر اساس سیگنال"""
        try:
            current_price = self.data.Close[-1]
            current_index = len(self.data) - 1
            
            action = signal_info.get('action', 'HOLD')
            
            if action == "BUY" and not self.position:
                # محاسبه حجم معامله
                stop_loss = signal_info.get('stop_loss', current_price * 0.98)
                take_profit = signal_info.get('take_profit', current_price * 1.04)
                
                position_size = self.calculate_position_size(current_price, stop_loss)
                
                if position_size > 0:
                    # استفاده از حد سود و ضرر در معامله
                    self.buy(
                        size=position_size,
                        sl=stop_loss,
                        tp=take_profit
                    )
                    self.record_trade_entry('LONG', current_price, current_index, signal_info)
                    logger.info(f"📈 BUY signal executed at {current_price:.4f}, SL: {stop_loss:.4f}, TP: {take_profit:.4f}")
                    
            elif action == "SHORT" and not self.position:
                stop_loss = signal_info.get('stop_loss', current_price * 1.02)
                take_profit = signal_info.get('take_profit', current_price * 0.96)
                
                position_size = self.calculate_position_size(current_price, stop_loss)
                
                if position_size > 0:
                    self.sell(
                        size=position_size,
                        sl=stop_loss,
                        tp=take_profit
                    )
                    self.record_trade_entry('SHORT', current_price, current_index, signal_info)
                    logger.info(f"📉 SHORT signal executed at {current_price:.4f}, SL: {stop_loss:.4f}, TP: {take_profit:.4f}")
                    
            elif action in ["SELL", "COVER"] and self.position:
                self.position.close()
                self.record_trade_exit(current_price, current_index, signal_info)
                logger.info(f"🔚 {action} signal executed at {current_price:.4f}")
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """محاسبه حجم معامله با مدیریت ریسک"""
        try:
            # استفاده از equity به جای cash
            account_equity = self.equity
            
            risk_amount = account_equity * self.risk_per_trade
            price_risk = abs(entry_price - stop_loss)
            
            if price_risk == 0:
                return 0
                
            position_size = risk_amount / price_risk
            
            # محدودیت حداکثر پوزیشن (20% سرمایه)
            max_position_value = account_equity * 0.20
            max_position_size = max_position_value / entry_price
            
            return min(position_size, max_position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def record_trade_entry(self, position_type: str, price: float, time_index: int, signal_info: Dict):
        """ثبت اطلاعات ورود به معامله"""
        trade_info = {
            'entry_time': time_index,
            'entry_price': price,
            'position_type': position_type,
            'signal_strength': signal_info.get('signal_strength', 'UNKNOWN'),
            'confluence_score': signal_info.get('confluence_score', 0),
            'rsi': signal_info.get('rsi', 0),
            'reason': signal_info.get('reason', ''),
            'volatility_regime': signal_info.get('volatility_regime', 'UNKNOWN')
        }
        self.trade_history.append(trade_info)
    
    def record_trade_exit(self, price: float, time_index: int, signal_info: Dict):
        """ثبت اطلاعات خروج از معامله"""
        if self.trade_history:
            last_trade = self.trade_history[-1]
            last_trade['exit_time'] = time_index
            last_trade['exit_price'] = price
            last_trade['exit_reason'] = signal_info.get('reason', 'Exit signal')
            
            # محاسبه سود/زیان
            if last_trade['position_type'] == 'LONG':
                pnl_pct = ((price - last_trade['entry_price']) / last_trade['entry_price']) * 100
            else:
                pnl_pct = ((last_trade['entry_price'] - price) / last_trade['entry_price']) * 100
                
            last_trade['pnl_percentage'] = pnl_pct
            last_trade['duration'] = time_index - last_trade['entry_time']

class SimpleBacktestEngine:
    """
    موتور بکتست ساده‌شده برای تست استراتژی
    """
    
    def __init__(self):
        self.results = {}
        
    def run_simple_backtest(
        self,
        symbol: str,
        interval: str,
        data_source: str = "MT5",
        days: int = 60,
        initial_balance: float = 10000.0,
        commission: float = 0.001
    ) -> Dict:
        """
        اجرای تست ساده استراتژی
        """
        print(f"🚀 Starting backtest for {symbol}...")
        
        try:
            # دریافت داده‌های تاریخی
            data = self.fetch_simple_data(symbol, interval, data_source, days)
            if data.empty:
                print(f"❌ No data fetched for {symbol}")
                return {'error': 'No data available'}
            
            print(f"📊 Data loaded: {len(data)} candles")
            
            # اجرای بکتست
            bt = Backtest(
                data,
                ProfessionalRSIStrategyBacktest,
                cash=initial_balance,
                commission=commission,
                exclusive_orders=True,
                trade_on_close=True
            )
            
            # اجرای تست
            output = bt.run()
            
            # نمایش نتایج پایه
            self.display_results(symbol, output)
            
            # ذخیره نتایج تفصیلی
            try:
                results_folder = "backtest_results"
                os.makedirs(results_folder, exist_ok=True)
                html_file = f"{results_folder}/backtest_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                bt.plot(filename=html_file)
                print(f"💾 Detailed results saved to: {html_file}")
            except Exception as e:
                print(f"⚠️ Could not save HTML results: {e}")
            
            return {
                'output': output,
                'backtest': bt,
                'data': data
            }
            
        except Exception as e:
            print(f"❌ Backtest failed for {symbol}: {e}")
            return {'error': str(e)}
    
    def fetch_simple_data(self, symbol: str, interval: str, data_source: str, days: int) -> pd.DataFrame:
        """دریافت داده‌های تاریخی ساده"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            print(f"📥 Fetching data for {symbol} (last {days} days)...")
            data = fetch_market_data(symbol, interval, limit=days*24, data_source=data_source)
            
            if data.empty:
                print(f"⚠️ No real data for {symbol}, creating sample data...")
                return self.create_realistic_sample_data(days, symbol)
            
            # تبدیل به فرمت مورد نیاز backtesting.py
            data = data.rename(columns={
                'open_time': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # حذف ستون‌های اضافی
            keep_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            data = data[[col for col in keep_columns if col in data.columns]]
            
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.set_index('Date')
            
            # حذف ردیف‌های تکراری و مرتب‌سازی
            data = data[~data.index.duplicated(keep='first')]
            data = data.sort_index()
            
            print(f"✅ Data prepared: {data.shape[0]} rows, {data.shape[1]} columns")
            return data
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return self.create_realistic_sample_data(days, symbol)
    
    def create_realistic_sample_data(self, days: int, symbol: str = "XAUUSD") -> pd.DataFrame:
        """ایجاد داده نمونه واقعی‌تر با نوسانات بیشتر"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                            end=datetime.now(), freq='H')
        n = len(dates)
        
        # ایجاد داده قیمت مصنوعی با روند واقعی‌تر و نوسانات بیشتر
        np.random.seed(42)
        
        # انتخاب پایه قیمت بر اساس نوع نماد
        base_prices = {
            'XAUUSD': 1800,
            'EURUSD': 1.08,
            'GBPUSD': 1.26,
            'USDJPY': 150.0
        }
        
        # استفاده از پایه قیمت مناسب یا مقدار پیش‌فرض
        base_price = base_prices.get(symbol, 100.0)
        
        # ایجاد روند + نوسانات قوی‌تر برای تست استراتژی
        trend = np.cumsum(np.random.normal(0.001, 0.002, n))  # افزایش نوسان
        noise = np.random.normal(0, 0.01, n)  # افزایش نویز
        prices = base_price * np.exp(trend + noise)
        
        # اضافه کردن حرکات شدید قیمت برای تست شرایط اشباع خرید/فروش
        for i in range(len(prices)):
            if i % 50 == 25:  # هر 50 کندل یک افت شدید
                prices[i:i+10] = prices[i:i+10] * 0.95
            elif i % 50 == 0:  # هر 50 کندل یک صعود شدید
                prices[i:i+10] = prices[i:i+10] * 1.05
        
        # ایجاد OHLC داده
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.002, n)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.015, n))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.015, n))),
            'Close': prices,
            'Volume': np.random.lognormal(10, 1, n) * 1000
        }, index=dates)
        
        # اطمینان از High >= Low و High >= Open, Close و Low <= Open, Close
        data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
        data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
        
        print(f"✅ Created realistic sample data for {symbol} with {len(data)} candles")
        return data
    
    def display_results(self, symbol: str, output):
        """نمایش نتایج تست"""
        print(f"\n{'='*50}")
        print(f"📊 BACKTEST RESULTS for {symbol}")
        print(f"{'='*50}")
        
        # نمایش معیارهای اصلی
        metrics = [
            ('Return [%]', 'Total Return', '{:.2f}%'),
            ('Max. Drawdown [%]', 'Max Drawdown', '{:.2f}%'),
            ('# Trades', 'Total Trades', '{:.0f}'),
            ('Win Rate [%]', 'Win Rate', '{:.1f}%'),
            ('Sharpe Ratio', 'Sharpe Ratio', '{:.2f}'),
            ('Profit Factor', 'Profit Factor', '{:.2f}'),
            ('Avg. Trade [%]', 'Avg Trade', '{:.2f}%')
        ]
        
        for key, label, fmt in metrics:
            if key in output:
                value = output[key]
                # جایگزینی nan با 0
                if pd.isna(value):
                    value = 0
                print(f"   {label}: {fmt.format(value)}")
        
        # نمایش اطلاعات معاملات اگر موجود باشد
        if hasattr(output, '_trades') and not output._trades.empty:
            trades = output._trades
            print(f"\n   First Trade: {trades.iloc[0]['EntryBar']} -> {trades.iloc[0]['ExitBar']}")
            print(f"   Last Trade: {trades.iloc[-1]['EntryBar']} -> {trades.iloc[-1]['ExitBar']}")
            
            # تحلیل جزئی‌تر معاملات
            winning_trades = trades[trades['PnL'] > 0]
            losing_trades = trades[trades['PnL'] < 0]
            
            if len(trades) > 0:
                print(f"   Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)")
                print(f"   Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)")
                print(f"   Best Trade: ${trades['PnL'].max():.2f}")
                print(f"   Worst Trade: ${trades['PnL'].min():.2f}")
        
        print(f"{'='*50}")

def run_simple_backtest_demo():
    """اجرای دموی تست ساده"""
    engine = SimpleBacktestEngine()
    
    # استفاده از نمادهای اصلی
    symbols = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]
    
    print("🚀 Starting Simple Backtest Demo...")
    print("This will test the RSI strategy on multiple symbols")
    print("=" * 60)
    
    results = {}
    
    for symbol in symbols:
        print(f"\n🎯 Testing {symbol}...")
        result = engine.run_simple_backtest(
            symbol=symbol,
            interval="H1",
            data_source="MT5",
            days=90,  # 90 روز داده
            initial_balance=10000.0
        )
        
        results[symbol] = result
        
        # وقفه کوتاه بین تست‌ها
        import time
        time.sleep(1)
    
    # خلاصه نتایج
    print(f"\n{'='*60}")
    print("🎯 BACKTEST SUMMARY")
    print(f"{'='*60}")
    
    successful_tests = 0
    total_return = 0
    
    for symbol, result in results.items():
        if 'error' not in result:
            output = result['output']
            return_pct = output['Return [%]'] if 'Return [%]' in output else 0
            total_trades = output['# Trades'] if '# Trades' in output else 0
            
            status = "✅" if total_trades > 0 else "⚠️"
            print(f"{status} {symbol}: Return: {return_pct:.2f}%, Trades: {total_trades}")
            
            if total_trades > 0:
                successful_tests += 1
                total_return += return_pct
        else:
            print(f"❌ {symbol}: {result['error']}")
    
    if successful_tests > 0:
        avg_return = total_return / successful_tests
        print(f"\n📈 Average Return: {avg_return:.2f}%")
        print(f"🎯 Successful Tests: {successful_tests}/{len(symbols)}")
    else:
        print(f"\n❌ No successful tests completed")
    
    print(f"\n💡 Tip: Check the generated HTML files for detailed charts")
    print(f"📁 Logs saved in: advanced_backtest.log")

def run_single_symbol_test():
    """تست روی یک نماد خاص"""
    engine = SimpleBacktestEngine()
    
    symbol = "XAUUSD"  # تست روی طلا
    
    print(f"🎯 Running single symbol test for {symbol}...")
    print("=" * 50)
    
    result = engine.run_simple_backtest(
        symbol=symbol,
        interval="H1", 
        data_source="MT5",
        days=120,  # 120 روز داده
        initial_balance=10000.0
    )
    
    if 'error' not in result:
        print(f"\n✅ {symbol} test completed successfully!")
        
        # نمایش جزئیات بیشتر
        output = result['output']
        if hasattr(output, '_trades') and not output._trades.empty:
            trades = output._trades
            print(f"\n📋 Trade Details:")
            print(f"   Total Trades: {len(trades)}")
            print(f"   Winning Trades: {(trades['PnL'] > 0).sum()}")
            print(f"   Losing Trades: {(trades['PnL'] < 0).sum()}")
            
            if len(trades) > 0:
                best_trade = trades['PnL'].max()
                worst_trade = trades['PnL'].min()
                print(f"   Best Trade: ${best_trade:.2f}")
                print(f"   Worst Trade: ${worst_trade:.2f}")
    else:
        print(f"❌ Test failed: {result['error']}")

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Multi-symbol demo (4 symbols)")
    print("2. Single symbol test (XAUUSD)")
    
    try:
        choice = input("Enter choice (1 or 2, default=1): ").strip()
        
        if choice == "2":
            run_single_symbol_test()
        else:
            run_simple_backtest_demo()
            
    except KeyboardInterrupt:
        print("\n⏹️ Test cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")