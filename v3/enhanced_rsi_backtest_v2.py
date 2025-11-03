# backtesting/enhanced_rsi_backtest_mt5.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
import sys
import os

# اضافه کردن مسیرهای مورد نیاز
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import MT5 data fetcher
from data.mt5_data import mt5_fetcher, MT5_AVAILABLE
from enhanced_rsi_strategy_v2 import EnhancedRsiStrategyV2Improved

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedRSIBacktestMT5:
    """
    سیستم بکتست پیشرفته با داده‌های واقعی از MT5
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        enable_plotting: bool = True,
        detailed_logging: bool = True
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.enable_plotting = enable_plotting
        self.detailed_logging = detailed_logging
        
        # Results storage
        self.results = {}
        self.trades_df = None
        self.equity_curve = None
        self.daily_returns = None
        
    def fetch_real_data_from_mt5(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "H1",
        days_back: int = 365
    ) -> pd.DataFrame:
        """
        دریافت داده‌های واقعی از MT5
        
        Parameters:
        -----------
        symbol : str
            نماد معاملاتی (مثلاً EURUSD, GBPUSD, etc.)
        timeframe : str
            تایم‌فریم (M1, M5, M15, M30, H1, H4, D1, W1)
        days_back : int
            تعداد روزهای تاریخی مورد نیاز
            
        Returns:
        --------
        pd.DataFrame with OHLCV data
        """
        logger.info(f"📥 دریافت داده‌های واقعی از MT5 برای {symbol} ({timeframe})")
        
        if not MT5_AVAILABLE or mt5_fetcher is None:
            raise ConnectionError("❌ MT5 در دسترس نیست. لطفاً اطمینان حاصل کنید که MetaTrader5 نصب شده و متصل است.")
        
        try:
            # محاسبه تعداد کندل‌های مورد نیاز
            timeframe_minutes = {
                "M1": 1, "M5": 5, "M15": 15, "M30": 30,
                "H1": 60, "H4": 240, "D1": 1440, "W1": 10080
            }
            
            minutes_per_day = 1440
            total_minutes = days_back * minutes_per_day
            candles_needed = total_minutes // timeframe_minutes.get(timeframe, 60)
            
            # دریافت داده از MT5
            data = mt5_fetcher.fetch_market_data(symbol, timeframe, candles_needed)
            
            if data.empty:
                raise ValueError(f"❌ هیچ داده‌ای برای {symbol} دریافت نشد")
            
            # محاسبه اندیکاتور RSI
            data = self._calculate_rsi(data)
            
            # محاسبه میانگین‌های متحرک برای فیلتر روند
            data['MA_10'] = data['close'].rolling(10).mean()
            data['MA_20'] = data['close'].rolling(20).mean()
            
            logger.info(f"✅ دریافت {len(data)} کندل از MT5 برای {symbol}")
            logger.info(f"📅 بازه زمانی: {data.index[0]} تا {data.index[-1]}")
            logger.info(f"💰 محدوده قیمت: {data['low'].min():.4f} - {data['high'].max():.4f}")
            logger.info(f"📊 نوسان بازار: {data['close'].pct_change().std():.4f}")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ خطا در دریافت داده از MT5: {e}")
            raise
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """محاسبه اندیکاتور RSI"""
        try:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # پر کردن مقادیر NaN
            data['RSI'] = data['RSI'].fillna(method='bfill')
            
            return data
        except Exception as e:
            logger.error(f"خطا در محاسبه RSI: {e}")
            return data
    
    def run_backtest(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "H1",
        days_back: int = 180,
        strategy_params: Dict[str, Any] = None,
        calculate_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        اجرای بکتست کامل با داده‌های واقعی MT5
        """
        logger.info("🚀 شروع بکتست با داده‌های واقعی MT5")
        
        # پارامترهای پیش‌فرض استراتژی بهبود یافته
        if strategy_params is None:
            strategy_params = {
 
            }
        
        # دریافت داده‌های واقعی
        data = self.fetch_real_data_from_mt5(symbol, timeframe, days_back)
        
        # Initialize strategy
        strategy = EnhancedRsiStrategyV2Improved(**strategy_params)
        
        # Storage for backtest results
        portfolio_values = []
        trades = []
        signals = []
        
        # Process each candle
        for i in range(len(data)):
            if i < 50:  # نیاز به داده کافی برای محاسبات
                continue
                
            current_data = data.iloc[:i+1].copy()
            current_time = current_data.index[-1]
            current_price = current_data['close'].iloc[-1]
            
            # تولید سیگنال
            signal = strategy.generate_signal(current_data, i)
            signal['timestamp'] = current_time
            signal['price'] = current_price
            signal['symbol'] = symbol
            signals.append(signal)
            
            # شبیه‌سازی اجرای معامله
            if signal['action'] in ['BUY', 'SELL']:
                trade_result = self._execute_trade(
                    strategy, signal, current_data, i
                )
                if trade_result:
                    trades.append(trade_result)
            
            # ذخیره وضعیت پورتفو
            portfolio_values.append({
                'timestamp': current_time,
                'portfolio_value': strategy._portfolio_value,
                'position': strategy._position.value,
                'price': current_price,
                'symbol': symbol
            })
        
        # جمع‌آوری نتایج
        self._compile_results(
            strategy, trades, portfolio_values, signals, data, symbol
        )
        
        if calculate_metrics:
            self._calculate_performance_metrics()
            
        if self.enable_plotting:
            self._generate_plots(data, symbol, timeframe)
            
        logger.info("✅ بکتست با داده‌های واقعی MT5 completed")
        return self.results
    
    def _execute_trade(
        self,
        strategy: EnhancedRsiStrategyV2Improved,
        signal: Dict[str, Any],
        data: pd.DataFrame,
        index: int
    ) -> Optional[Dict[str, Any]]:
        """شبیه‌سازی اجرای معامله"""
        try:
            current_price = data['close'].iloc[-1]
            
            # اعمال slippage
            if signal['action'] == 'BUY':
                execution_price = current_price * (1 + self.slippage)
            else:  # SELL
                execution_price = current_price * (1 - self.slippage)
            
            # اعمال کارمزد
            position_size = signal.get('position_size', 0)
            commission_cost = 0
            if position_size > 0:
                commission_cost = execution_price * position_size * self.commission
            
            # ذخیره اطلاعات معامله
            trade_record = {
                'timestamp': data.index[-1],
                'action': signal['action'],
                'symbol': signal.get('symbol', 'UNKNOWN'),
                'price': execution_price,
                'position_size': position_size,
                'stop_loss': signal.get('stop_loss', 0),
                'take_profit': signal.get('take_profit', 0),
                'commission': commission_cost,
                'rsi': signal.get('rsi', 0),
                'signal_strength': signal.get('signal_strength', 'NORMAL'),
                'signal_score': signal.get('signal_score', 1.0),
                'reason': signal.get('reason', ''),
                'portfolio_value_before': strategy._portfolio_value
            }
            
            if self.detailed_logging:
                logger.info(
                    f"🎯 {signal['action']} {signal.get('symbol', '')} at {execution_price:.4f} | "
                    f"Size: {position_size:.0f} | "
                    f"RSI: {signal.get('rsi', 0):.1f} | "
                    f"Strength: {signal.get('signal_strength', 'NORMAL')}"
                )
            
            return trade_record
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
    
    def _compile_results(
        self,
        strategy: EnhancedRsiStrategyV2Improved,
        trades: List[Dict],
        portfolio_values: List[Dict],
        signals: List[Dict],
        data: pd.DataFrame,
        symbol: str
    ):
        """کامپایل نتایج بکتست"""
        
        # معیارهای عملکرد از استراتژی
        strategy_metrics = strategy.get_performance_metrics()
        
        # ایجاد DataFrame معاملات
        self.trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # ایجاد منحنی سرمایه
        self.equity_curve = pd.DataFrame(portfolio_values)
        self.equity_curve.set_index('timestamp', inplace=True)
        
        # محاسبه بازده روزانه
        self.daily_returns = self._calculate_daily_returns()
        
        # ذخیره نتایج
        self.results = {
            'strategy_metrics': strategy_metrics,
            'trades': self.trades_df,
            'equity_curve': self.equity_curve,
            'signals': signals,
            'parameters': {k: v for k, v in strategy.__dict__.items() if not k.startswith('_')},
            'symbol': symbol,
            'data_info': {
                'start_date': data.index[0],
                'end_date': data.index[-1],
                'total_candles': len(data),
                'price_range': f"{data['low'].min():.4f} - {data['high'].max():.4f}",
                'initial_price': data['close'].iloc[0],
                'final_price': data['close'].iloc[-1],
                'price_change_pct': ((data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * 100),
                'volatility': data['close'].pct_change().std()
            }
        }
    
    def _calculate_daily_returns(self) -> pd.Series:
        if self.equity_curve is None or len(self.equity_curve) < 2:
            return pd.Series()
           
        try:
            # 🔧 اطمینان از DatetimeIndex
            if not isinstance(self.equity_curve.index, pd.DatetimeIndex):
                self.equity_curve.index = pd.to_datetime(self.equity_curve.index)
            
            daily_portfolio = self.equity_curve['portfolio_value'].resample('D').last()
            daily_returns = daily_portfolio.pct_change().dropna()
            
            return daily_returns

        except Exception as e:
            logger.error(f"خطا در محاسبه بازده روزانه: {e}")
            # 🔧 راه حل جایگزین: محاسبه ساده بازده
            try:
                portfolio_values = self.equity_curve['portfolio_value']
                simple_returns = portfolio_values.pct_change().dropna()
                return simple_returns
            except:
                return pd.Series()    
    
    def _calculate_performance_metrics(self):
        """محاسبه معیارهای عملکرد پیشرفته"""
        if self.equity_curve is None:
            return
            
        portfolio_values = self.equity_curve['portfolio_value']
        
        # محاسبه بازده کل
        total_return = ((portfolio_values.iloc[-1] - self.initial_capital) / 
                       self.initial_capital * 100)
        
        # محاسبه حداکثر drawdown
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max * 100
        max_drawdown = drawdowns.min()
        
        # محاسبه Sharpe Ratio
        if len(self.daily_returns) > 1:
            sharpe_ratio = (self.daily_returns.mean() / self.daily_returns.std() * 
                           np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        # معیارهای مبتنی بر معاملات
        if not self.trades_df.empty:
            total_trades = len(self.trades_df)
            winning_trades = len(self.trades_df[self.trades_df['action'].isin(['BUY', 'SELL'])])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # برای سادگی، میانگین سود را بر اساس تغییرات پورتفو محاسبه می‌کنیم
            if total_trades > 0:
                portfolio_change = portfolio_values.iloc[-1] - self.initial_capital
                avg_profit = portfolio_change / total_trades
            else:
                avg_profit = 0
            
            # Profit Factor ساده
            if portfolio_change > 0:
                profit_factor = 2.0  # مقدار تخمینی
            else:
                profit_factor = 0.5
        else:
            total_trades = 0
            win_rate = 0
            avg_profit = 0
            profit_factor = 0
        
        # ذخیره معیارهای پیشرفته
        self.results['advanced_metrics'] = {
            'total_return_pct': round(total_return, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'total_trades': total_trades,
            'win_rate_pct': round(win_rate, 2),
            'avg_trade_profit': round(avg_profit, 4),
            'profit_factor': round(profit_factor, 3),
            'cagr': self._calculate_cagr(),
            'calmar_ratio': self._calculate_calmar_ratio(total_return, max_drawdown),
            'initial_capital': self.initial_capital,
            'final_portfolio_value': round(portfolio_values.iloc[-1], 2)
        }
    
    def _calculate_cagr(self) -> float:
        """محاسبه نرخ رشد سالانه مرکب"""
        if self.equity_curve is None or len(self.equity_curve) < 2:
            return 0.0
            
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        years = days / 365.25
        
        if years <= 0:
            return 0.0
            
        ending_value = self.equity_curve['portfolio_value'].iloc[-1]
        beginning_value = self.initial_capital
        
        cagr = (ending_value / beginning_value) ** (1 / years) - 1
        return round(cagr * 100, 2)
    
    def _calculate_calmar_ratio(self, total_return: float, max_drawdown: float) -> float:
        """محاسبه Calmar Ratio"""
        if max_drawdown >= 0:
            return 0.0
        return round(total_return / abs(max_drawdown), 3)
    
    def _generate_plots(self, data: pd.DataFrame, symbol: str, timeframe: str):
        """تولید نمودارهای تحلیلی"""
        if self.equity_curve is None:
            return
            
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Enhanced RSI Strategy - MT5 Backtest Results\n{symbol} ({timeframe})', fontsize=16)
            
            # نمودار ۱: منحنی سرمایه و قیمت
            ax1 = axes[0, 0]
            ax1.plot(self.equity_curve.index, self.equity_curve['portfolio_value'], 
                    label='Portfolio Value', linewidth=2, color='blue')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_title('Equity Curve')
            
            # نمودار ۲: Drawdown
            ax2 = axes[0, 1]
            rolling_max = self.equity_curve['portfolio_value'].expanding().max()
            drawdown = (self.equity_curve['portfolio_value'] - rolling_max) / rolling_max * 100
            ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
            ax2.plot(drawdown.index, drawdown, color='red', linewidth=1)
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_title('Portfolio Drawdown')
            ax2.grid(True, alpha=0.3)
            
            # نمودار ۳: قیمت و RSI
            ax3 = axes[1, 0]
            # قیمت
            color = 'tab:blue'
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Price', color=color)
            ax3.plot(data.index, data['close'], color=color, linewidth=1, alpha=0.7)
            ax3.tick_params(axis='y', labelcolor=color)
            
            # RSI
            ax3_rsi = ax3.twinx()
            color = 'tab:red'
            ax3_rsi.set_ylabel('RSI', color=color)
            ax3_rsi.plot(data.index, data['RSI'], color=color, linewidth=1, alpha=0.7)
            ax3_rsi.tick_params(axis='y', labelcolor=color)
            ax3_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold')
            ax3_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought')
            ax3_rsi.set_ylim(0, 100)
            
            ax3.set_title(f'Price and RSI - {symbol}')
            ax3.grid(True, alpha=0.3)
            
            # نمودار ۴: توزیع بازده‌ها
            ax4 = axes[1, 1]
            if len(self.daily_returns) > 0:
                ax4.hist(self.daily_returns * 100, bins=50, alpha=0.7, color='green', edgecolor='black')
                ax4.axvline(self.daily_returns.mean() * 100, color='red', 
                           linestyle='--', label=f'Mean: {self.daily_returns.mean()*100:.2f}%')
                ax4.set_xlabel('Daily Return (%)')
                ax4.set_ylabel('Frequency')
                ax4.legend()
                ax4.set_title('Distribution of Daily Returns')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No daily returns data', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax4.transAxes)
                ax4.set_title('Distribution of Daily Returns')
            
            plt.tight_layout()
            
            # ذخیره نمودار
            filename = f"backtest_mt5_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 نمودارهای تحلیلی در {filename} ذخیره شدند")
            
        except Exception as e:
            logger.error(f"خطا در تولید نمودارها: {e}")
    
    def generate_report(self) -> str:
        """تولید گزارش کامل بکتست"""
        if not self.results:
            return "No backtest results available"
        
        report = []
        report.append("=" * 80)
        report.append("ENHANCED RSI STRATEGY - MT5 REAL DATA BACKTEST REPORT")
        report.append("=" * 80)
        
        # اطلاعات نماد و داده
        symbol = self.results.get('symbol', 'UNKNOWN')
        data_info = self.results.get('data_info', {})
        
        report.append(f"📊 SYMBOL & DATA INFO")
        report.append(f"Symbol: {symbol}")
        report.append(f"Period: {data_info.get('start_date')} to {data_info.get('end_date')}")
        report.append(f"Total Candles: {data_info.get('total_candles', 0)}")
        report.append(f"Price Range: {data_info.get('price_range', 'N/A')}")
        report.append(f"Price Change: {data_info.get('price_change_pct', 0):.2f}%")
        report.append(f"Market Volatility: {data_info.get('volatility', 0):.4f}")
        
        # اطلاعات کلی
        metrics = self.results.get('advanced_metrics', {})
        strategy_metrics = self.results.get('strategy_metrics', {})
        
        report.append("")
        report.append(f"📈 PERFORMANCE SUMMARY")
        report.append(f"Initial Capital: ${metrics.get('initial_capital', 0):,.2f}")
        report.append(f"Final Portfolio Value: ${metrics.get('final_portfolio_value', 0):,.2f}")
        report.append(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
        report.append(f"CAGR: {metrics.get('cagr', 0):.2f}%")
        report.append(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
        report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        report.append(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")
        
        report.append("")
        report.append("🎯 TRADING METRICS")
        report.append(f"Total Trades: {strategy_metrics.get('total_trades', 0)}")
        report.append(f"Winning Trades: {strategy_metrics.get('winning_trades', 0)}")
        report.append(f"Losing Trades: {strategy_metrics.get('losing_trades', 0)}")
        report.append(f"Win Rate: {strategy_metrics.get('win_rate', 0):.2f}%")
        report.append(f"Profit Factor: {strategy_metrics.get('profit_factor', 0):.2f}")
        report.append(f"Average Trade PnL: ${strategy_metrics.get('average_trade_pnl', 0):.2f}")
        
        report.append("")
        report.append("⚙️ STRATEGY PARAMETERS")
        params = self.results.get('parameters', {})
        important_params = [
            'rsi_period', 'rsi_base_oversold', 'rsi_base_overbought',
            'risk_per_trade', 'use_trend_filter', 'use_divergence',
            'enable_trailing_stop', 'min_position_size', 'max_trades_per_100'
        ]
        for param in important_params:
            if param in params:
                report.append(f"{param}: {params[param]}")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = None):
        """ذخیره نتایج بکتست"""
        if filename is None:
            symbol = self.results.get('symbol', 'UNKNOWN')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{symbol}_{timestamp}.json"
        
        try:
            # تبدیل به فرمت قابل ذخیره
            save_data = {
                'symbol': self.results.get('symbol'),
                'data_info': self.results.get('data_info', {}),
                'parameters': self.results.get('parameters', {}),
                'metrics': {
                    'strategy_metrics': self.results.get('strategy_metrics', {}),
                    'advanced_metrics': self.results.get('advanced_metrics', {})
                },
                'summary': {
                    'initial_capital': self.initial_capital,
                    'final_portfolio_value': self.equity_curve['portfolio_value'].iloc[-1] if self.equity_curve is not None else 0,
                    'total_trades': len(self.trades_df) if self.trades_df is not None else 0
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"💾 نتایج در {filename} ذخیره شد")
            
        except Exception as e:
            logger.error(f"خطا در ذخیره نتایج: {e}")

def run_mt5_backtest_example():
    """مثال اجرای بکتست با داده‌های واقعی MT5"""
    
    # پارامترهای استراتژی بهبود یافته
    strategy_params = {
 # غیرفعال برای سیگنال‌های بیشتر
    }
    
    # ایجاد بکتستر
    backtester = EnhancedRSIBacktestMT5(
        initial_capital=10000,
        commission=0.001,
        slippage=0.0005,
        enable_plotting=True,
        detailed_logging=True
    )
    
    try:
        # اجرای بکتست
        results = backtester.run_backtest(
            symbol="EURUSD",
            timeframe="H1",
            days_back=90,  # 3 ماه داده تاریخی
            strategy_params=strategy_params
        )
        
        # نمایش گزارش
        print(backtester.generate_report())
        
        # ذخیره نتایج
        backtester.save_results()
        
        return results
        
    except Exception as e:
        logger.error(f"خطا در اجرای بکتست: {e}")
        return None

def run_multiple_symbols_backtest():
    """اجرای بکتست روی چند نماد مختلف"""
    
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
    results = {}
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"🚀 اجرای بکتست برای {symbol}")
        print(f"{'='*50}")
        
        try:
            backtester = EnhancedRSIBacktestMT5(
                initial_capital=10000,
                commission=0.001,
                slippage=0.0005
            )
            
            result = backtester.run_backtest(
                symbol=symbol,
                timeframe="H1",
                days_back=60,  # 2 ماه داده
                strategy_params=None  # استفاده از پارامترهای پیش‌فرض
            )
            
            results[symbol] = {
                'metrics': backtester.results.get('advanced_metrics', {}),
                'trades_count': len(backtester.trades_df) if backtester.trades_df is not None else 0
            }
            
            print(backtester.generate_report())
            
        except Exception as e:
            print(f"❌ خطا در بکتست {symbol}: {e}")
            results[symbol] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    print("🎯 Enhanced RSI Strategy Backtesting with Real MT5 Data")
    print("=" * 60)
    
    # بررسی اتصال MT5
    if not MT5_AVAILABLE:
        print("❌ MetaTrader5 در دسترس نیست. لطفاً اطمینان حاصل کنید که:")
        print("   - MetaTrader5 نصب شده است")
        print("   - کتابخانه MetaTrader5 برای پایتون نصب شده است (pip install MetaTrader5)")
        print("   - متاتریدر باز است و به سرور متصل شده است")
        sys.exit(1)
    
    # اجرای مثال بکتست
    results = run_mt5_backtest_example()
    
    if results:
        print("\n✅ بکتست با موفقیت انجام شد!")
        print("📊 برای مشاهده نمودارها، فایل PNG ذخیره شده را بررسی کنید")
        print("💾 برای مشاهده جزئیات کامل، فایل JSON ذخیره شده را بررسی کنید")
    else:
        print("\n❌ بکتست با خطا مواجه شد!")
    
    # اجرای بکتست روی چند نماد (اختیاری)
    # print("\n🔄 اجرای بکتست روی چند نماد...")
    # multiple_results = run_multiple_symbols_backtest()