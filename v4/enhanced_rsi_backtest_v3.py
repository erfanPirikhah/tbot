# enhanced_rsi_backtest_v3_fixed.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.mt5_data import mt5_fetcher, MT5_AVAILABLE
from enhanced_rsi_strategy_v3 import EnhancedRsiStrategyV3, PositionType

warnings.filterwarnings('ignore')

# تنظیمات لاگ‌گیری بدون emoji
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'backtest_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedRSIBacktestV3:
    """سیستم بکتست نسخه ۳ با لاگ‌های پیشرفته"""
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.0005,
        slippage: float = 0.0002,
        enable_plotting: bool = True,
        detailed_logging: bool = True
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.enable_plotting = enable_plotting
        self.detailed_logging = detailed_logging
        
        self.results = {}
        self.trades_df = None
        self.equity_curve = None
        self.daily_returns = None
        
        logger.info(">>> Enhanced RSI Backtest V3 Initialized")

    def fetch_real_data_from_mt5(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "H1",
        days_back: int = 90
    ) -> pd.DataFrame:
        """دریافت داده از MT5 با مدیریت خطا"""
        logger.info(f">>> دریافت داده از MT5 برای {symbol} ({timeframe})")
        
        if not MT5_AVAILABLE or mt5_fetcher is None:
            raise ConnectionError("MT5 در دسترس نیست")
        
        try:
            # محاسبه تعداد کندل مورد نیاز
            timeframe_minutes = {
                "M1": 1, "M5": 5, "M15": 15, "M30": 30,
                "H1": 60, "H4": 240, "D1": 1440, "W1": 10080
            }
            
            minutes_needed = days_back * 1440
            candles_needed = minutes_needed // timeframe_minutes.get(timeframe, 60)
            candles_needed = min(candles_needed, 5000)  # محدودیت برای جلوگیری از overload
            
            data = mt5_fetcher.fetch_market_data(symbol, timeframe, candles_needed)
            
            if data.empty:
                raise ValueError(f"هیچ داده‌ای برای {symbol} دریافت نشد")
            
            # اطمینان از فرمت داده
            if 'open_time' in data.columns:
                data['open_time'] = pd.to_datetime(data['open_time'])
                data.set_index('open_time', inplace=True)
            
            # محاسبه RSI
            data = self._calculate_rsi(data)
            
            logger.info(f">>> دریافت {len(data)} کندل از {symbol}")
            logger.info(f">>> بازه: {data.index[0]} تا {data.index[-1]}")
            logger.info(f">>> قیمت: {data['close'].min():.4f} - {data['close'].max():.4f}")
            
            return data
            
        except Exception as e:
            logger.error(f">>> خطا در دریافت داده: {e}")
            raise

    def _process_exit(self, strategy, signal, data, index):
        """پردازش سیگنال خروج"""
        try:
            current_price = data['close'].iloc[-1]
            
            # اعمال slippage
            if strategy._position == PositionType.LONG:
                execution_price = current_price * (1 - self.slippage)
            else:
                execution_price = current_price * (1 + self.slippage)
            
            # اعمال کارمزد
            position_size = strategy._current_trade.quantity if strategy._current_trade else 0
            commission = execution_price * position_size * self.commission
            
            trade_record = {
                'timestamp': data.index[-1],
                'action': 'EXIT',
                'symbol': 'EURUSD',
                'price': execution_price,
                'position_size': position_size,
                'commission': commission,
                'pnl_percentage': signal.get('pnl_percentage', 0),
                'pnl_amount': signal.get('pnl_amount', 0),
                'exit_reason': signal.get('exit_reason', ''),
                'portfolio_value': strategy._portfolio_value - commission  # PnL قبلاً محاسبه شده
            }
            
            if self.detailed_logging:
                logger.info(f">>> EXIT at {execution_price:.4f}, PnL: {signal.get('pnl_percentage', 0):.2f}%")
            
            return trade_record
            
        except Exception as e:
            logger.error(f"خطا در پردازش خروج: {e}")
            return None
            
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """محاسبه RSI"""
        try:
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
            
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))
            data['RSI'] = data['RSI'].fillna(method='bfill').fillna(50)
            
            return data
        except Exception as e:
            logger.error(f"خطا در محاسبه RSI: {e}")
            data['RSI'] = 50
            return data

    def run_backtest(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "H1",
        days_back: int = 90,
        strategy_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """اجرای بکتست"""
        logger.info(">>> شروع بکتست نسخه ۳")
        
        # پارامترهای پیش‌فرض
        if strategy_params is None:
            strategy_params = {
                'rsi_period': 11,  # کاهش دوره برای سیگنال‌های سریع‌تر
                'rsi_oversold': 32,  # کاهش ناحیه اشباع فروش
                'rsi_overbought': 68,  # افزایش ناحیه اشباع خرید
                'rsi_entry_buffer': 2,
                'risk_per_trade': 0.01,  # کاهش ریسک
                'stop_loss_atr_multiplier': 1.5,  # افزایش فاصله استاپ
                'take_profit_ratio': 2.5,  # افزایش نسبت سود به زیان
                'min_candles_between': 5,  # افزایش فاصله بین معاملات
                'enable_trend_filter': True,
                'enable_short_trades': True,
                'enable_trailing_stop': True,
                'trailing_activation_percent': 0.3,  # فعال‌سازی زودتر تریلینگ استاپ
                'max_consecutive_losses': 3,  # حساسیت بیشتر به ضررهای متوالی
                'pause_after_losses': 15  # استراحت طولانی‌تر
            }
        
        try:
            # دریافت داده
            data = self.fetch_real_data_from_mt5(symbol, timeframe, days_back)
            
            # ایجاد استراتژی
            strategy = EnhancedRsiStrategyV3(**strategy_params)
            
            # اجرای بکتست
            trades = []
            portfolio_values = []
            
            for i in range(len(data)):
                if i < 50:  # صبر برای داده کافی
                    continue
                    
                current_data = data.iloc[:i+1].copy()
                
                # تولید سیگنال
                signal = strategy.generate_signal(current_data, i)
                
                # شبیه‌سازی اجرا
                if signal['action'] in ['BUY', 'SELL']:
                    trade_result = self._execute_trade(strategy, signal, current_data, i)
                    if trade_result:
                        trades.append(trade_result)
                        
                elif signal['action'] == 'EXIT':
                    exit_result = self._process_exit(strategy, signal, current_data, i)
                    if exit_result:
                        trades.append(exit_result)
                
                # ثبت ارزش پورتفو
                portfolio_value = self._calculate_portfolio_value(strategy, current_data)
                portfolio_values.append({
                    'timestamp': current_data.index[-1],
                    'portfolio_value': portfolio_value,
                    'price': current_data['close'].iloc[-1]
                })
            
            # کامپایل نتایج
            self._compile_results(strategy, trades, portfolio_values, data, symbol)
            self._calculate_performance_metrics()
            
            if self.enable_plotting:
                self._generate_plots(data, symbol, timeframe)
            
            logger.info(">>> بکتست با موفقیت انجام شد")
            return self.results
            
        except Exception as e:
            logger.error(f">>> خطا در اجرای بکتست: {e}")
            raise

    def _execute_trade(self, strategy, signal, data, index):
        """شبیه‌سازی اجرای معامله"""
        try:
            current_price = data['close'].iloc[-1]
            
            # اعمال slippage
            if signal['action'] == 'BUY':
                execution_price = current_price * (1 + self.slippage)
            else:
                execution_price = current_price * (1 - self.slippage)
            
            # اعمال کارمزد
            position_size = signal.get('position_size', 0)
            commission = execution_price * position_size * self.commission
            
            trade_record = {
                'timestamp': data.index[-1],
                'action': signal['action'],
                'symbol': 'EURUSD',
                'price': execution_price,
                'position_size': position_size,
                'stop_loss': signal.get('stop_loss', 0),
                'take_profit': signal.get('take_profit', 0),
                'commission': commission,
                'rsi': data['RSI'].iloc[-1],
                'reason': signal.get('reason', ''),
                'portfolio_value': strategy._portfolio_value - commission
            }
            
            if self.detailed_logging:
                logger.info(f">>> {signal['action']} at {execution_price:.4f}, Size: {position_size:.0f}")
            
            return trade_record
            
        except Exception as e:
            logger.error(f"خطا در اجرای معامله: {e}")
            return None

    def _calculate_portfolio_value(self, strategy, data):
        """محاسبه ارزش پورتفو"""
        if strategy._current_trade is None:
            return strategy._portfolio_value
        
        current_price = data['close'].iloc[-1]
        entry_price = strategy._current_trade.entry_price
        quantity = strategy._current_trade.quantity
        
        if strategy._position == PositionType.LONG:
            unrealized_pnl = (current_price - entry_price) * quantity
        else:
            unrealized_pnl = (entry_price - current_price) * quantity
        
        return strategy._portfolio_value + unrealized_pnl

    def _compile_results(self, strategy, trades, portfolio_values, data, symbol):
        """کامپایل نتایج"""
        self.trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        self.equity_curve = pd.DataFrame(portfolio_values)
        
        if not self.equity_curve.empty:
            self.equity_curve['timestamp'] = pd.to_datetime(self.equity_curve['timestamp'])
            self.equity_curve.set_index('timestamp', inplace=True)
        
        self.results = {
            'trades': self.trades_df,
            'equity_curve': self.equity_curve,
            'strategy_metrics': strategy.get_performance_metrics(),
            'signal_log': strategy.get_signal_log(),
            'symbol': symbol,
            'data_info': {
                'start_date': data.index[0],
                'end_date': data.index[-1],
                'total_candles': len(data),
                'price_range': f"{data['close'].min():.4f} - {data['close'].max():.4f}",
                'price_change_pct': ((data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * 100)
            }
        }

    def _calculate_performance_metrics(self):
        """محاسبه معیارهای عملکرد"""
        if self.equity_curve is None or self.equity_curve.empty:
            return
        
        portfolio_values = self.equity_curve['portfolio_value']
        total_return = (portfolio_values.iloc[-1] - self.initial_capital) / self.initial_capital * 100
        
        # محاسبه drawdown
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max * 100
        max_drawdown = drawdowns.min()
        
        # محاسبه معاملات
        total_trades = len(self.trades_df) if self.trades_df is not None else 0
        
        self.results['performance_metrics'] = {
            'total_return_pct': round(total_return, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'total_trades': total_trades,
            'final_portfolio_value': round(portfolio_values.iloc[-1], 2),
            'initial_capital': self.initial_capital
        }

    def _generate_plots(self, data, symbol, timeframe):
        """تولید نمودارها"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'RSI Strategy V3 - {symbol} ({timeframe})', fontsize=16)
            
            # نمودار ۱: قیمت و RSI
            ax1 = axes[0, 0]
            ax1.plot(data.index, data['close'], label='Price', color='blue', linewidth=1)
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_title('Price Chart')
            
            ax1_rsi = ax1.twinx()
            ax1_rsi.plot(data.index, data['RSI'], color='red', linewidth=1, alpha=0.7)
            ax1_rsi.axhline(y=35, color='green', linestyle='--', alpha=0.5)
            ax1_rsi.axhline(y=65, color='red', linestyle='--', alpha=0.5)
            ax1_rsi.set_ylabel('RSI')
            ax1_rsi.set_ylim(0, 100)
            
            # نمودار ۲: منحنی سرمایه
            if self.equity_curve is not None and not self.equity_curve.empty:
                ax2 = axes[0, 1]
                ax2.plot(self.equity_curve.index, self.equity_curve['portfolio_value'], 
                        color='green', linewidth=2)
                ax2.set_ylabel('Portfolio Value')
                ax2.grid(True, alpha=0.3)
                ax2.set_title('Equity Curve')
            
            # نمودار ۳: Drawdown
            if self.equity_curve is not None and not self.equity_curve.empty:
                ax3 = axes[1, 0]
                rolling_max = self.equity_curve['portfolio_value'].expanding().max()
                drawdown = (self.equity_curve['portfolio_value'] - rolling_max) / rolling_max * 100
                ax3.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
                ax3.set_ylabel('Drawdown %')
                ax3.set_title('Portfolio Drawdown')
                ax3.grid(True, alpha=0.3)
            
            # نمودار ۴: توزیع معاملات
            ax4 = axes[1, 1]
            if self.trades_df is not None and not self.trades_df.empty:
                trades_pnl = [t.get('pnl_percentage', 0) for t in self.results['trades'].to_dict('records')]
                if trades_pnl:
                    ax4.hist(trades_pnl, bins=20, alpha=0.7, color='blue', edgecolor='black')
                    ax4.axvline(np.mean(trades_pnl), color='red', linestyle='--', label=f'Mean: {np.mean(trades_pnl):.2f}%')
                    ax4.legend()
            ax4.set_xlabel('PnL %')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Trade PnL Distribution')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = f"backtest_v3_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f">>> نمودارها در {filename} ذخیره شد")
            
        except Exception as e:
            logger.error(f"خطا در تولید نمودار: {e}")

    def generate_report(self) -> str:
        """تولید گزارش"""
        if not self.results:
            return "No results available"
        
        metrics = self.results.get('performance_metrics', {})
        strategy_metrics = self.results.get('strategy_metrics', {})
        
        report = [
            "=" * 60,
            "ENHANCED RSI STRATEGY V3 - BACKTEST REPORT",
            "=" * 60,
            f"Symbol: {self.results.get('symbol', 'N/A')}",
            f"Period: {self.results.get('data_info', {}).get('start_date')} to {self.results.get('data_info', {}).get('end_date')}",
            "",
            "PERFORMANCE SUMMARY:",
            f"Initial Capital: ${metrics.get('initial_capital', 0):,.2f}",
            f"Final Portfolio Value: ${metrics.get('final_portfolio_value', 0):,.2f}",
            f"Total Return: {metrics.get('total_return_pct', 0):.2f}%",
            f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%",
            "",
            "TRADING METRICS:",
            f"Total Trades: {strategy_metrics.get('total_trades', 0)}",
            f"Winning Trades: {strategy_metrics.get('winning_trades', 0)}",
            f"Win Rate: {strategy_metrics.get('win_rate', 0):.2f}%",
            f"Total PnL: ${strategy_metrics.get('total_pnl', 0):.2f}",
            f"Profit Factor: {strategy_metrics.get('profit_factor', 0):.2f}",
            f"Consecutive Losses: {strategy_metrics.get('consecutive_losses', 0)}",
            "",
            "DATA INFO:",
            f"Total Candles: {self.results.get('data_info', {}).get('total_candles', 0)}",
            f"Price Change: {self.results.get('data_info', {}).get('price_change_pct', 0):.2f}%"
        ]
        
        return "\n".join(report)





def run_optimized_backtest():
    """اجرای بکتست بهینه‌شده"""
    logger.info(">>> اجرای بکتست بهینه‌شده")
    
    strategy_params = {
        'rsi_period': 14,
        'rsi_oversold': 35,
        'rsi_overbought': 65,
        'rsi_entry_buffer': 3,
        'risk_per_trade': 0.015,
        'stop_loss_atr_multiplier': 1.2,
        'take_profit_ratio': 2.0,
        'min_position_size': 500,
        'max_trades_per_100': 25,
        'min_candles_between': 3,
        'enable_trend_filter': True,
        'enable_short_trades': True,
        'enable_trailing_stop': True
    }
    
    backtester = EnhancedRSIBacktestV3(
        initial_capital=10000,
        commission=0.0005,
        slippage=0.0002,
        detailed_logging=True
    )
    
    try:
        results = backtester.run_backtest(
            symbol="EURUSD",
            timeframe="H1",
            days_back=90,
            strategy_params=strategy_params
        )
        
        print(backtester.generate_report())
        return results
        
    except Exception as e:
        logger.error(f">>> خطا در اجرای بکتست: {e}")
        return None



if __name__ == "__main__":
    results = run_optimized_backtest()
    if results:
        print("\n>>> بکتست با موفقیت انجام شد!")