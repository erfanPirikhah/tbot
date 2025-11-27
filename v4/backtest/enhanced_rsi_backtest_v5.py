# backtest/enhanced_rsi_backtest_v5.py

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
import traceback
from scipy import stats
try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    sp = None

# اضافه کردن مسیرهای مورد نیاز
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.mt5_data import mt5_fetcher, MT5_AVAILABLE
from strategies.enhanced_rsi_strategy_v5 import EnhancedRsiStrategyV5, PositionType
from config.parameters import OPTIMIZED_PARAMS_V4, OPTIMIZED_PARAMS_V5, MARKET_CONDITION_PARAMS
from config.market_config import SYMBOL_MAPPING, TIMEFRAME_MAPPING, DEFAULT_CONFIG
from utils.logger import get_mongo_collection

warnings.filterwarnings('ignore')

# تنظیمات لاگ‌گیری - پیکربندی لاگ‌ها به سطح بالاتر واگذار می‌شود
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EnhancedRSIBacktestV5:
    """سیستم بکتست نسخه ۵ با تحلیل‌های پیشرفته و تمام بهبودهای تشخیص داده شده"""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.0003,
        slippage: float = 0.0001,
        enable_plotting: bool = True,
        detailed_logging: bool = True,
        save_trade_logs: bool = True,
        output_dir: str = os.path.join("logs", "backtests")
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.enable_plotting = enable_plotting
        self.detailed_logging = detailed_logging
        self.save_trade_logs = save_trade_logs
        self.output_dir = output_dir

        # ایجاد پوشه خروجی در صورت عدم وجود
        os.makedirs(self.output_dir, exist_ok=True)

        # نتایج بکتست
        self.results = {}
        self.trades_df = None
        self.equity_curve = None
        self.daily_returns = None
        self.monthly_returns = None
        self.yearly_returns = None

        logger.info("🚀 Enhanced RSI Backtest V5 Initialized with ALL improvements from diagnostic analysis")
        logger.info(f"💰 Initial Capital: ${initial_capital:,.2f}")
        logger.info(f"📊 Commission: {commission*100:.4f}%, Slippage: {slippage*100:.4f}%")

    def fetch_real_data_from_mt5(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "H1",
        days_back: int = 90
    ) -> pd.DataFrame:
        """دریافت داده از MT5 با مدیریت خطا - V5 with improved error handling"""
        logger.info(f"📥 Fetching data for {symbol} ({timeframe}) - {days_back} days")

        if not MT5_AVAILABLE or mt5_fetcher is None:
            raise ConnectionError("MT5 is not available")

        try:
            # محاسبه تعداد کندل مورد نیاز
            timeframe_minutes = {
                "M1": 1, "M5": 5, "M15": 15, "M30": 30,
                "H1": 60, "H4": 240, "D1": 1440, "W1": 10080
            }

            minutes_needed = days_back * 1440
            candles_needed = min(minutes_needed // timeframe_minutes.get(timeframe, 60), 10000)  # محدودیت معقول

            data = mt5_fetcher.fetch_market_data(symbol, timeframe, candles_needed)

            if data.empty:
                raise ValueError(f"No data received for {symbol}")

            # اطمینان از فرمت داده
            if 'open_time' in data.columns:
                data['open_time'] = pd.to_datetime(data['open_time'])
                data.set_index('open_time', inplace=True)

            # محاسبه اندیکاتورهای تکنیکال بهبود یافته
            data = self._calculate_technical_indicators(data)

            logger.info(f"✅ Fetched {len(data)} candles for {symbol}")
            logger.info(f"📊 Price range: {data['close'].min():.5f} - {data['close'].max():.5f}")
            logger.info(f"📈 Date range: {data.index[0]} to {data.index[-1]}")

            return data

        except Exception as e:
            logger.error(f"❌ Error fetching data: {e}")
            raise

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """محاسبه اندیکاتورهای تکنیکال - V5 with enhanced indicators"""
        logger.info("⚙️ Calculating enhanced technical indicators for V5 strategy")
        
        try:
            # RSI اصلی
            data = self._calculate_rsi(data, column_name='RSI')
            
            # ATR برای stop loss بهبود یافته
            data['ATR'] = self._calculate_atr(data)
            
            # اندیکاتورهای ایچی موفلا
            data['EMA_8'] = data['close'].ewm(span=8).mean()
            data['EMA_21'] = data['close'].ewm(span=21).mean()
            data['EMA_50'] = data['close'].ewm(span=50).mean()
            
            # Bollinger Bands
            data['BB_Middle'] = data['close'].rolling(20).mean()
            bb_std = data['close'].rolling(20).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
            data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
            
            # MACD
            exp1 = data['close'].ewm(span=12).mean()
            exp2 = data['close'].ewm(span=26).mean()
            data['MACD'] = exp1 - exp2
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            
            # ADX (Directional Movement)
            data = self._calculate_adx(data)
            
            # ایجاد اندیکاتورهای چندتایم‌فریم در صورت امکان
            # این اندیکاتورهای HTF در حین بکتست ایجاد خواهند شد، اما می‌توانیم ستون‌ها را پیش‌تعریف کنیم
            for tf in ['H4', 'D1', 'H1']:
                data[f'RSI_{tf}'] = np.nan
                data[f'EMA_21_{tf}'] = np.nan
                data[f'EMA_50_{tf}'] = np.nan
                data[f'TrendDir_{tf}'] = np.nan
            
            logger.info("✅ Enhanced technical indicators calculated successfully")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error calculating technical indicators: {e}")
            logger.error(traceback.format_exc())
            return data

    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14, column_name: str = 'RSI') -> pd.DataFrame:
        """محاسبه RSI با مدیریت خطا - V5"""
        try:
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

            rs = avg_gain / avg_loss
            data[column_name] = 100 - (100 / (1 + rs))
            data[column_name] = data[column_name].fillna(method='bfill').fillna(50)

            logger.debug(f"📊 {column_name} calculated")
            return data

        except Exception as e:
            logger.error(f"❌ Error calculating {column_name}: {e}")
            data[column_name] = 50
            return data

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """محاسبه ATR - V5"""
        try:
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr.fillna(method='bfill')

        except Exception as e:
            logger.error(f"❌ Error calculating ATR: {e}")
            return pd.Series([data['close'].iloc[0] * 0.01] * len(data), index=data.index)

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """محاسبه ADX - V5"""
        try:
            up_move = data['high'] - data['high'].shift(1)
            down_move = data['low'].shift(1) - data['low']
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            tr = self._calculate_atr(data, period)  # Already calculated ATR
            plus_di = (pd.Series(plus_dm).rolling(period).mean() / tr) * 100
            minus_di = (pd.Series(minus_dm).rolling(period).mean() / tr) * 100
            
            dx = (np.abs(plus_di - minus_di) / np.abs(plus_di + minus_di)) * 100
            adx = dx.rolling(period).mean()
            
            data['+DI'] = plus_di
            data['-DI'] = minus_di
            data['ADX'] = adx.fillna(20)  # Fill with neutral value
            
            return data

        except Exception as e:
            logger.error(f"❌ Error calculating ADX: {e}")
            data['ADX'] = 20
            return data

    def run_backtest(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "H1",
        days_back: int = 90,
        strategy_params: Dict[str, Any] = None,
        enable_detailed_logging: bool = True
    ) -> Dict[str, Any]:
        """اجرای بکتست - V5 با تمام بهبودهای تشخیص داده شده"""
        logger.info(f"🚀 Starting Enhanced RSI Backtest V5 for {symbol} ({timeframe}) - {days_back} days")
        logger.info("✨ This backtest incorporates ALL improvements from the diagnostic analysis:")
        logger.info("   • Enhanced MTF logic (majority alignment instead of all alignment)")
        logger.info("   • Active trend filter with multiple confirmations")
        logger.info("   • Dynamic risk management based on market regime")
        logger.info("   • Improved contradiction detection system")
        logger.info("   • Adaptive position sizing")
        logger.info("   • Better exit logic with optimized trailing stops")

        # پارامترهای پیش‌فرض V5
        if strategy_params is None:
            strategy_params = OPTIMIZED_PARAMS_V5.copy()
            logger.info("📋 Using OPTIMIZED_PARAMS_V5 with all diagnostic improvements")

        # ایجاد استراتژی V5
        strategy = EnhancedRsiStrategyV5(**strategy_params)

        try:
            # دریافت داده
            logger.info("📥 Fetching market data...")
            data = self.fetch_real_data_from_mt5(symbol, timeframe, days_back)

            # ایجاد تاریخچه معاملات
            trades = []
            portfolio_values = []
            signals_log = []

            logger.info("🔄 Starting backtest simulation...")
            
            # حلقه اصلی بکتست
            for i in range(len(data)):
                if i < 50:  # صبر برای ایجاد اندیکاتورهای کافی
                    continue

                current_data = data.iloc[:i+1].copy()

                # تولید سیگنال
                signal = strategy.generate_signal(current_data, i)
                
                # ذخیره سیگنال برای تحلیل
                signals_log.append({
                    'timestamp': current_data.index[-1],
                    'signal': signal.get('action', 'HOLD'),
                    'price': signal.get('price', current_data['close'].iloc[-1]),
                    'reason': signal.get('reason', ''),
                    'regime': signal.get('regime', 'UNKNOWN'),
                    'contradiction_score': signal.get('contradiction_score', 0.0)
                })

                # پردازش سیگنال‌های ورود
                if signal['action'] in ['BUY', 'SELL']:
                    trade_result = self._process_entry(strategy, signal, current_data, i, symbol)
                    if trade_result:
                        trades.append(trade_result)

                # پردازش سیگنال‌های خروج
                elif signal['action'] in ['EXIT', 'PARTIAL_EXIT']:
                    exit_result = self._process_exit(strategy, signal, current_data, i, symbol)
                    if exit_result:
                        trades.append(exit_result)

                # ثبت ارزش پورتفو
                portfolio_value = strategy._portfolio_value
                portfolio_values.append({
                    'timestamp': current_data.index[-1],
                    'portfolio_value': portfolio_value,
                    'price': current_data['close'].iloc[-1],
                    'position': strategy._position.value
                })

                # نمایش پیشرفت
                if (i + 1) % max(1, len(data) // 20) == 0:  # Show progress 20 times
                    progress = (i + 1) / len(data) * 100
                    logger.info(f"📈 Progress: {progress:.1f}% - Portfolio: ${portfolio_value:.2f}")

            # کامپایل نتایج
            logger.info("📊 Compiling backtest results...")
            self._compile_results(strategy, trades, portfolio_values, data, symbol)
            self._calculate_performance_metrics(strategy, data)
            
            # تولید گزارش
            report = self.generate_comprehensive_report()
            
            # ذخیره نتایج
            if self.save_trade_logs:
                self._save_results_to_file(symbol, timeframe, strategy, report)

            if self.enable_plotting:
                self._generate_enhanced_plots(data, symbol, timeframe, strategy)

            logger.info("✅ Enhanced RSI Backtest V5 completed successfully!")
            return self.results

        except Exception as e:
            logger.error(f"❌ Error in backtest: {e}")
            logger.error(traceback.format_exc())
            raise

    def _process_entry(self, strategy, signal, data, index, symbol):
        """پردازش سیگنال ورود - V5"""
        try:
            current_price = data['close'].iloc[-1]

            # اعمال اسلیپیج
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
                'symbol': symbol,
                'price': execution_price,
                'position_size': position_size,
                'stop_loss': signal.get('stop_loss', 0),
                'take_profit': signal.get('take_profit', 0),
                'commission': commission,
                'rsi': data['RSI'].iloc[-1] if 'RSI' in data.columns else 50,
                'reason': signal.get('reason', ''),
                'regime': signal.get('regime', 'UNKNOWN'),
                'contradiction_score': signal.get('contradiction_score', 0),
                'portfolio_value': strategy._portfolio_value - commission
            }

            if self.detailed_logging:
                logger.info(f"🟢 {signal['action']} at {execution_price:.5f}, Size: {position_size:.0f}, Regime: {trade_record['regime']}, Contr: {trade_record['contradiction_score']:.3f}")

            return trade_record

        except Exception as e:
            logger.error(f"❌ Error processing entry: {e}")
            return None

    def _process_exit(self, strategy, signal, data, index, symbol):
        """پردازش سیگنال خروج - V5"""
        try:
            current_price = data['close'].iloc[-1]

            # اعمال اسلیپیج
            if strategy._position == PositionType.LONG:
                execution_price = current_price * (1 - self.slippage)
            else:
                execution_price = current_price * (1 + self.slippage)

            # اعمال کارمزد
            position_size = strategy._current_trade.quantity if strategy._current_trade else 0
            commission = execution_price * position_size * self.commission

            trade_record = {
                'timestamp': data.index[-1],
                'action': signal['action'],
                'symbol': symbol,
                'price': execution_price,
                'position_size': position_size,
                'commission': commission,
                'pnl_percentage': signal.get('pnl_percentage', 0),
                'pnl_amount': signal.get('pnl_amount', 0),
                'exit_reason': signal.get('exit_reason', ''),
                'portfolio_value': strategy._portfolio_value - commission
            }

            if self.detailed_logging:
                logger.info(f"🔴 {signal['action']} at {execution_price:.5f}, PnL: {signal.get('pnl_percentage', 0):.2f}%")

            return trade_record

        except Exception as e:
            logger.error(f"❌ Error processing exit: {e}")
            return None

    def _compile_results(self, strategy, trades, portfolio_values, data, symbol):
        """کامپایل نتایج - V5"""
        self.trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        self.equity_curve = pd.DataFrame(portfolio_values)

        if not self.equity_curve.empty:
            self.equity_curve['timestamp'] = pd.to_datetime(self.equity_curve['timestamp'])
            self.equity_curve.set_index('timestamp', inplace=True)

        # محاسبه بازدهی روزانه
        if not self.equity_curve.empty:
            self.daily_returns = self.equity_curve['portfolio_value'].pct_change().dropna()
        
        # اضافه کردن متادیتای افزوده V5
        self.results = {
            'trades': self.trades_df,
            'equity_curve': self.equity_curve,
            'strategy_metrics': strategy.get_performance_metrics(),  # V5 metrics with enhanced info
            'symbol': symbol,
            'data_info': {
                'start_date': data.index[0],
                'end_date': data.index[-1],
                'total_candles': len(data),
                'price_range': f"{data['close'].min():.5f} - {data['close'].max():.5f}",
                'price_change_pct': ((data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * 100)
            },
            'enhanced_metrics': {
                'v5_improvements_applied': True,
                'diagnostic_based_enhancements': [
                    'Enhanced MTF analysis',
                    'Active trend filtering',
                    'Dynamic risk management',
                    'Contradiction detection',
                    'Regime-aware trading'
                ]
            }
        }

    def _calculate_performance_metrics(self, strategy, data):
        """محاسبه معیارهای عملکرد - V5 با تحلیل‌های پیشرفته"""
        try:
            logger.info("📊 Calculating enhanced V5 performance metrics...")
            
            # دریافت معیارهای استراتژی V5 (که شامل تحلیل تناقضات و رژیم است)
            strategy_metrics = strategy.get_performance_metrics()
            
            # معیارهای اضافی V5
            additional_metrics = {}

            # تحلیل بر اساس رژیم بازار
            regime_performance = strategy_metrics.get('regime_performance', {})
            additional_metrics['regime_analysis'] = {
                'performance_by_regime': regime_performance,
                'best_regime': max(regime_performance.items(), key=lambda x: x[1]['win_rate']) if regime_performance else ('UNKNOWN', {}),
                'worst_regime': min(regime_performance.items(), key=lambda x: x[1]['win_rate']) if regime_performance else ('UNKNOWN', {})
            }

            # تحلیل بر اساس نمره تناقض
            avg_contradiction = strategy_metrics.get('avg_contradiction_score', 0)
            low_contradiction_trades = strategy_metrics.get('trade_quality_insights', {}).get('low_contradiction_trades', 0)
            high_contradiction_trades = strategy_metrics.get('trade_quality_insights', {}).get('high_contradiction_trades', 0)
            
            additional_metrics['contradiction_analysis'] = {
                'average_contradiction_score': avg_contradiction,
                'low_contradiction_trades': low_contradiction_trades,
                'high_contradiction_trades': high_contradiction_trades,
                'contradiction_impact': 'Analyzed' if avg_contradiction > 0 else 'Not available'
            }

            # تحلیل مکانیزم‌های بهبود یافته
            additional_metrics['improvement_impact'] = {
                'mtf_filtering_effectiveness': 'Enabled with majority alignment',
                'trend_filtering_active': strategy.enable_trend_filter,
                'dynamic_risk_management': 'Active',
                'regime_aware_trading': 'Active',
                'contradiction_prevention': 'Active'
            }

            self.results['performance_metrics'] = {
                **strategy_metrics,
                **additional_metrics
            }

            logger.info(f"📈 V5 Performance Summary:")
            logger.info(f"   Total Trades: {strategy_metrics.get('total_trades', 0)}")
            logger.info(f"   Win Rate: {strategy_metrics.get('win_rate', 0):.2f}%")
            logger.info(f"   Total P&L: ${strategy_metrics.get('total_pnl', 0):,.2f}")
            logger.info(f"   Sharpe Ratio: {strategy_metrics.get('sharpe_ratio', 0):.4f}")
            logger.info(f"   Max Drawdown: {strategy_metrics.get('max_drawdown', 0):.2f}%")
            logger.info(f"   Avg Contradiction Score: {avg_contradiction:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Error calculating enhanced performance metrics: {e}")
            self.results['performance_metrics'] = strategy.get_performance_metrics()

    def generate_comprehensive_report(self) -> str:
        """تولید گزارش جامع - V5 با تمام بهبودهای تشخیص داده شده"""
        if not self.results:
            return "No results available"

        metrics = self.results.get('performance_metrics', {})
        strategy_metrics = self.results.get('strategy_metrics', {})
        data_info = self.results.get('data_info', {})

        # استخراج معیارهای V5
        total_trades = metrics.get('total_trades', 0)
        win_rate = metrics.get('win_rate', 0)
        total_pnl = metrics.get('total_pnl', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        avg_contradiction = metrics.get('avg_contradiction_score', 0)
        
        # تحلیل رژیم
        regime_analysis = metrics.get('regime_analysis', {})
        best_regime = regime_analysis.get('best_regime', ('UNKNOWN', {}))
        worst_regime = regime_analysis.get('worst_regime', ('UNKNOWN', {}))
        
        # تحلیل تناقض
        contradition_analysis = metrics.get('contradiction_analysis', {})

        report_lines = [
            "=" * 80,
            "ENHANCED RSI STRATEGY V5 - COMPREHENSIVE BACKTEST REPORT",
            "Incorporating ALL diagnostic analysis improvements:",
            "• Enhanced MTF with majority alignment (not all-or-nothing)",
            "• Active trend filtering with multiple confirmations", 
            "• Dynamic risk management based on market regime",
            "• Advanced contradiction detection and prevention",
            "• Regime-aware trading parameters",
            "=" * 80,
            f"Symbol: {self.results.get('symbol', 'N/A')}",
            f"Period: {data_info.get('start_date', 'N/A')} to {data_info.get('end_date', 'N/A')}",
            f"Data Points: {data_info.get('total_candles', 'N/A')} candles",
            "",
            "PERFORMANCE SUMMARY:",
            f"Initial Capital: ${self.initial_capital:,.2f}",
            f"Final Portfolio Value: ${metrics.get('current_portfolio_value', self.initial_capital):,.2f}",
            f"Total Return: {((metrics.get('current_portfolio_value', self.initial_capital) / self.initial_capital) - 1) * 100:.2f}%",
            f"Total P&L: ${total_pnl:,.2f}",
            f"Max Drawdown: {max_drawdown:.2f}%",
            "",
            "TRADING METRICS:",
            f"Total Trades: {total_trades}",
            f"Winning Trades: {metrics.get('winning_trades', 0)}",
            f"Losing Trades: {metrics.get('losing_trades', 0)}",
            f"Win Rate: {win_rate:.2f}%",
            f"Profit Factor: {metrics.get('profit_factor', 0):.2f}",
            f"Sharpe Ratio: {sharpe_ratio:.4f}",
            f"Consecutive Losses: {metrics.get('consecutive_losses', 0)}",
            f"Average Trade Return: {metrics.get('avg_trade_return', 0):.2f}%",
            "",
            "V5 ENHANCEMENT ANALYSIS:",
            f"Average Contradiction Score: {avg_contradiction:.3f}",
            f"Low Contradiction Trades: {contradition_analysis.get('low_contradiction_trades', 0)}",
            f"High Contradiction Trades: {contradition_analysis.get('high_contradiction_trades', 0)}",
            ""
        ]

        # اضافه کردن تحلیل رژیم
        if best_regime[0] != 'UNKNOWN':
            report_lines.extend([
                "MARKET REGIME ANALYSIS:",
                f"Best Performing Regime: {best_regime[0]} (Win Rate: {best_regime[1].get('win_rate', 0):.2f}%)",
                f"Worst Performing Regime: {worst_regime[0]} (Win Rate: {worst_regime[1].get('win_rate', 0):.2f}%)",
                ""
            ])

        # اضافه کردن اطلاعات افزوده V5
        report_lines.extend([
            "V5 IMPROVEMENT IMPACT:",
            "✓ MTF: Changed from 'all must align' to 'majority alignment' approach",
            "✓ Trend Filter: Now active with multi-indicator confirmation", 
            "✓ Risk Management: Dynamic based on volatility and regime",
            "✓ Entry Conditions: Improved with contradiction detection",
            "✓ Exit Logic: Optimized trailing stop activation",
            "✓ Win Rate Target: Improved from ~31% to expected higher values",
            "",
            f"Backtest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ])

        report = "\n".join(report_lines)
        return report

    def _save_results_to_file(self, symbol, timeframe, strategy, report):
        """ذخیره نتایج در فایل - V5"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                self.output_dir, 
                f"v5_enhanced_backtest_{symbol}_{timeframe}_{timestamp}.json"
            )
            
            # آماده‌سازی داده‌های خروجی
            strategy_config = strategy.__dict__.copy()
            strategy_config.pop('_position', None)  # Remove complex object
            strategy_config['_current_trade'] = None  # Remove complex object
            strategy_config['_trade_history'] = len(strategy._trade_history)  # Just count
            strategy_config['_signal_log'] = len(strategy._signal_log)  # Just count

            output_data = {
                'timestamp': timestamp,
                'symbol': symbol,
                'timeframe': timeframe,
                'strategy_config': strategy_config,
                'performance_metrics': self.results.get('performance_metrics', {}),
                'report_summary': report,
                'trade_count': len(self.results.get('trades', []))
            }
                
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
                
            logger.info(f"💾 Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"❌ Error saving results: {e}")

    def _generate_enhanced_plots(self, data, symbol, timeframe, strategy):
        """تولید نمودارهای بهبود یافته - V5"""
        if not PLOTLY_AVAILABLE:
            logger.warning("⚠️ Plotly not available, skipping enhanced plots generation")
            return

        try:
            logger.info("📊 Generating enhanced V5 plots...")

            # ایجاد نمودارهای چندگانه با Plotly
            fig = sp.make_subplots(
                rows=4, cols=2,
                subplot_titles=(
                    'Price & RSI', 'Equity Curve',
                    'Drawdown', 'P&L Distribution',
                    'Win Rate by Regime', 'Contradiction Impact',
                    'Monthly Returns', 'Risk Metrics'
                ),
                specs=[
                    [{"secondary_y": True}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "histogram"}],
                    [{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "scatter"}]
                ],
                vertical_spacing=0.08
            )

            # نمودار 1: قیمت و RSI
            if not data.empty and 'RSI' in data.columns:
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['close'], name='Price', line=dict(color='blue')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='red'), yaxis='y2'),
                    row=1, col=1, secondary_y=True
                )

            # نمودار 2: منحنی سرمایه
            if self.equity_curve is not None and not self.equity_curve.empty:
                fig.add_trace(
                    go.Scatter(
                        x=self.equity_curve.index,
                        y=self.equity_curve['portfolio_value'],
                        name='Equity Curve',
                        line=dict(color='green', width=2)
                    ),
                    row=1, col=2
                )

            # نمودار 3: کشش سرمایه (Drawdown)
            if self.equity_curve is not None and not self.equity_curve.empty:
                portfolio_values = self.equity_curve['portfolio_value']
                rolling_max = portfolio_values.expanding().max()
                drawdown = (portfolio_values - rolling_max) / rolling_max * 100
                fig.add_trace(
                    go.Bar(x=drawdown.index, y=drawdown.values, name='Drawdown', marker_color='red'),
                    row=2, col=1
                )

            # نمودار 4: توزیع P&L
            if self.trades_df is not None and not self.trades_df.empty and 'pnl_percentage' in self.trades_df.columns:
                pnl_data = self.trades_df['pnl_percentage'].dropna()
                if not pnl_data.empty:
                    fig.add_trace(
                        go.Histogram(x=pnl_data, name='P&L Distribution', nbinsx=20),
                        row=2, col=2
                    )

            # ذخیره نمودار
            filename = os.path.join(
                self.output_dir,
                f"v5_enhanced_charts_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )
            fig.update_layout(height=1200, showlegend=True, title_text=f"Enhanced RSI V5 Analysis: {symbol} ({timeframe})")
            fig.write_html(filename)

            logger.info(f"📊 Enhanced charts saved to {filename}")

        except Exception as e:
            logger.error(f"❌ Error generating enhanced plots: {e}")

# تابع میانبر برای اجرای سریع
def run_enhanced_v5_backtest(
    symbol: str = "EURUSD",
    timeframe: str = "H1", 
    days_back: int = 90,
    initial_capital: float = 10000.0,
    custom_params: Dict[str, Any] = None
):
    """تابع میانبر برای اجرای سریع بکتست V5"""
    logger.info(f"🚀 Running quick Enhanced RSI V5 backtest for {symbol}")
    
    backtester = EnhancedRSIBacktestV5(initial_capital=initial_capital)
    
    params = custom_params or OPTIMIZED_PARAMS_V5.copy()
    
    results = backtester.run_backtest(
        symbol=symbol,
        timeframe=timeframe,
        days_back=days_back,
        strategy_params=params
    )
    
    report = backtester.generate_comprehensive_report()
    print(report)
    
    return results

if __name__ == "__main__":
    # مثال از اجرای بکتست V5
    print("🚀 Running Enhanced RSI Strategy V5 Backtest")
    print("="*60)
    
    # اجرای یک بکتست سریع
    results = run_enhanced_v5_backtest(
        symbol="EURUSD",
        timeframe="H1", 
        days_back=60,
        initial_capital=10000.0
    )
    
    print("\n✅ Enhanced RSI V5 Backtest completed!")