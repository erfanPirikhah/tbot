# backtest/enhanced_rsi_backtest_v4.py

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

# اضافه کردن مسیرهای مورد نیاز
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.mt5_data import mt5_fetcher, MT5_AVAILABLE
from strategies.enhanced_rsi_strategy_v4 import EnhancedRsiStrategyV4, PositionType
from config.parameters import OPTIMIZED_PARAMS_V4
from config.market_config import SYMBOL_MAPPING, TIMEFRAME_MAPPING, DEFAULT_CONFIG

warnings.filterwarnings('ignore')

# تنظیمات لاگ‌گیری - پیکربندی لاگ‌ها به سطح بالاتر واگذار می‌شود
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EnhancedRSIBacktestV4:
    """سیستم بکتست نسخه ۴ با تحلیل‌های پیشرفته"""
    
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
        os.makedirs(self.output_dir, exist_ok=True)

        # ایجاد زیرپوشه‌های ساختارمند برای خروجی‌ها
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.trades_dir = os.path.join(self.output_dir, "trades")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.trades_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # اتصال فایل‌هندر به لاگر این ماژول در مسیر مشخص‌شده
        try:
            log_path = os.path.join(self.logs_dir, f'backtest_v4_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
                fh = logging.FileHandler(log_path, encoding='utf-8')
                fh.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                fh.setFormatter(formatter)
                logger.addHandler(fh)
            logger.propagate = True
        except Exception:
            pass
        
        self.results = {}
        self.trades_df = None
        self.equity_curve = None
        self.daily_returns = None
        self.performance_metrics = {}
        
        logger.info(">>> Enhanced RSI Backtest V4 Initialized")

    def fetch_real_data_from_mt5(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "H1",
        days_back: int = 120
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
            candles_needed = min(candles_needed, 5000)
            
            logger.info(f"📥 درخواست {candles_needed} کندل برای {symbol}")
            data = mt5_fetcher.fetch_market_data(symbol, timeframe, candles_needed)
            
            if data.empty:
                raise ValueError(f"هیچ داده‌ای برای {symbol} دریافت نشد")
            
            # اطمینان از فرمت داده
            if 'open_time' in data.columns:
                data['open_time'] = pd.to_datetime(data['open_time'])
                data.set_index('open_time', inplace=True)
            
            # محاسبه RSI و سایر اندیکاتورها
            data = self._calculate_technical_indicators(data)
            
            logger.info(f">>> دریافت {len(data)} کندل از {symbol}")
            logger.info(f">>> بازه: {data.index[0]} تا {data.index[-1]}")
            logger.info(f">>> قیمت: {data['close'].min():.4f} - {data['close'].max():.4f}")
            logger.info(f">>> تغییر قیمت: {((data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * 100):.2f}%")
            
            return data
            
        except Exception as e:
            logger.error(f">>> خطا در دریافت داده: {e}")
            logger.error(traceback.format_exc())
            raise

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """محاسبه اندیکاتورهای تکنیکال"""
        try:
            # RSI
            data = self._calculate_rsi(data, 14)
            
            # میانگین‌های متحرک
            data['EMA_9'] = data['close'].ewm(span=9).mean()
            data['EMA_21'] = data['close'].ewm(span=21).mean()
            data['EMA_50'] = data['close'].ewm(span=50).mean()
            
            # ATR
            data['ATR'] = self._calculate_atr_series(data)
            
            # باندهای بولینگر
            data = self._calculate_bollinger_bands(data)
            
            # MACD
            data = self._calculate_macd(data)
            
            return data
            
        except Exception as e:
            logger.error(f"خطا در محاسبه اندیکاتورها: {e}")
            return data

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

    def _calculate_atr_series(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """محاسبه سری ATR"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr.fillna(method='bfill')
        except Exception as e:
            logger.error(f"خطا در محاسبه ATR: {e}")
            return pd.Series(index=data.index, data=data['close'] * 0.01)

    def _calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std: int = 2) -> pd.DataFrame:
        """محاسبه باندهای بولینگر"""
        try:
            data['BB_Middle'] = data['close'].rolling(period).mean()
            data['BB_Std'] = data['close'].rolling(period).std()
            data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * std)
            data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * std)
            
            return data
        except Exception as e:
            logger.error(f"خطا در محاسبه بولینگر: {e}")
            return data

    def _calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """محاسبه MACD"""
        try:
            exp1 = data['close'].ewm(span=fast).mean()
            exp2 = data['close'].ewm(span=slow).mean()
            data['MACD'] = exp1 - exp2
            data['MACD_Signal'] = data['MACD'].ewm(span=signal).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            return data
        except Exception as e:
            logger.error(f"خطا در محاسبه MACD: {e}")
            return data

    def _attach_mtf_to_base(self, base_df: pd.DataFrame, symbol: str, days_back: int, timeframes: List[str]) -> pd.DataFrame:
        """ضمیمه‌کردن ویژگی‌های HTF/D1 به دیتافریم تایم‌فریم پایه بدون lookahead.
        ایجاد ستون‌های زیر در base_df (اگر داده موجود باشد):
          - RSI_{TF}, EMA_21_{TF}, EMA_50_{TF}, TrendDir_{TF} (1=UP, -1=DOWN, 0=FLAT)
        با ffill روی ایندکس زمانی base_df برای هم‌ترازی ایمن.
        """
        try:
            if base_df is None or base_df.empty or not timeframes:
                return base_df

            base_idx = base_df.index
            for tf in timeframes:
                try:
                    htf_df = self.fetch_real_data_from_mt5(symbol, tf, days_back)
                    cols_map = {}

                    if 'RSI' in htf_df.columns:
                        cols_map[f'RSI_{tf}'] = htf_df['RSI']
                    if 'EMA_21' in htf_df.columns:
                        cols_map[f'EMA_21_{tf}'] = htf_df['EMA_21']
                    if 'EMA_50' in htf_df.columns:
                        cols_map[f'EMA_50_{tf}'] = htf_df['EMA_50']

                    if cols_map:
                        tmp = pd.DataFrame(cols_map)
                        if f'EMA_21_{tf}' in tmp.columns and f'EMA_50_{tf}' in tmp.columns:
                            trend = np.sign(tmp[f'EMA_21_{tf}'] - tmp[f'EMA_50_{tf}']).replace({np.nan: 0})
                            tmp[f'TrendDir_{tf}'] = trend

                        # Align HTF snapshots to base timeframe without lookahead
                        tmp = tmp.reindex(base_idx, method='ffill')

                        for c in tmp.columns:
                            base_df[c] = tmp[c]
                except Exception as ie:
                    logger.warning(f"MTF attach failed for {tf}: {ie}")

            return base_df
        except Exception as e:
            logger.error(f"Error attaching MTF features: {e}")
            return base_df

    def run_backtest(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "H1",
        days_back: int = 120,
        strategy_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """اجرای بکتست پیشرفته"""
        logger.info(">>> شروع بکتست نسخه ۴")
        
        # استفاده از پارامترهای بهینه‌شده اگر پارامتری ارائه نشده
        if strategy_params is None:
            strategy_params = OPTIMIZED_PARAMS_V4
        
        try:
            # دریافت داده
            data = self.fetch_real_data_from_mt5(symbol, timeframe, days_back)
            
            # ایجاد استراتژی
            strategy = EnhancedRsiStrategyV4(**strategy_params)

            # ضمیمه‌کردن ویژگی‌های MTF در صورت فعال‌بودن
            try:
                if getattr(strategy, 'enable_mtf', False):
                    mtf_tfs = getattr(strategy, 'mtf_timeframes', ['H4', 'D1'])
                    data = self._attach_mtf_to_base(data, symbol, days_back, mtf_tfs)
                    logger.info(f"MTF features attached for timeframes: {mtf_tfs}")
            except Exception as e:
                logger.warning(f"Failed to attach MTF features: {e}")
            
            # اجرای بکتست
            trades = []
            portfolio_values = []
            signals = []
            
            logger.info(f">>> شروع شبیه‌سازی بر روی {len(data)} کندل")
            
            for i in range(len(data)):
                if i < 50:  # صبر برای داده کافی
                    continue
                    
                current_data = data.iloc[:i+1].copy()
                current_time = current_data.index[-1]
                
                # تولید سیگنال
                signal = strategy.generate_signal(current_data, i)
                signal['timestamp'] = current_time
                signals.append(signal)
                
                # پردازش سیگنال
                if signal['action'] in ['BUY', 'SELL']:
                    trade_result = self._execute_trade(strategy, signal, current_data, i)
                    if trade_result:
                        trades.append(trade_result)
                        if self.detailed_logging:
                            logger.info(f"🎯 {signal['action']} at {trade_result['price']:.4f}")
                        
                elif signal['action'] == 'EXIT':
                    exit_result = self._process_exit(strategy, signal, current_data, i)
                    if exit_result:
                        trades.append(exit_result)
                        if self.detailed_logging:
                            pnl = exit_result.get('pnl_percentage', 0)
                            logger.info(f"🔚 EXIT at {exit_result['price']:.4f}, PnL: {pnl:.2f}%")
                
                elif signal['action'] == 'PARTIAL_EXIT':
                    partial_result = self._process_partial_exit(strategy, signal, current_data, i)
                    if partial_result:
                        trades.append(partial_result)
                        if self.detailed_logging:
                            pnl = partial_result.get('pnl_percentage', 0)
                            logger.info(f"📦 PARTIAL at {partial_result['price']:.4f}, PnL: {pnl:.2f}%")
                
                # ثبت ارزش پورتفو
                portfolio_value = self._calculate_portfolio_value(strategy, current_data)
                portfolio_values.append({
                    'timestamp': current_time,
                    'portfolio_value': portfolio_value,
                    'price': current_data['close'].iloc[-1],
                    'market_condition': signal.get('market_condition', 'UNKNOWN')
                })
                
                # لاگ پیشرفت
                if (i + 1) % 500 == 0:
                    progress = (i + 1) / len(data) * 100
                    logger.info(f"📊 پیشرفت: {progress:.1f}% ({i + 1}/{len(data)})")
            
            # کامپایل نتایج
            self._compile_results(strategy, trades, portfolio_values, signals, data, symbol)
            self._calculate_performance_metrics()
            
            if self.enable_plotting:
                self._generate_advanced_plots(data, symbol, timeframe)
            
            if self.save_trade_logs:
                self._save_detailed_trade_logs(strategy)
            
            logger.info(">>> بکتست با موفقیت انجام شد")
            return self.results
            
        except Exception as e:
            logger.error(f">>> خطا در اجرای بکتست: {e}")
            logger.error(traceback.format_exc())
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
                'atr': data['ATR'].iloc[-1] if 'ATR' in data.columns else 0,
                'market_condition': signal.get('market_condition', 'UNKNOWN'),
                'reason': signal.get('reason', ''),
                'portfolio_value': strategy._portfolio_value - commission
            }
            
            return trade_record
            
        except Exception as e:
            logger.error(f"خطا در اجرای معامله: {e}")
            return None

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
                'market_condition': signal.get('market_condition', 'UNKNOWN'),
                'portfolio_value': strategy._portfolio_value - commission
            }
            
            return trade_record
            
        except Exception as e:
            logger.error(f"خطا در پردازش خروج: {e}")
            return None

    def _process_partial_exit(self, strategy, signal, data, index):
        """پردازش خروج جزئی"""
        try:
            current_price = data['close'].iloc[-1]
            execution_price = current_price  # بدون slippage برای خروج جزئی
            
            trade_record = {
                'timestamp': data.index[-1],
                'action': 'PARTIAL_EXIT',
                'symbol': 'EURUSD',
                'price': execution_price,
                'position_size': signal.get('quantity', 0),
                'pnl_percentage': signal.get('pnl_percentage', 0),
                'pnl_amount': signal.get('pnl_amount', 0),
                'exit_reason': signal.get('reason', ''),
                'market_condition': signal.get('market_condition', 'UNKNOWN'),
                'portfolio_value': strategy._portfolio_value
            }
            
            return trade_record
            
        except Exception as e:
            logger.error(f"خطا در پردازش خروج جزئی: {e}")
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

    def _compile_results(self, strategy, trades, portfolio_values, signals, data, symbol):
        """کامپایل نتایج پیشرفته"""
        self.trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        self.equity_curve = pd.DataFrame(portfolio_values)
        self.signals_df = pd.DataFrame(signals)
        
        if not self.equity_curve.empty:
            self.equity_curve['timestamp'] = pd.to_datetime(self.equity_curve['timestamp'])
            self.equity_curve.set_index('timestamp', inplace=True)
        
        # محاسبه معیارهای استراتژی
        strategy_metrics = strategy.get_performance_metrics()
        trade_history = strategy.get_trade_history()
        signal_log = strategy.get_signal_log()
        
        self.results = {
            'trades': self.trades_df,
            'equity_curve': self.equity_curve,
            'signals': self.signals_df,
            'strategy_metrics': strategy_metrics,
            'trade_history': trade_history,
            'signal_log': signal_log,
            'symbol': symbol,
            'data_info': {
                'start_date': data.index[0],
                'end_date': data.index[-1],
                'total_candles': len(data),
                'price_range': f"{data['close'].min():.4f} - {data['close'].max():.4f}",
                'price_change_pct': ((data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * 100),
                'volatility': data['close'].pct_change().std()
            }
        }

    def _calculate_performance_metrics(self):
        """محاسبه معیارهای عملکرد پیشرفته"""
        if self.equity_curve is None or self.equity_curve.empty:
            return
        
        portfolio_values = self.equity_curve['portfolio_value']
        
        # محاسبات پایه
        total_return = (portfolio_values.iloc[-1] - self.initial_capital) / self.initial_capital * 100
        
        # محاسبه drawdown
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max * 100
        max_drawdown = drawdowns.min()
        
        # محاسبه بازده روزانه
        daily_returns = portfolio_values.pct_change().dropna()
        annual_return = total_return / (len(portfolio_values) / 365) if len(portfolio_values) > 0 else 0
        volatility = daily_returns.std() * np.sqrt(365)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # محاسبه معاملات
        total_trades = len(self.trades_df) if self.trades_df is not None else 0
        winning_trades = len([t for t in self.results.get('trade_history', []) if t.pnl_amount and t.pnl_amount > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # نسبت سود به زیان
        profitable_trades = [t for t in self.results.get('trade_history', []) if t.pnl_amount and t.pnl_amount > 0]
        losing_trades = [t for t in self.results.get('trade_history', []) if t.pnl_amount and t.pnl_amount < 0]
        
        avg_win = np.mean([t.pnl_amount for t in profitable_trades]) if profitable_trades else 0
        avg_loss = np.mean([t.pnl_amount for t in losing_trades]) if losing_trades else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        self.performance_metrics = {
            'total_return_pct': round(total_return, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'volatility_pct': round(volatility * 100, 2),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': round(win_rate, 2),
            'profit_loss_ratio': round(profit_loss_ratio, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'final_portfolio_value': round(portfolio_values.iloc[-1], 2),
            'initial_capital': self.initial_capital
        }
        
        self.results['performance_metrics'] = self.performance_metrics

    def _generate_advanced_plots(self, data, symbol, timeframe):
        """تولید نمودارهای پیشرفته"""
        try:
            fig = plt.figure(figsize=(20, 16))
            gs = plt.GridSpec(4, 2, figure=fig)
            fig.suptitle(f'RSI Strategy V4 - {symbol} ({timeframe}) - Comprehensive Analysis', fontsize=16, y=0.98)
            
            # نمودار ۱: قیمت و سیگنال‌ها
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(data.index, data['close'], label='Price', color='blue', linewidth=1, alpha=0.7)
            
            # علامت‌گذاری معاملات
            if self.trades_df is not None and not self.trades_df.empty:
                buy_trades = self.trades_df[self.trades_df['action'] == 'BUY']
                sell_trades = self.trades_df[self.trades_df['action'] == 'SELL']
                exit_trades = self.trades_df[self.trades_df['action'] == 'EXIT']
                
                ax1.scatter(buy_trades['timestamp'], buy_trades['price'], 
                           color='green', marker='^', s=50, label='Buy', alpha=0.7)
                ax1.scatter(sell_trades['timestamp'], sell_trades['price'], 
                           color='red', marker='v', s=50, label='Sell', alpha=0.7)
                ax1.scatter(exit_trades['timestamp'], exit_trades['price'], 
                           color='orange', marker='o', s=30, label='Exit', alpha=0.7)
            
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_title('Price Chart with Trade Signals')
            
            # نمودار ۲: RSI
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.plot(data.index, data['RSI'], color='purple', linewidth=1)
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold')
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought')
            ax2.set_ylabel('RSI')
            ax2.set_ylim(0, 100)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_title('RSI Indicator')
            
            # نمودار ۳: منحنی سرمایه
            ax3 = fig.add_subplot(gs[1, 1])
            if self.equity_curve is not None and not self.equity_curve.empty:
                ax3.plot(self.equity_curve.index, self.equity_curve['portfolio_value'], 
                        color='green', linewidth=2, label='Portfolio Value')
                ax3.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
            ax3.set_ylabel('Portfolio Value ($)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_title('Equity Curve')
            
            # نمودار ۴: Drawdown
            ax4 = fig.add_subplot(gs[2, 0])
            if self.equity_curve is not None and not self.equity_curve.empty:
                rolling_max = self.equity_curve['portfolio_value'].expanding().max()
                drawdown = (self.equity_curve['portfolio_value'] - rolling_max) / rolling_max * 100
                ax4.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
                ax4.set_ylabel('Drawdown %')
                ax4.set_title('Portfolio Drawdown')
                ax4.grid(True, alpha=0.3)
            
            # نمودار ۵: توزیع سود/زیان معاملات
            ax5 = fig.add_subplot(gs[2, 1])
            if self.trades_df is not None and not self.trades_df.empty:
                trades_with_pnl = [t for t in self.results.get('trade_history', []) if t.pnl_percentage is not None]
                if trades_with_pnl:
                    pnl_values = [t.pnl_percentage for t in trades_with_pnl]
                    colors = ['green' if x > 0 else 'red' for x in pnl_values]
                    ax5.hist(pnl_values, bins=20, alpha=0.7, color='blue', edgecolor='black')
                    ax5.axvline(np.mean(pnl_values), color='red', linestyle='--', 
                               label=f'Mean: {np.mean(pnl_values):.2f}%')
                    ax5.legend()
            ax5.set_xlabel('PnL %')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Trade PnL Distribution')
            ax5.grid(True, alpha=0.3)
            
            # نمودار ۶: تحلیل زمانی
            ax6 = fig.add_subplot(gs[3, 0])
            if self.trades_df is not None and not self.trades_df.empty:
                # گروه‌بندی معاملات بر اساس ساعت روز
                trades_df = self.trades_df.copy()
                trades_df['hour'] = trades_df['timestamp'].dt.hour
                hourly_trades = trades_df.groupby('hour').size()
                ax6.bar(hourly_trades.index, hourly_trades.values, alpha=0.7)
                ax6.set_xlabel('Hour of Day')
                ax6.set_ylabel('Number of Trades')
                ax6.set_title('Trades by Hour of Day')
                ax6.grid(True, alpha=0.3)
            
            # نمودار ۷: کارایی بر اساس شرایط بازار
            ax7 = fig.add_subplot(gs[3, 1])
            if self.equity_curve is not None and not self.equity_curve.empty:
                market_conditions = self.equity_curve['market_condition'].value_counts()
                ax7.pie(market_conditions.values, labels=market_conditions.index, autopct='%1.1f%%')
                ax7.set_title('Market Conditions Distribution')
            
            plt.tight_layout()
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.plots_dir, f"backtest_v4_{symbol}_{timeframe}_{ts}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f">>> نمودارهای پیشرفته در {filename} ذخیره شد")
            
        except Exception as e:
            logger.error(f"خطا در تولید نمودارهای پیشرفته: {e}")

    def _save_detailed_trade_logs(self, strategy):
        """ذخیره لاگ‌های دقیق معاملات"""
        try:
            trade_logs = []
            for trade in strategy.get_trade_history():
                trade_log = {
                    'entry_time': trade.entries[0].time if trade.entries else None,
                    'entry_price': trade.entry_price,
                    'exit_time': trade.exit_time,
                    'exit_price': trade.exit_price,
                    'position_type': trade.position_type.value,
                    'quantity': trade.quantity,
                    'pnl_percentage': trade.pnl_percentage,
                    'pnl_amount': trade.pnl_amount,
                    'exit_reason': trade.exit_reason.value if trade.exit_reason else None,
                    'stop_loss': trade.stop_loss,
                    'take_profit': trade.take_profit,
                    'partial_exits': len(trade.partial_exits)
                }
                trade_logs.append(trade_log)
            
            # ذخیره به صورت JSON
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = os.path.join(self.trades_dir, f"detailed_trades_{ts}.json")
            with open(log_filename, 'w', encoding='utf-8') as f:
                json.dump(trade_logs, f, indent=2, default=str)
            
            logger.info(f">>> لاگ‌های دقیق معاملات در {log_filename} ذخیره شد")
            
        except Exception as e:
            logger.error(f"خطا در ذخیره لاگ معاملات: {e}")

    def generate_comprehensive_report(self) -> str:
        """تولید گزارش جامع"""
        if not self.results:
            return "No results available"
        
        metrics = self.results.get('performance_metrics', {})
        strategy_metrics = self.results.get('strategy_metrics', {})
        data_info = self.results.get('data_info', {})
        
        report = [
            "=" * 70,
            "ENHANCED RSI STRATEGY V4 - COMPREHENSIVE BACKTEST REPORT",
            "=" * 70,
            f"Symbol: {self.results.get('symbol', 'N/A')}",
            f"Period: {data_info.get('start_date')} to {data_info.get('end_date')}",
            f"Total Candles: {data_info.get('total_candles', 0)}",
            "",
            "PERFORMANCE SUMMARY:",
            f"Initial Capital: ${metrics.get('initial_capital', 0):,.2f}",
            f"Final Portfolio Value: ${metrics.get('final_portfolio_value', 0):,.2f}",
            f"Total Return: {metrics.get('total_return_pct', 0):.2f}%",
            f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%",
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
            f"Volatility: {metrics.get('volatility_pct', 0):.2f}%",
            "",
            "TRADING METRICS:",
            f"Total Trades: {metrics.get('total_trades', 0)}",
            f"Winning Trades: {metrics.get('winning_trades', 0)}",
            f"Win Rate: {metrics.get('win_rate', 0):.2f}%",
            f"Profit/Loss Ratio: {metrics.get('profit_loss_ratio', 0):.2f}",
            f"Average Win: ${metrics.get('avg_win', 0):.2f}",
            f"Average Loss: ${metrics.get('avg_loss', 0):.2f}",
            "",
            "STRATEGY INSIGHTS:",
            f"Market Condition: {strategy_metrics.get('market_condition', 'UNKNOWN')}",
            f"Consecutive Losses: {strategy_metrics.get('consecutive_losses', 0)}",
            f"Current Position: {strategy_metrics.get('current_position', 'OUT')}",
            "",
            "MARKET ANALYSIS:",
            f"Price Change: {data_info.get('price_change_pct', 0):.2f}%",
            f"Price Range: {data_info.get('price_range', 'N/A')}",
            f"Market Volatility: {data_info.get('volatility', 0):.4f}",
            "",
            "STRATEGY ASSESSMENT:"
        ]
        
        # ارزیابی استراتژی
        total_return = metrics.get('total_return_pct', 0)
        max_dd = metrics.get('max_drawdown_pct', 0)
        win_rate = metrics.get('win_rate', 0)
        
        if total_return > 15 and max_dd > -5 and win_rate > 55:
            assessment = "EXCELLENT - استراتژی عملکرد بسیار خوبی دارد"
        elif total_return > 8 and max_dd > -8 and win_rate > 50:
            assessment = "GOOD - استراتژی سودده است"
        elif total_return > 0 and max_dd > -10:
            assessment = "ACCEPTABLE - استراتژی marginally سودده است"
        elif total_return > -5:
            assessment = "NEEDS IMPROVEMENT - استراتژی نیاز به تنظیم دارد"
        else:
            assessment = "POOR - استراتژی نیاز به بازنگری اساسی دارد"
        
        report.append(f"Assessment: {assessment}")
        report.append("=" * 70)
        
        return "\n".join(report)

    def optimize_parameters(self, data: pd.DataFrame, param_grid: Dict[str, List]) -> Dict[str, Any]:
        """بهینه‌سازی پارامترهای استراتژی"""
        logger.info(">>> شروع بهینه‌سازی پارامترها")
        
        best_params = {}
        best_performance = -float('inf')
        
        try:
            # ایجاد ترکیب‌های پارامتری
            from itertools import product
            param_combinations = list(product(*param_grid.values()))
            
            logger.info(f">>> تست {len(param_combinations)} ترکیب پارامتری")
            
            for i, combination in enumerate(param_combinations):
                params = dict(zip(param_grid.keys(), combination))
                
                try:
                    # اجرای بکتست با پارامترهای فعلی
                    strategy = EnhancedRsiStrategyV4(**params)
                    
                    # شبیه‌سازی سریع
                    portfolio_values = []
                    for j in range(50, len(data)):
                        current_data = data.iloc[:j+1].copy()
                        signal = strategy.generate_signal(current_data, j)
                        
                        # محاسبه ارزش پورتفو
                        portfolio_value = self._calculate_portfolio_value(strategy, current_data)
                        portfolio_values.append(portfolio_value)
                    
                    # ارزیابی عملکرد
                    final_value = portfolio_values[-1] if portfolio_values else self.initial_capital
                    performance = (final_value - self.initial_capital) / self.initial_capital
                    
                    if performance > best_performance:
                        best_performance = performance
                        best_params = params.copy()
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"🔧 بهینه‌سازی: {i + 1}/{len(param_combinations)}")
                        
                except Exception as e:
                    logger.warning(f"خطا در ترکیب {params}: {e}")
                    continue
            
            logger.info(f">>> بهینه‌سازی تکمیل شد. بهترین عملکرد: {best_performance:.2%}")
            return best_params
            
        except Exception as e:
            logger.error(f"خطا در بهینه‌سازی: {e}")
            return OPTIMIZED_PARAMS_V4