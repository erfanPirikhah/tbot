# backtest/adaptive_rsi_backtest.py

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import warnings

# تنظیم مسیرهای پروژه
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# غیرفعال کردن warnings
warnings.filterwarnings('ignore')

try:
    from strategies.adaptive_elite_rsi_strategy import ProfessionalAdvancedRsiStrategy, PositionType
    from data.mt5_data import mt5_fetcher, MT5_AVAILABLE
    from indicators.rsi import calculate_rsi
    print("✅ تمام ماژول‌ها با موفقیت import شدند")
except ImportError as e:
    print(f"❌ خطا در import ماژول‌ها: {e}")
    print("🔄 تلاش برای import مستقیم...")
    
    # import مستقیم در صورت نیاز
    import importlib.util
    def import_from_path(module_name, path):
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    # مسیرهای مستقیم
    strategy_path = os.path.join(parent_dir, 'strategies', 'adaptive_elite_rsi_strategy.py')
    mt5_data_path = os.path.join(parent_dir, 'data', 'mt5_data.py')
    rsi_path = os.path.join(parent_dir, 'indicators', 'rsi.py')
    
    try:
        strategy_module = import_from_path('adaptive_elite_rsi_strategy', strategy_path)
        ProfessionalAdvancedRsiStrategy = strategy_module.ProfessionalAdvancedRsiStrategy
        PositionType = strategy_module.PositionType
        print("✅ استراتژی import شد")
    except Exception as e:
        print(f"❌ خطا در import استراتژی: {e}")
        sys.exit(1)

# تنظیمات لاگ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('adaptive_rsi_backtest.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdaptiveRSIBacktester:
    """بکتستر کامل برای استراتژی Adaptive Elite RSI"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.results = {}
        self.detailed_trades = []
        self.all_signals = []
        
        # نمادهای مختلف برای تست
        self.symbols = [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
            "XAUUSD", "XAGUSD",  # طلا و نقره
            "USOIL",  # نفت
            "BTCUSD", "ETHUSD"   # ارزهای دیجیتال
        ]
        
        # تایم‌فریم‌های مختلف
        self.timeframes = ["H1", "H4", "D1"]
        
        # پارامترهای استراتژی
        self.strategy_params = {
            "rsi_period": 14,
            "rsi_base_oversold": 35,
            "rsi_base_overbought": 65,
            "risk_per_trade": 0.02,
            "base_stop_atr_multiplier": 2.0,
            "base_take_profit_ratio": 2.0,
            "max_trade_duration": 48,
            "enable_short_trades": True,
            "use_dynamic_trailing": True,
            "max_trades_per_100": 20,
            "min_candles_between": 5
        }
        
        logger.info(f"✅ بکتستر راه‌اندازی شد - سرمایه اولیه: ${initial_capital:,.2f}")
    
    def check_mt5_connection(self) -> bool:
        """بررسی اتصال به MT5"""
        if not MT5_AVAILABLE:
            logger.error("❌ MetaTrader5 در دسترس نیست")
            return False
        
        try:
            if mt5_fetcher and mt5_fetcher.ensure_connected():
                logger.info("✅ اتصال به MT5 برقرار است")
                return True
            else:
                logger.error("❌ اتصال به MT5 برقرار نیست")
                return False
        except Exception as e:
            logger.error(f"❌ خطا در بررسی اتصال MT5: {e}")
            return False
    
    def fetch_historical_data(self, symbol: str, timeframe: str, bars: int = 1000) -> pd.DataFrame:
        """دریافت داده‌های تاریخی از MT5"""
        logger.info(f"📥 دریافت داده برای {symbol} ({timeframe}) - {bars} کندل")
        
        if not self.check_mt5_connection():
            return pd.DataFrame()
        
        try:
            data = mt5_fetcher.fetch_market_data(symbol, timeframe, bars)
            
            if data.empty:
                logger.warning(f"⚠️ داده‌ای برای {symbol} دریافت نشد")
                return pd.DataFrame()
            
            # محاسبه RSI
            data = calculate_rsi(data, period=14)
            
            # اطمینان از وجود ستون‌های ضروری
            required_columns = ['open', 'high', 'low', 'close', 'RSI']
            for col in required_columns:
                if col not in data.columns:
                    logger.error(f"❌ ستون {col} در داده‌ها وجود ندارد")
                    return pd.DataFrame()
            
            logger.info(f"✅ {len(data)} کندل برای {symbol} دریافت شد - آخرین قیمت: {data['close'].iloc[-1]:.5f}")
            return data
            
        except Exception as e:
            logger.error(f"❌ خطا در دریافت داده برای {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """محاسبه ATR برای داده‌ها"""
        try:
            high = data['high']
            low = data['low']
            close_prev = data['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close_prev)
            tr3 = abs(low - close_prev)
            
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr = tr.rolling(period).mean()
            
            return atr
        except Exception as e:
            logger.error(f"خطا در محاسبه ATR: {e}")
            return pd.Series(index=data.index, data=0.0)
    
    def run_single_backtest(self, symbol: str, timeframe: str, days_back: int = 180) -> Dict:
        """اجرای بکتست برای یک نماد و تایم‌فریم"""
        logger.info(f"🚀 شروع بکتست برای {symbol} ({timeframe})")
        
        # محاسبه تعداد کندل مورد نیاز
        bars_per_day = {
            "H1": 24,
            "H4": 6,
            "D1": 1
        }
        bars = bars_per_day.get(timeframe, 24) * days_back
        
        # دریافت داده‌ها
        data = self.fetch_historical_data(symbol, timeframe, bars)
        if data.empty:
            return {}
        
        # محاسبه ATR
        data['ATR'] = self.calculate_atr(data)
        
        # ایجاد استراتژی
        strategy = ProfessionalAdvancedRsiStrategy(**self.strategy_params)
        strategy._portfolio_value = self.initial_capital
        
        trades = []
        signals = []
        
        # اجرای استراتژی روی تمام داده‌ها
        for i in range(len(data)):
            if i < 50:  # صبر برای تشکیل اندیکاتورها
                continue
                
            try:
                current_data = data.iloc[:i+1].copy()
                
                # بررسی وجود RSI و ATR
                if 'RSI' not in current_data.columns or current_data['RSI'].isna().iloc[-1]:
                    continue
                    
                if 'ATR' not in current_data.columns or current_data['ATR'].isna().iloc[-1]:
                    continue
                
                signal = strategy.generate_signal(current_data, i)
                
                # ثبت سیگنال
                signal_record = {
                    'timestamp': current_data.index[-1],
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'action': signal['action'],
                    'price': signal.get('price', 0),
                    'rsi': current_data['RSI'].iloc[-1],
                    'atr': current_data['ATR'].iloc[-1],
                    'reason': signal.get('reason', ''),
                    'position': signal.get('position', 'OUT')
                }
                signals.append(signal_record)
                self.all_signals.append(signal_record)
                
                # ثبت معاملات بسته شده
                if (strategy._current_trade is None and 
                    len(strategy.trade_history) > len(trades)):
                    
                    latest_trade = strategy.trade_history[-1]
                    trade_info = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'entry_time': latest_trade.entry_time,
                        'exit_time': latest_trade.exit_time,
                        'position_type': latest_trade.position_type.value,
                        'entry_price': latest_trade.entry_price,
                        'exit_price': latest_trade.exit_price,
                        'quantity': latest_trade.quantity,
                        'stop_loss': latest_trade.stop_loss,
                        'take_profit': latest_trade.take_profit,
                        'pnl_percentage': latest_trade.pnl_percentage,
                        'pnl_amount': latest_trade.pnl_amount,
                        'exit_reason': latest_trade.exit_reason.value if latest_trade.exit_reason else None,
                        'duration_hours': (latest_trade.exit_time - latest_trade.entry_time).total_seconds() / 3600
                    }
                    trades.append(trade_info)
                    self.detailed_trades.append(trade_info)
                    
                    logger.info(f"🔁 معامله بسته شد: {trade_info['position_type']} - سود: {trade_info['pnl_percentage']:.2f}%")
                    
            except Exception as e:
                logger.error(f"❌ خطا در مرحله {i} برای {symbol}: {e}")
                continue
        
        # محاسبه نتایج
        metrics = strategy.get_performance_metrics()
        
        # محاسبه معیارهای پیشرفته
        advanced_metrics = self.calculate_advanced_metrics(trades)
        metrics.update(advanced_metrics)
        
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'total_trades': len(trades),
            'trades': trades,
            'signals': signals,
            'metrics': metrics,
            'final_equity': strategy._portfolio_value,
            'total_return': ((strategy._portfolio_value - self.initial_capital) / self.initial_capital) * 100,
            'data_points': len(data)
        }
        
        logger.info(f"✅ بکتست {symbol} ({timeframe}) تکمیل: {len(trades)} معامله, بازدهی: {result['total_return']:.2f}%")
        
        return result
    
    def calculate_advanced_metrics(self, trades: List[Dict]) -> Dict:
        """محاسبه معیارهای پیشرفته عملکرد"""
        if not trades:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'total_profit': 0,
                'total_loss': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'avg_trade': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'max_losing_streak': 0,
                'sharpe_ratio': 0,
                'avg_trade_duration': 0
            }
        
        try:
            trades_df = pd.DataFrame(trades)
            
            # معیارهای پایه
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl_amount'] > 0])
            losing_trades = len(trades_df[trades_df['pnl_amount'] < 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # سود/ضرر
            total_profit = trades_df[trades_df['pnl_amount'] > 0]['pnl_amount'].sum()
            total_loss = abs(trades_df[trades_df['pnl_amount'] < 0]['pnl_amount'].sum())
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # میانگین‌ها
            avg_win = trades_df[trades_df['pnl_amount'] > 0]['pnl_amount'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl_amount'] < 0]['pnl_amount'].mean() if losing_trades > 0 else 0
            avg_trade = trades_df['pnl_amount'].mean()
            
            # بزرگترین‌ها
            largest_win = trades_df['pnl_amount'].max()
            largest_loss = trades_df['pnl_amount'].min()
            
            # استرین (توالی ضرر)
            current_streak = 0
            max_losing_streak = 0
            for pnl in trades_df['pnl_amount']:
                if pnl < 0:
                    current_streak += 1
                    max_losing_streak = max(max_losing_streak, current_streak)
                else:
                    current_streak = 0
            
            # شارپ ریتیو (ساده)
            returns_std = trades_df['pnl_amount'].std()
            sharpe_ratio = avg_trade / returns_std if returns_std > 0 else 0
            
            # میانگین مدت معامله
            avg_duration = trades_df['duration_hours'].mean()
            
            return {
                'win_rate': round(win_rate, 2),
                'profit_factor': round(profit_factor, 2),
                'total_profit': round(total_profit, 2),
                'total_loss': round(total_loss, 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'avg_trade': round(avg_trade, 2),
                'largest_win': round(largest_win, 2),
                'largest_loss': round(largest_loss, 2),
                'max_losing_streak': max_losing_streak,
                'sharpe_ratio': round(sharpe_ratio, 2),
                'avg_trade_duration': round(avg_duration, 2)
            }
            
        except Exception as e:
            logger.error(f"خطا در محاسبه معیارهای پیشرفته: {e}")
            return {}
    
    def run_comprehensive_backtest(self):
        """اجرای بکتست جامع روی تمام نمادها و تایم‌فریم‌ها"""
        logger.info("🎯 شروع بکتست جامع استراتژی Adaptive Elite RSI")
        
        # ایجاد پوشه نتایج
        os.makedirs('backtest_results', exist_ok=True)
        
        all_results = []
        
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                try:
                    logger.info(f"🔍 تست {symbol} در تایم‌فریم {timeframe}")
                    
                    result = self.run_single_backtest(
                        symbol=symbol,
                        timeframe=timeframe,
                        days_back=90  # 3 ماه داده
                    )
                    
                    if result and result['total_trades'] > 0:
                        all_results.append(result)
                        
                        # ذخیره نتایج جزئی
                        self.save_single_result(result)
                        
                    else:
                        logger.warning(f"⚠️ هیچ معامله‌ای برای {symbol}({timeframe}) انجام نشد")
                        
                except Exception as e:
                    logger.error(f"❌ خطا در بکتست {symbol}({timeframe}): {e}")
                    continue
        
        # تولید گزارش نهایی
        if all_results:
            self.generate_final_report(all_results)
            logger.info("✅ بکتست جامع تکمیل شد")
        else:
            logger.warning("⚠️ هیچ نتیجه‌ای از بکتست بدست نیامد")
        
        return all_results
    
    def save_single_result(self, result: Dict):
        """ذخیره نتایج یک نماد"""
        identifier = f"{result['symbol']}_{result['timeframe']}"
        
        try:
            # ذخیره به صورت JSON
            with open(f'backtest_results/{identifier}_result.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            
            # ذخیره معاملات به صورت CSV
            if result['trades']:
                trades_df = pd.DataFrame(result['trades'])
                trades_df.to_csv(f'backtest_results/{identifier}_trades.csv', 
                               index=False, encoding='utf-8-sig')
            
            logger.info(f"💾 نتایج {identifier} ذخیره شد")
            
        except Exception as e:
            logger.error(f"❌ خطا در ذخیره نتایج {identifier}: {e}")
    
    def generate_final_report(self, all_results: List[Dict]):
        """تولید گزارش نهایی و نمودارها"""
        logger.info("📊 تولید گزارش نهایی")
        
        try:
            # جمع‌آوری خلاصه نتایج
            summary_data = []
            for result in all_results:
                summary_data.append({
                    'symbol': result['symbol'],
                    'timeframe': result['timeframe'],
                    'total_trades': result['total_trades'],
                    'win_rate': result['metrics'].get('win_rate', 0),
                    'total_return': result['total_return'],
                    'profit_factor': result['metrics'].get('profit_factor', 0),
                    'final_equity': result['final_equity'],
                    'avg_trade': result['metrics'].get('avg_trade', 0),
                    'max_losing_streak': result['metrics'].get('max_losing_streak', 0)
                })
            
            summary_df = pd.DataFrame(summary_data)
            
            # ذخیره خلاصه
            summary_df.to_csv('backtest_results/summary_results.csv', 
                            index=False, encoding='utf-8-sig')
            
            # تولید نمودارها
            self.create_performance_charts(summary_df, all_results)
            
            # گزارش متنی
            self.generate_text_report(summary_df, all_results)
            
        except Exception as e:
            logger.error(f"❌ خطا در تولید گزارش نهایی: {e}")
    
    def create_performance_charts(self, summary_df: pd.DataFrame, all_results: List[Dict]):
        """ایجاد نمودارهای عملکرد"""
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('گزارش عملکرد استراتژی Adaptive Elite RSI', fontsize=16, fontweight='bold')
            
            # نمودار ۱: بازدهی بر اساس نماد و تایم‌فریم
            if not summary_df.empty:
                pivot_returns = summary_df.pivot(index='symbol', columns='timeframe', values='total_return')
                pivot_returns.plot(kind='bar', ax=axes[0,0], 
                                 title='بازدهی بر اساس نماد و تایم‌فریم (%)',
                                 color=['#2E8B57', '#4682B4', '#D2691E'])
                axes[0,0].set_ylabel('بازدهی (%)')
                axes[0,0].tick_params(axis='x', rotation=45)
                axes[0,0].grid(True, alpha=0.3)
                axes[0,0].legend(title='تایم‌فریم')
            
            # نمودار ۲: نرخ برد
            if not summary_df.empty:
                pivot_winrate = summary_df.pivot(index='symbol', columns='timeframe', values='win_rate')
                pivot_winrate.plot(kind='bar', ax=axes[0,1], 
                                 title='نرخ برد بر اساس نماد و تایم‌فریم (%)',
                                 color=['#2E8B57', '#4682B4', '#D2691E'])
                axes[0,1].set_ylabel('نرخ برد (%)')
                axes[0,1].tick_params(axis='x', rotation=45)
                axes[0,1].grid(True, alpha=0.3)
                axes[0,1].legend(title='تایم‌فریم')
            
            # نمودار ۳: توزیع سود/ضرر
            all_trades_df = pd.DataFrame(self.detailed_trades)
            if not all_trades_df.empty:
                # هیستوگرام سود/ضرر
                profits = all_trades_df['pnl_percentage']
                axes[1,0].hist(profits, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[1,0].axvline(profits.mean(), color='red', linestyle='--', 
                                label=f'میانگین: {profits.mean():.2f}%')
                axes[1,0].set_title('توزیع سود/ضرر معاملات')
                axes[1,0].set_xlabel('سود/ضرر (%)')
                axes[1,0].set_ylabel('تعداد معاملات')
                axes[1,0].legend()
                axes[1,0].grid(True, alpha=0.3)
                
                # نمودار ۴: سود/ضرر تجمعی
                all_trades_df = all_trades_df.sort_values('exit_time')
                all_trades_df['cumulative_pnl'] = all_trades_df['pnl_amount'].cumsum() + self.initial_capital
                axes[1,1].plot(all_trades_df['exit_time'], all_trades_df['cumulative_pnl'], 
                             linewidth=2, color='green')
                axes[1,1].axhline(y=self.initial_capital, color='red', linestyle='--', 
                                label=f'سرمایه اولیه: ${self.initial_capital:,.0f}')
                axes[1,1].set_title('سود/ضرر تجمعی')
                axes[1,1].set_xlabel('زمان')
                axes[1,1].set_ylabel('سرمایه ($)')
                axes[1,1].tick_params(axis='x', rotation=45)
                axes[1,1].legend()
                axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('backtest_results/performance_charts.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("✅ نمودارهای عملکرد ایجاد شدند")
            
        except Exception as e:
            logger.error(f"❌ خطا در ایجاد نمودارها: {e}")
    
    def generate_text_report(self, summary_df: pd.DataFrame, all_results: List[Dict]):
        """تولید گزارش متنی کامل"""
        try:
            report = []
            report.append("=" * 100)
            report.append("📊 گزارش جامع بکتست استراتژی Adaptive Elite RSI")
            report.append("=" * 100)
            report.append(f"🕒 تاریخ تولید: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"💰 سرمایه اولیه: ${self.initial_capital:,.2f}")
            report.append(f"📈 تعداد نمادها: {len(self.symbols)}")
            report.append(f"⏰ تعداد تایم‌فریم‌ها: {len(self.timeframes)}")
            report.append("")
            
            # آمار کلی
            total_trades = len(self.detailed_trades)
            winning_trades = len([t for t in self.detailed_trades if t['pnl_amount'] > 0])
            losing_trades = len([t for t in self.detailed_trades if t['pnl_amount'] < 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            total_pnl = sum(t['pnl_amount'] for t in self.detailed_trades)
            final_equity = self.initial_capital + total_pnl
            total_return = (total_pnl / self.initial_capital) * 100
            
            report.append("📈 آمار کلی عملکرد:")
            report.append(f"  • کل معاملات: {total_trades}")
            report.append(f"  • معاملات سودده: {winning_trades}")
            report.append(f"  • معاملات ضررده: {losing_trades}")
            report.append(f"  • نرخ برد: {win_rate:.2f}%")
            report.append(f"  • سود/ضرر کل: ${total_pnl:,.2f}")
            report.append(f"  • بازدهی کل: {total_return:.2f}%")
            report.append(f"  • سرمایه نهایی: ${final_equity:,.2f}")
            report.append("")
            
            # بهترین و بدترین نمادها
            if not summary_df.empty:
                best_performer = summary_df.loc[summary_df['total_return'].idxmax()]
                worst_performer = summary_df.loc[summary_df['total_return'].idxmin()]
                
                report.append("🏆 بهترین عملکرد:")
                report.append(f"  • {best_performer['symbol']} ({best_performer['timeframe']})")
                report.append(f"    بازدهی: {best_performer['total_return']:.2f}%")
                report.append(f"    معاملات: {best_performer['total_trades']}")
                report.append(f"    نرخ برد: {best_performer['win_rate']:.2f}%")
                report.append("")
                
                report.append("📉 بدترین عملکرد:")
                report.append(f"  • {worst_performer['symbol']} ({worst_performer['timeframe']})")
                report.append(f"    بازدهی: {worst_performer['total_return']:.2f}%")
                report.append(f"    معاملات: {worst_performer['total_trades']}")
                report.append(f"    نرخ برد: {worst_performer['win_rate']:.2f}%")
                report.append("")
            
            # تحلیل معاملات
            if self.detailed_trades:
                trades_df = pd.DataFrame(self.detailed_trades)
                
                # بر اساس نوع پوزیشن
                long_trades = trades_df[trades_df['position_type'] == 'LONG']
                short_trades = trades_df[trades_df['position_type'] == 'SHORT']
                
                report.append("🔍 تحلیل بر اساس نوع معامله:")
                if not long_trades.empty:
                    long_win_rate = (len(long_trades[long_trades['pnl_amount'] > 0]) / len(long_trades)) * 100
                    report.append(f"  • LONG: {len(long_trades)} معامله - نرخ برد: {long_win_rate:.2f}%")
                
                if not short_trades.empty:
                    short_win_rate = (len(short_trades[short_trades['pnl_amount'] > 0]) / len(short_trades)) * 100
                    report.append(f"  • SHORT: {len(short_trades)} معامله - نرخ برد: {short_win_rate:.2f}%")
                report.append("")
                
                # بر اساس دلیل خروج
                exit_reasons = trades_df['exit_reason'].value_counts()
                report.append("🔚 دلایل خروج از معاملات:")
                for reason, count in exit_reasons.items():
                    percentage = (count / len(trades_df)) * 100
                    report.append(f"  • {reason}: {count} معامله ({percentage:.1f}%)")
                report.append("")
            
            # پارامترهای استراتژی
            report.append("⚙️ پارامترهای استراتژی:")
            for key, value in self.strategy_params.items():
                report.append(f"  • {key}: {value}")
            report.append("")
            
            # توصیه‌ها
            report.append("💡 توصیه‌های بهبود:")
            if win_rate < 50:
                report.append("  • بهبود نرخ برد نیاز به تنظیم پارامترهای RSI دارد")
            if total_return < 0:
                report.append("  • بازنگری در استراتژی یا مدیریت ریسک ضروری است")
            if len(self.detailed_trades) < 10:
                report.append("  • تعداد معاملات کم است - تست روی دوره زمانی طولانی‌تر")
            report.append("  • بررسی عملکرد در بازارهای مختلف ادامه دار باشد")
            
            # ذخیره گزارش
            with open('backtest_results/final_report.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            
            # نمایش در کنسول
            print("\n" + "="*100)
            for line in report[:20]:  # نمایش بخشی از گزارش در کنسول
                print(line)
            print("="*100)
            print("📄 گزارش کامل در فایل 'backtest_results/final_report.txt' ذخیره شد")
            
        except Exception as e:
            logger.error(f"❌ خطا در تولید گزارش متنی: {e}")
    
    def run_quick_test(self):
        """اجرای تست سریع برای بررسی عملکرد"""
        logger.info("🔍 اجرای تست سریع")
        
        # تست روی یک نماد
        symbol = "EURUSD"
        timeframe = "H1"
        
        result = self.run_single_backtest(symbol, timeframe, days_back=30)
        
        if result:
            print(f"\n📊 نتایج تست سریع {symbol} ({timeframe}):")
            print(f"• معاملات: {result['total_trades']}")
            print(f"• بازدهی: {result['total_return']:.2f}%")
            print(f"• نرخ برد: {result['metrics'].get('win_rate', 0):.2f}%")
            print(f"• فاکتور سود: {result['metrics'].get('profit_factor', 0):.2f}")
        else:
            print("❌ تست سریع انجام نشد")

def main():
    """تابع اصلی اجرای بکتست"""
    print("🎯 بکتست استراتژی Adaptive Elite RSI")
    print("=" * 50)
    
    try:
        # ایجاد بکتستر
        backtester = AdaptiveRSIBacktester(initial_capital=10000.0)
        
        # بررسی اتصال
        if not backtester.check_mt5_connection():
            print("❌ اتصال به MT5 برقرار نیست. لطفا مطمئن شوید:")
            print("  1. MetaTrader5 نصب است")
            print("  2. MT5 اجرا است و اکانت متصل است")
            print("  3. نمادهای مورد نظر در MT5 موجود هستند")
            return
        
        # اجرای تست سریع
        backtester.run_quick_test()
        
        # سوال برای اجرای بکتست کامل
        response = input("\n🔍 آیا می‌خواهید بکتست کامل اجرا شود؟ (y/n): ")
        if response.lower() in ['y', 'yes', 'بله']:
            results = backtester.run_comprehensive_backtest()
            
            if results:
                print(f"\n✅ بکتست کامل تکمیل شد!")
                print(f"📁 نتایج در پوشه 'backtest_results' ذخیره شدند")
                print(f"📊 تعداد تست‌های موفق: {len(results)}")
            else:
                print("❌ بکتست نتیجه‌ای نداشت")
        
    except Exception as e:
        logger.error(f"❌ خطا در اجرای بکتست: {e}")
        print(f"❌ خطا: {e}")

if __name__ == "__main__":
    main()