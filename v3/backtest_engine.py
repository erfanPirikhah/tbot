# backtest_engine.py
"""
بک‌تست حرفه‌ای برای استراتژی EnhancedRsiStrategyV2
🎯 ویژگی‌ها:
1. ✅ شبیه‌سازی دقیق معاملات (شامل کمیسیون و لغزش قیمت)
2. ✅ گزارش کامل معیارهای عملکرد
3. ✅ نمودارهای تحلیلی (منحنی سرمایه، دراوداون، بازده ماهانه)
4. ✅ لاگ کامل تمام معاملات
5. ✅ قابلیت تنظیم پارامترهای مختلف
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# وارد کردن استراتژی
from enhanced_rsi_strategy_v2 import EnhancedRsiStrategyV2

# تنظیمات فارسی برای نمودارها
plt.rcParams['font.family'] = 'B Nazanin'
plt.rcParams['axes.unicode_minus'] = False

class Backtester:
    """
    موتور اصلی بک‌تست برای شبیه‌سازی معاملات
    """
    
    def __init__(
        self,
        strategy: EnhancedRsiStrategyV2,
        data: pd.DataFrame,
        initial_cash: float = 10000,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0005,    # 0.05%
    ):
        self.strategy = strategy
        self.data = data.copy()
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        
        # متغیرهای داخلی
        self.cash = initial_cash
        self.position = 0.0
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.open_orders: List[Dict] = []
        
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """محاسبه RSI"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _prepare_data(self):
        """آماده‌سازی داده‌ها و محاسبه اندیکاتورها"""
        # محاسبه RSI
        self.data['RSI'] = self._calculate_rsi(self.data, self.strategy.rsi_period)
        
        # حذف ردیف‌های خالی
        self.data.dropna(inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        
        print(f"📊 داده‌های آماده شده: {len(self.data)} کندل")
        print(f"📅 بازه زمانی: {self.data.index[0]} تا {self.data.index[-1]}")
    
    def _execute_order(self, action: str, price: float, quantity: float, reason: str = "") -> Dict:
        """اجرای سفارش با در نظر گرفتن کمیسیون و لغزش قیمت"""
        # اعمال لغزش قیمت
        if action in ['BUY', 'SHORT']:
            execution_price = price * (1 + self.slippage)
            cost = execution_price * quantity * (1 + self.commission)
        else:  # SELL, COVER
            execution_price = price * (1 - self.slippage)
            cost = execution_price * quantity * (1 - self.commission)
        
        return {
            'action': action,
            'price': execution_price,
            'quantity': quantity,
            'cost': cost,
            'reason': reason
        }
    
    def _update_equity(self, current_price: float, timestamp):
        """به‌روزرسانی ارزش پورتفو"""
        equity = self.cash + (self.position * current_price)
        self.equity_curve.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'position': self.position,
            'equity': equity
        })
    
    def run(self) -> Dict[str, Any]:
        """اجرای اصلی بک‌تست"""
        print("🚀 شروع بک‌تست...")
        
        # آماده‌سازی داده‌ها
        self._prepare_data()
        
        # ریست استراتژی
        self.strategy.reset_state()
        
        # حلقه اصلی بک‌تست
        for i in range(len(self.data)):
            current_candle = self.data.iloc[i]
            current_price = current_candle['close']
            timestamp = current_candle.name if hasattr(current_candle.name, 'strftime') else pd.Timestamp.now()
            
            # به‌روزرسانی ارزش پورتفو
            self._update_equity(current_price, timestamp)
            
            # دریافت سیگنال از استراتژی
            current_data = self.data.iloc[:i+1]
            signal = self.strategy.generate_signal(current_data, i)
            
            # پردازش سیگنال
            if signal['action'] == 'BUY':
                quantity = signal['position_size']
                if self.cash >= (current_price * quantity * (1 + self.commission)):
                    order = self._execute_order('BUY', current_price, quantity, signal.get('reason', ''))
                    self.cash -= order['cost']
                    self.position += order['quantity']
                    
                    self.trades.append({
                        'entry_time': timestamp,
                        'entry_price': order['price'],
                        'quantity': order['quantity'],
                        'type': 'LONG',
                        'entry_reason': signal.get('reason', ''),
                    })
                    
            elif signal['action'] == 'SELL' and self.position > 0:
                order = self._execute_order('SELL', current_price, self.position, signal.get('reason', ''))
                self.cash += order['cost']
                
                # محاسبه سود/ضرر
                last_trade = self.trades[-1]
                pnl = (order['price'] - last_trade['entry_price']) * last_trade['quantity']
                pnl_pct = (pnl / (last_trade['entry_price'] * last_trade['quantity'])) * 100
                
                self.trades[-1].update({
                    'exit_time': timestamp,
                    'exit_price': order['price'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': signal.get('reason', ''),
                })
                
                self.position = 0
                
            elif signal['action'] == 'SHORT':
                quantity = signal['position_size']
                if self.cash >= (current_price * quantity * (1 + self.commission)):
                    order = self._execute_order('SHORT', current_price, quantity, signal.get('reason', ''))
                    self.cash -= order['cost']
                    self.position -= order['quantity']
                    
                    self.trades.append({
                        'entry_time': timestamp,
                        'entry_price': order['price'],
                        'quantity': order['quantity'],
                        'type': 'SHORT',
                        'entry_reason': signal.get('reason', ''),
                    })
                    
            elif signal['action'] == 'COVER' and self.position < 0:
                order = self._execute_order('COVER', current_price, abs(self.position), signal.get('reason', ''))
                self.cash += order['cost']
                
                # محاسبه سود/ضرر
                last_trade = self.trades[-1]
                pnl = (last_trade['entry_price'] - order['price']) * last_trade['quantity']
                pnl_pct = (pnl / (last_trade['entry_price'] * last_trade['quantity'])) * 100
                
                self.trades[-1].update({
                    'exit_time': timestamp,
                    'exit_price': order['price'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': signal.get('reason', ''),
                })
                
                self.position = 0
        
        # بستن پوزیشن‌های باز در انتها
        if self.position != 0:
            last_price = self.data['close'].iloc[-1]
            if self.position > 0:
                order = self._execute_order('SELL', last_price, self.position, 'Force close')
            else:
                order = self._execute_order('COVER', last_price, abs(self.position), 'Force close')
            
            self.cash += order['cost']
            
            last_trade = self.trades[-1]
            if last_trade['type'] == 'LONG':
                pnl = (order['price'] - last_trade['entry_price']) * last_trade['quantity']
            else:
                pnl = (last_trade['entry_price'] - order['price']) * last_trade['quantity']
            
            pnl_pct = (pnl / (last_trade['entry_price'] * last_trade['quantity'])) * 100
            
            self.trades[-1].update({
                'exit_time': self.data.index[-1],
                'exit_price': order['price'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': 'Force close',
            })
            
            self.position = 0
        
        print("✅ بک‌تست با موفقیت تکمیل شد")
        
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'final_cash': self.cash,
            'final_equity': self.equity_curve[-1]['equity'] if self.equity_curve else self.initial_cash
        }

class BacktestReport:
    """
    کلاس تولید گزارش‌های تحلیلی از نتایج بک‌تست
    """
    
    def __init__(self, results: Dict[str, Any], initial_cash: float):
        self.results = results
        self.initial_cash = initial_cash
        self.trades_df = pd.DataFrame(results['trades'])
        self.equity_df = pd.DataFrame(results['equity_curve'])
        
    def _calculate_metrics(self) -> Dict[str, Any]:
        """محاسبه معیارهای عملکرد"""
        if self.trades_df.empty:
            return {}
        
        # معیارهای پایه
        total_trades = len(self.trades_df)
        winning_trades = len(self.trades_df[self.trades_df['pnl'] > 0])
        losing_trades = len(self.trades_df[self.trades_df['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # معیارهای سود/ضرر
        total_pnl = self.trades_df['pnl'].sum()
        gross_profit = self.trades_df[self.trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(self.trades_df[self.trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = self.trades_df[self.trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = self.trades_df[self.trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        # معیارهای ریسک
        equity_series = self.equity_df['equity']
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # محاسبه بازده روزانه
        daily_returns = equity_series.pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if len(daily_returns) > 1 else 0
        
        negative_returns = daily_returns[daily_returns < 0]
        sortino_ratio = np.sqrt(252) * daily_returns.mean() / negative_returns.std() if len(negative_returns) > 1 else 0
        
        # معیارهای زمانی
        self.trades_df['duration'] = (self.trades_df['exit_time'] - self.trades_df['entry_time']).dt.total_seconds() / 3600
        avg_duration = self.trades_df['duration'].mean()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'avg_duration': avg_duration,
            'initial_cash': self.initial_cash,
            'final_cash': self.results['final_cash'],
            'final_equity': self.results['final_equity'],
            'total_return': ((self.results['final_equity'] - self.initial_cash) / self.initial_cash) * 100
        }
    
    def print_report(self):
        """چاپ گزارش متنی کامل"""
        metrics = self._calculate_metrics()
        
        print("\n" + "="*60)
        print("📊 گزارش کامل بک‌تست استراتژی RSI پیشرفته")
        print("="*60)
        
        print("\n💰 معیارهای عملکرد کلی:")
        print(f"   سرمایه اولیه: ${metrics['initial_cash']:,.2f}")
        print(f"   ارزش نهایی: ${metrics['final_equity']:,.2f}")
        print(f"   بازده کل: {metrics['total_return']:.2f}%")
        print(f"   حداکثر دراوداون: {metrics['max_drawdown']:.2f}%")
        
        print("\n📈 آمار معاملات:")
        print(f"   تعداد کل معاملات: {metrics['total_trades']}")
        print(f"   معاملات سودده: {metrics['winning_trades']}")
        print(f"   معاملات ضررده: {metrics['losing_trades']}")
        print(f"   نرخ برد: {metrics['win_rate']:.2f}%")
        print(f"   میانگین مدت معامله: {metrics['avg_duration']:.1f} ساعت")
        
        print("\n💵 معیارهای سود/ضرر:")
        print(f"   مجموع سود: ${metrics['gross_profit']:,.2f}")
        print(f"   مجموع ضرر: ${metrics['gross_loss']:,.2f}")
        print(f"   فاکتور سود: {metrics['profit_factor']:.2f}")
        print(f"   میانگین سود هر معامله: ${metrics['avg_win']:,.2f}")
        print(f"   میانگین ضرر هر معامله: ${metrics['avg_loss']:,.2f}")
        print(f"   میانگین سود هر معامله: ${metrics['avg_trade']:,.2f}")
        
        print("\n⚠️ معیارهای ریسک:")
        print(f"   نسبت شارپ: {metrics['sharpe_ratio']:.3f}")
        print(f"   نسبت سورتینو: {metrics['sortino_ratio']:.3f}")
        
        print("\n📋 جزئیات ۱۰ معامله آخر:")
        print(self.trades_df.tail(10)[['entry_time', 'entry_price', 'exit_time', 'exit_price', 'pnl', 'pnl_pct', 'exit_reason']].to_string())
        
        print("\n" + "="*60)
    
    def plot_results(self, figsize=(20, 12)):
        """رسم نمودارهای تحلیلی"""
        metrics = self._calculate_metrics()
        
        fig = plt.figure(figsize=figsize)
        
        # 1. منحنی سرمایه
        ax1 = plt.subplot(2, 3, 1)
        equity_series = pd.Series(self.equity_df['equity'].values, index=self.equity_df['timestamp'])
        equity_series.plot(ax=ax1, color='blue', linewidth=2)
        ax1.set_title('منحنی سرمایه (Equity Curve)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('ارزش پورتفو ($)')
        ax1.grid(True, alpha=0.3)
        
        # 2. نمودار دراوداون
        ax2 = plt.subplot(2, 3, 2)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        drawdown.plot(ax=ax2, color='red', linewidth=1)
        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax2.set_title('نمودار دراوداون (Drawdown)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('دراوداون (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. توزیع سود/ضرر معاملات
        ax3 = plt.subplot(2, 3, 3)
        if not self.trades_df.empty:
            colors = ['green' if x > 0 else 'red' for x in self.trades_df['pnl']]
            ax3.hist(self.trades_df['pnl'], bins=30, color=colors, alpha=0.7, edgecolor='black')
            ax3.set_title('توزیع سود/ضرر معاملات', fontsize=12, fontweight='bold')
            ax3.set_xlabel('سود/ضرر ($)')
            ax3.set_ylabel('تعداد معاملات')
            ax3.axvline(0, color='black', linestyle='--', linewidth=1)
            ax3.grid(True, alpha=0.3)
        
        # 4. سود هر معامله
        ax4 = plt.subplot(2, 3, 4)
        if not self.trades_df.empty:
            trade_numbers = range(1, len(self.trades_df) + 1)
            colors = ['green' if x > 0 else 'red' for x in self.trades_df['pnl']]
            ax4.bar(trade_numbers, self.trades_df['pnl'], color=colors, alpha=0.7)
            ax4.set_title('سود/ضرر هر معامله', fontsize=12, fontweight='bold')
            ax4.set_xlabel('شماره معامله')
            ax4.set_ylabel('سود/ضرر ($)')
            ax4.axhline(0, color='black', linestyle='-', linewidth=1)
            ax4.grid(True, alpha=0.3)
        
        # 5. بازده ماهانه
        ax5 = plt.subplot(2, 3, 5)
        if not self.equity_df.empty:
            self.equity_df['timestamp'] = pd.to_datetime(self.equity_df['timestamp'])
            monthly_returns = self.equity_df.set_index('timestamp')['equity'].resample('M').last().pct_change() * 100
            monthly_returns = monthly_returns.dropna()
            
            if not monthly_returns.empty:
                # ایجاد هیت‌مپ
                months = monthly_returns.index.month
                years = monthly_returns.index.year
                pivot_data = pd.DataFrame({
                    'month': months,
                    'year': years,
                    'return': monthly_returns.values
                }).pivot(index='year', columns='month', values='return')
                
                sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax5)
                ax5.set_title('بازده ماهانه (%)', fontsize=12, fontweight='bold')
                ax5.set_xlabel('ماه')
                ax5.set_ylabel('سال')
        
        # 6. معیارهای کلیدی
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        metrics_text = f"""
        معیارهای کلیدی:
        ─────────────────
        بازده کل: {metrics['total_return']:.2f}%
        نرخ برد: {metrics['win_rate']:.2f}%
        فاکتور سود: {metrics['profit_factor']:.2f}
        نسبت شارپ: {metrics['sharpe_ratio']:.3f}
        حداکثر دراوداون: {metrics['max_drawdown']:.2f}%
        تعداد معاملات: {metrics['total_trades']}
        """
        
        ax6.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.show()

def run_backtest(
    data_path: str,
    strategy_params: Optional[Dict] = None,
    initial_cash: float = 10000,
    commission: float = 0.001,
    slippage: float = 0.0005,
):
    """
    تابع اصلی برای اجرای بک‌تست
    
    Args:
        data_path: مسیر فایل داده‌ها (CSV)
        strategy_params: پارامترهای استراتژی
        initial_cash: سرمایه اولیه
        commission: کمیسیون معاملات
        slippage: لغزش قیمت
    """
    
    # بارگذاری داده‌ها
    try:
        data = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
        print(f"✅ داده‌ها با موفقیت بارگذاری شدند: {len(data)} ردیف")
    except Exception as e:
        print(f"❌ خطا در بارگذاری داده‌ها: {e}")
        return
    
    # تنظیمات پیش‌فرض استراتژی
    if strategy_params is None:
        strategy_params = {
            'rsi_period': 14,
            'rsi_base_oversold': 30,
            'rsi_base_overbought': 70,
            'risk_per_trade': 0.02,
            'enable_pyramiding': True,
            'pyramid_profit_threshold': 1.5,
            'enable_trailing_stop': True,
            'enable_adaptive_rsi': True,
            'enable_analytical_logging': False,  # غیرفعال در بک‌تست برای جلوگیری از شلوغی
        }
    
    # ساخت استراتژی
    strategy = EnhancedRsiStrategyV2(**strategy_params)
    
    # اجرای بک‌تست
    backtester = Backtester(
        strategy=strategy,
        data=data,
        initial_cash=initial_cash,
        commission=commission,
        slippage=slippage
    )
    
    results = backtester.run()
    
    # تولید گزارش
    report = BacktestReport(results, initial_cash)
    report.print_report()
    report.plot_results()
    
    return report

# مثال استفاده
if __name__ == "__main__":
    # تنظیمات بک‌تست
    config = {
        'data_path': 'BTCUSDT_1h.csv',  # مسیر فایل داده‌ها
        'initial_cash': 10000,
        'commission': 0.001,  # 0.1%
        'slippage': 0.0005,   # 0.05%
    }
    
    # پارامترهای استراتژی
    strategy_config = {
        'rsi_period': 14,
        'rsi_base_oversold': 30,
        'rsi_base_overbought': 70,
        'risk_per_trade': 0.02,
        'enable_pyramiding': True,
        'pyramid_profit_threshold': 1.5,
        'pyramid_max_entries': 3,
        'enable_trailing_stop': True,
        'trailing_atr_multiplier': 1.5,
        'enable_adaptive_rsi': True,
        'adaptive_rsi_sensitivity': 0.5,
        'enable_analytical_logging': False,
    }
    
    # اجرای بک‌تست
    report = run_backtest(
        data_path=config['data_path'],
        strategy_params=strategy_config,
        initial_cash=config['initial_cash'],
        commission=config['commission'],
        slippage=config['slippage']
    )
    
    # دسترسی به نتایج برای تحلیل بیشتر
    trades_df = pd.DataFrame(report.results['trades'])
    equity_df = pd.DataFrame(report.results['equity_curve'])
    
    print("\n📁 داده‌ها برای تحلیل بیشتر در متغیرهای trades_df و equity_df ذخیره شدند")