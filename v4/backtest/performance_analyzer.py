# backtest/performance_analyzer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """تحلیل‌گر پیشرفته عملکرد استراتژی"""
    
    def __init__(self, trades_data: pd.DataFrame, equity_curve: pd.DataFrame):
        self.trades_data = trades_data
        self.equity_curve = equity_curve
        self.metrics = {}
        
        logger.info("✅ Performance Analyzer initialized")

    def calculate_comprehensive_metrics(self) -> Dict[str, Any]:
        """محاسبه معیارهای جامع عملکرد"""
        try:
            # معیارهای پایه
            self._calculate_basic_metrics()
            
            # معیارهای پیشرفته
            self._calculate_advanced_metrics()
            
            # تحلیل ریسک
            self._calculate_risk_metrics()
            
            # تحلیل زمانی
            self._calculate_time_based_metrics()
            
            logger.info("✅ Comprehensive metrics calculated")
            return self.metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating metrics: {e}")
            return {}

    def _calculate_basic_metrics(self):
        """محاسبه معیارهای پایه"""
        if self.trades_data.empty:
            return
            
        # معیارهای معاملاتی
        total_trades = len(self.trades_data)
        winning_trades = len(self.trades_data[self.trades_data['pnl_percentage'] > 0])
        losing_trades = len(self.trades_data[self.trades_data['pnl_percentage'] < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # سود/زیان
        total_pnl = self.trades_data['pnl_percentage'].sum()
        avg_win = self.trades_data[self.trades_data['pnl_percentage'] > 0]['pnl_percentage'].mean()
        avg_loss = self.trades_data[self.trades_data['pnl_percentage'] < 0]['pnl_percentage'].mean()
        
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else float('inf')
        
        self.metrics.update({
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl_pct': round(total_pnl, 2),
            'avg_win_pct': round(avg_win, 2) if not np.isnan(avg_win) else 0,
            'avg_loss_pct': round(avg_loss, 2) if not np.isnan(avg_loss) else 0,
            'profit_factor': round(profit_factor, 2)
        })

    def _calculate_advanced_metrics(self):
        """محاسبه معیارهای پیشرفته"""
        if self.equity_curve.empty:
            return
            
        # محاسبات بر اساس منحنی سرمایه
        returns = self.equity_curve['portfolio_value'].pct_change().dropna()
        
        # شارپ ریشو
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # سورتینو ریشو
        negative_returns = returns[returns < 0]
        sortino_ratio = (returns.mean() / negative_returns.std() * np.sqrt(252)) if len(negative_returns) > 0 else 0
        
        # کالمار ریشو
        max_drawdown = self._calculate_max_drawdown()
        calmar_ratio = (returns.mean() * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
        
        self.metrics.update({
            'sharpe_ratio': round(sharpe_ratio, 2),
            'sortino_ratio': round(sortino_ratio, 2),
            'calmar_ratio': round(calmar_ratio, 2),
            'volatility_pct': round(returns.std() * np.sqrt(252) * 100, 2)
        })

    def _calculate_risk_metrics(self):
        """محاسبه معیارهای ریسک"""
        if self.equity_curve.empty:
            return
            
        # حداکثر افت سرمایه
        max_drawdown = self._calculate_max_drawdown()
        
        # واریانس و ارزش در معرض ریسک (VaR)
        returns = self.equity_curve['portfolio_value'].pct_change().dropna()
        var_95 = np.percentile(returns, 5)
        
        # افت‌های متوالی
        consecutive_losses = self._calculate_consecutive_losses()
        
        self.metrics.update({
            'max_drawdown_pct': round(max_drawdown, 2),
            'var_95': round(var_95 * 100, 2),
            'max_consecutive_losses': consecutive_losses,
            'avg_trade_duration': self._calculate_avg_trade_duration()
        })

    def _calculate_time_based_metrics(self):
        """محاسبه معیارهای مبتنی بر زمان"""
        if self.trades_data.empty:
            return
            
        # تحلیل ساعتی
        self.trades_data['hour'] = self.trades_data['timestamp'].dt.hour
        hourly_performance = self.trades_data.groupby('hour')['pnl_percentage'].mean()
        
        best_hour = hourly_performance.idxmax() if not hourly_performance.empty else -1
        worst_hour = hourly_performance.idxmin() if not hourly_performance.empty else -1
        
        # تحلیل روزهای هفته
        self.trades_data['weekday'] = self.trades_data['timestamp'].dt.day_name()
        weekday_performance = self.trades_data.groupby('weekday')['pnl_percentage'].mean()
        
        self.metrics.update({
            'best_hour': best_hour,
            'worst_hour': worst_hour,
            'weekday_performance': weekday_performance.to_dict()
        })

    def _calculate_max_drawdown(self) -> float:
        """محاسبه حداکثر افت سرمایه"""
        if self.equity_curve.empty:
            return 0
            
        portfolio_values = self.equity_curve['portfolio_value']
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max * 100
        
        return drawdowns.min()

    def _calculate_consecutive_losses(self) -> int:
        """محاسبه بیشترین تعداد ضررهای متوالی"""
        if self.trades_data.empty:
            return 0
            
        pnl_series = self.trades_data['pnl_percentage'] < 0
        consecutive_losses = 0
        current_streak = 0
        
        for is_loss in pnl_series:
            if is_loss:
                current_streak += 1
                consecutive_losses = max(consecutive_losses, current_streak)
            else:
                current_streak = 0
                
        return consecutive_losses

    def _calculate_avg_trade_duration(self) -> float:
        """محاسبه میانگین مدت زمان معاملات"""
        # این تابع نیاز به داده‌های زمانی دقیق‌تری دارد
        return 0.0

    def generate_performance_report(self) -> str:
        """تولید گزارش عملکرد"""
        report = [
            "=" * 70,
            "ANALYSIS REPORT - ENHANCED RSI STRATEGY V4",
            "=" * 70,
            f"تعداد معاملات: {self.metrics.get('total_trades', 0)}",
            f"نرخ برد: {self.metrics.get('win_rate', 0):.1f}%",
            f"سود/زیان کل: {self.metrics.get('total_pnl_pct', 0):.2f}%",
            f"فاکتور سود: {self.metrics.get('profit_factor', 0):.2f}",
            "",
            "معیارهای ریسک:",
            f"حداکثر افت سرمایه: {self.metrics.get('max_drawdown_pct', 0):.2f}%",
            f"نسبت شارپ: {self.metrics.get('sharpe_ratio', 0):.2f}",
            f"نسبت سورتینو: {self.metrics.get('sortino_ratio', 0):.2f}",
            f"VaR (95%): {self.metrics.get('var_95', 0):.2f}%",
            "",
            "تحلیل زمانی:",
            f"بهترین ساعت معامله: {self.metrics.get('best_hour', -1)}",
            f"بدترین ساعت معامله: {self.metrics.get('worst_hour', -1)}",
            "=" * 70
        ]
        
        return "\n".join(report)

    def plot_detailed_analysis(self, save_path: str = None):
        """ترسیم تحلیل‌های دقیق"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Detailed Performance Analysis - RSI Strategy V4', fontsize=16)
            
            # نمودار ۱: توزیع سود/زیان
            if not self.trades_data.empty:
                axes[0, 0].hist(self.trades_data['pnl_percentage'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=1)
                axes[0, 0].set_title('Distribution of Trade PnL (%)')
                axes[0, 0].set_xlabel('PnL %')
                axes[0, 0].set_ylabel('Frequency')
            
            # نمودار ۲: منحنی سرمایه
            if not self.equity_curve.empty:
                axes[0, 1].plot(self.equity_curve.index, self.equity_curve['portfolio_value'], linewidth=2)
                axes[0, 1].set_title('Equity Curve')
                axes[0, 1].set_ylabel('Portfolio Value ($)')
                axes[0, 1].grid(True, alpha=0.3)
            
            # نمودار ۳: افت سرمایه
            if not self.equity_curve.empty:
                rolling_max = self.equity_curve['portfolio_value'].expanding().max()
                drawdown = (self.equity_curve['portfolio_value'] - rolling_max) / rolling_max * 100
                axes[0, 2].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
                axes[0, 2].set_title('Drawdown %')
                axes[0, 2].set_ylabel('Drawdown %')
                axes[0, 2].grid(True, alpha=0.3)
            
            # نمودار ۴: عملکرد ساعتی
            if not self.trades_data.empty:
                hourly_pnl = self.trades_data.groupby('hour')['pnl_percentage'].mean()
                axes[1, 0].bar(hourly_pnl.index, hourly_pnl.values, alpha=0.7)
                axes[1, 0].set_title('Average PnL by Hour')
                axes[1, 0].set_xlabel('Hour of Day')
                axes[1, 0].set_ylabel('Average PnL %')
            
            # نمودار ۵: اندازه معاملات
            if not self.trades_data.empty:
                trade_sizes = self.trades_data['position_size']
                axes[1, 1].hist(trade_sizes, bins=15, alpha=0.7, color='green', edgecolor='black')
                axes[1, 1].set_title('Trade Size Distribution')
                axes[1, 1].set_xlabel('Position Size')
                axes[1, 1].set_ylabel('Frequency')
            
            # نمودار ۶: مقایسه برد و باخت
            if not self.trades_data.empty:
                win_loss_data = [
                    self.metrics.get('winning_trades', 0),
                    self.metrics.get('losing_trades', 0)
                ]
                axes[1, 2].pie(win_loss_data, labels=['Wins', 'Losses'], autopct='%1.1f%%', colors=['green', 'red'])
                axes[1, 2].set_title('Win/Loss Distribution')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"✅ Analysis plots saved to {save_path}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ Error plotting analysis: {e}")