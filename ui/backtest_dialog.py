# ui/backtest_dialog.py
import sys
import os
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QComboBox, QPushButton, QDateTimeEdit, 
                            QDoubleSpinBox, QCheckBox, QGroupBox, 
                            QFormLayout, QProgressBar, QTextEdit,
                            QTabWidget, QTableWidget, QTableWidgetItem,
                            QHeaderView, QApplication)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.run_backtest import run_strategy_backtest, compare_strategies
from config import MT5_SYMBOL_MAP, CRYPTOCOMPARE_SYMBOL_MAP, ALL_SYMBOL_MAP, ALL_INTERVAL_MAP

logger = logging.getLogger(__name__)

class BacktestThread(QThread):
    """Backtest thread"""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        
    def run(self):
        try:
            self.status_updated.emit("Fetching historical data...")
            self.progress_updated.emit(10)
            
            # Run backtest
            if self.params.get('compare', False):
                # Compare two strategies
                result = compare_strategies(
                    symbol=self.params['symbol'],
                    interval=self.params['interval'],
                    data_source=self.params['data_source'],
                    start_date=self.params['start_date'],
                    end_date=self.params['end_date'],
                    initial_balance=self.params['initial_balance'],
                    commission=self.params['commission'],
                    slippage=self.params['slippage'],
                    save_report=self.params['save_report'],
                    report_path=self.params.get('report_path')
                )
            else:
                # Single strategy backtest
                result = run_strategy_backtest(
                    symbol=self.params['symbol'],
                    interval=self.params['interval'],
                    data_source=self.params['data_source'],
                    strategy_type=self.params['strategy_type'],
                    start_date=self.params['start_date'],
                    end_date=self.params['end_date'],
                    initial_balance=self.params['initial_balance'],
                    commission=self.params['commission'],
                    slippage=self.params['slippage'],
                    save_report=self.params['save_report'],
                    report_path=self.params.get('report_path')
                )
            
            self.status_updated.emit("Backtest completed")
            self.progress_updated.emit(100)
            self.finished.emit(result)
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            self.error.emit(str(e))

class BacktestResultsDialog(QDialog):
    """Backtest results dialog"""
    
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.results = results
        self.is_comparison = isinstance(results, dict)
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Backtest Results")
        self.setMinimumSize(1000, 700)
        
        layout = QVBoxLayout()
        
        # Create tabs
        tabs = QTabWidget()
        
        # Add charts tab
        chart_tab = self.create_chart_tab()
        tabs.addTab(chart_tab, "Charts")
        
        # Add performance metrics tab
        metrics_tab = self.create_metrics_tab()
        tabs.addTab(metrics_tab, "Performance Metrics")
        
        # Add trade history tab
        trades_tab = self.create_trades_tab()
        tabs.addTab(trades_tab, "Trade History")
        
        layout.addWidget(tabs)
        
        # Add close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)
        
        self.setLayout(layout)
    
    def create_chart_tab(self):
        """Create charts tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Plot charts
        self.plot_charts()
        
        widget.setLayout(layout)
        return widget
    
    def create_metrics_tab(self):
        """Create performance metrics tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Create table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Fill data
        self.populate_metrics_table()
        
        layout.addWidget(self.metrics_table)
        widget.setLayout(layout)
        return widget
    
    def create_trades_tab(self):
        """Create trade history tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Create table
        self.trades_table = QTableWidget()
        
        # Set columns
        if self.is_comparison:
            # Comparison mode, show trades for both strategies
            self.trades_table.setColumnCount(7)
            self.trades_table.setHorizontalHeaderLabels([
                "Strategy", "Time", "Action", "Price", "Size", "P&L", "P&L %"
            ])
        else:
            # Single strategy mode
            self.trades_table.setColumnCount(6)
            self.trades_table.setHorizontalHeaderLabels([
                "Time", "Action", "Price", "Size", "P&L", "P&L %"
            ])
        
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Fill data
        self.populate_trades_table()
        
        layout.addWidget(self.trades_table)
        widget.setLayout(layout)
        return widget
    
    def plot_charts(self):
        """Plot charts"""
        self.figure.clear()
        
        if self.is_comparison:
            # Comparison mode, plot charts for both strategies
            axes = self.figure.subplots(2, 2, figsize=(12, 8))
            
            # 1. Equity curve comparison
            ax1 = axes[0, 0]
            for strategy_name, result in self.results.items():
                equity_df = result.equity_curve
                ax1.plot(equity_df['timestamp'], equity_df['equity'], label=f'{strategy_name} Strategy', linewidth=2)
            
            ax1.set_title('Equity Curve Comparison')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Account Equity')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 2. Drawdown comparison
            ax2 = axes[0, 1]
            for strategy_name, result in self.results.items():
                drawdown_df = result.drawdowns
                ax2.plot(drawdown_df['timestamp'], drawdown_df['drawdown'], label=f'{strategy_name} Strategy', linewidth=2)
            
            ax2.set_title('Drawdown Comparison')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # 3. Performance metrics comparison
            ax3 = axes[1, 0]
            metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 'win_rate', 'profit_factor']
            metric_names = ['Total Return (%)', 'Annualized Return (%)', 'Sharpe Ratio', 'Win Rate (%)', 'Profit Factor']
            
            x = range(len(metrics))
            width = 0.35
            
            for i, (strategy_name, result) in enumerate(self.results.items()):
                values = [result.performance_metrics[m] for m in metrics]
                ax3.bar([xi + i*width for xi in x], values, width, label=f'{strategy_name} Strategy')
            
            ax3.set_title('Performance Metrics Comparison')
            ax3.set_xticks([xi + width/2 for xi in x])
            ax3.set_xticklabels(metric_names, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Trade statistics comparison
            ax4 = axes[1, 1]
            trade_metrics = ['total_trades', 'winning_trades', 'losing_trades']
            trade_metric_names = ['Total Trades', 'Winning Trades', 'Losing Trades']
            
            x = range(len(trade_metrics))
            width = 0.35
            
            for i, (strategy_name, result) in enumerate(self.results.items()):
                values = [result.performance_metrics[m] for m in trade_metrics]
                ax4.bar([xi + i*width for xi in x], values, width, label=f'{strategy_name} Strategy')
            
            ax4.set_title('Trade Statistics Comparison')
            ax4.set_xticks([xi + width/2 for xi in x])
            ax4.set_xticklabels(trade_metric_names)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
        else:
            # Single strategy mode, plot charts for single strategy
            axes = self.figure.subplots(2, 2, figsize=(12, 8))
            
            result = self.results
            
            # 1. Equity curve
            ax1 = axes[0, 0]
            equity_df = result.equity_curve
            ax1.plot(equity_df['timestamp'], equity_df['equity'], linewidth=2, color='blue')
            ax1.axhline(y=result.performance_metrics['initial_balance'], color='red', linestyle='--', alpha=0.7)
            ax1.set_title('Equity Curve')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Account Equity')
            ax1.grid(True, alpha=0.3)
            
            # 2. Drawdown chart
            ax2 = axes[0, 1]
            drawdown_df = result.drawdowns
            ax2.fill_between(drawdown_df['timestamp'], drawdown_df['drawdown'], 0, color='red', alpha=0.3)
            ax2.plot(drawdown_df['timestamp'], drawdown_df['drawdown'], color='red', linewidth=1)
            ax2.set_title('Drawdown Analysis')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)
            
            # 3. Monthly returns
            ax3 = axes[1, 0]
            monthly_returns = result.monthly_returns
            if not monthly_returns.empty:
                ax3.bar(range(len(monthly_returns)), monthly_returns['return'], color='green' if monthly_returns['return'].mean() > 0 else 'red')
                ax3.set_title('Monthly Returns')
                ax3.set_xlabel('Month')
                ax3.set_ylabel('Return (%)')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No monthly return data', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Monthly Returns')
            
            # 4. Trade distribution
            ax4 = axes[1, 1]
            exit_trades = [t for t in result.trades if 'pnl' in t]
            if exit_trades:
                pnl_values = [t['pnl'] for t in exit_trades]
                ax4.hist(pnl_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
                ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                ax4.set_title('Trade P&L Distribution')
                ax4.set_xlabel('P&L Amount')
                ax4.set_ylabel('Number of Trades')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No trade data', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Trade P&L Distribution')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def populate_metrics_table(self):
        """Populate performance metrics table"""
        if self.is_comparison:
            # Comparison mode, show metrics for both strategies
            self.metrics_table.setRowCount(15)
            
            metrics = [
                ('Initial Balance', 'initial_balance'),
                ('Final Balance', 'final_balance'),
                ('Total Return', 'total_return'),
                ('Annualized Return', 'annualized_return'),
                ('Volatility', 'volatility'),
                ('Sharpe Ratio', 'sharpe_ratio'),
                ('Total Trades', 'total_trades'),
                ('Winning Trades', 'winning_trades'),
                ('Losing Trades', 'losing_trades'),
                ('Win Rate', 'win_rate'),
                ('Average Win', 'avg_win'),
                ('Average Loss', 'avg_loss'),
                ('Max Win', 'max_win'),
                ('Max Loss', 'max_loss'),
                ('Profit Factor', 'profit_factor')
            ]
            
            for i, (name, key) in enumerate(metrics):
                self.metrics_table.setItem(i, 0, QTableWidgetItem(name))
                
                # Create a string to display values for both strategies
                values_str = ""
                for strategy_name, result in self.results.items():
                    value = result.performance_metrics.get(key, 0)
                    if key in ['total_return', 'annualized_return', 'volatility', 'win_rate']:
                        values_str += f"{strategy_name}: {value:.2f}%\n"
                    else:
                        values_str += f"{strategy_name}: {value:.2f}\n"
                
                self.metrics_table.setItem(i, 1, QTableWidgetItem(values_str.strip()))
        else:
            # Single strategy mode
            result = self.results
            metrics = result.performance_metrics
            
            self.metrics_table.setRowCount(len(metrics))
            
            for i, (key, value) in enumerate(metrics.items()):
                # Convert key names to more friendly display names
                display_name = {
                    'initial_balance': 'Initial Balance',
                    'final_balance': 'Final Balance',
                    'total_return': 'Total Return',
                    'annualized_return': 'Annualized Return',
                    'volatility': 'Volatility',
                    'sharpe_ratio': 'Sharpe Ratio',
                    'total_trades': 'Total Trades',
                    'winning_trades': 'Winning Trades',
                    'losing_trades': 'Losing Trades',
                    'win_rate': 'Win Rate',
                    'avg_win': 'Average Win',
                    'avg_loss': 'Average Loss',
                    'max_win': 'Max Win',
                    'max_loss': 'Max Loss',
                    'profit_factor': 'Profit Factor',
                    'avg_duration_hours': 'Avg Duration (hrs)'
                }.get(key, key)
                
                self.metrics_table.setItem(i, 0, QTableWidgetItem(display_name))
                
                # Format value
                if key in ['total_return', 'annualized_return', 'volatility', 'win_rate']:
                    formatted_value = f"{value:.2f}%"
                elif key in ['initial_balance', 'final_balance', 'avg_win', 'avg_loss', 'max_win', 'max_loss']:
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                
                self.metrics_table.setItem(i, 1, QTableWidgetItem(formatted_value))
    
    def populate_trades_table(self):
        """Populate trade history table"""
        if self.is_comparison:
            # Comparison mode, show trades for both strategies
            all_trades = []
            for strategy_name, result in self.results.items():
                for trade in result.trades:
                    if 'pnl' in trade:  # Only show trades with P&L (closed trades)
                        trade_copy = trade.copy()
                        trade_copy['strategy'] = strategy_name
                        all_trades.append(trade_copy)
            
            self.trades_table.setRowCount(len(all_trades))
            
            for i, trade in enumerate(all_trades):
                # Strategy name
                self.trades_table.setItem(i, 0, QTableWidgetItem(trade['strategy']))
                
                # Time
                time_str = trade['timestamp'].strftime('%Y-%m-%d %H:%M') if hasattr(trade['timestamp'], 'strftime') else str(trade['timestamp'])
                self.trades_table.setItem(i, 1, QTableWidgetItem(time_str))
                
                # Action
                self.trades_table.setItem(i, 2, QTableWidgetItem(trade['action']))
                
                # Price
                self.trades_table.setItem(i, 3, QTableWidgetItem(f"{trade['price']:.4f}"))
                
                # Size
                self.trades_table.setItem(i, 4, QTableWidgetItem(f"{trade['size']:.4f}"))
                
                # P&L
                pnl = trade.get('pnl', 0)
                pnl_item = QTableWidgetItem(f"{pnl:.2f}")
                pnl_item.setForeground(Qt.green if pnl > 0 else Qt.red)
                self.trades_table.setItem(i, 5, QTableWidgetItem(pnl_item))
                
                # P&L percentage
                pnl_pct = trade.get('pnl_percentage', 0)
                pnl_pct_item = QTableWidgetItem(f"{pnl_pct:.2f}%")
                pnl_pct_item.setForeground(Qt.green if pnl_pct > 0 else Qt.red)
                self.trades_table.setItem(i, 6, QTableWidgetItem(pnl_pct_item))
        else:
            # Single strategy mode
            result = self.results
            exit_trades = [t for t in result.trades if 'pnl' in t]
            
            self.trades_table.setRowCount(len(exit_trades))
            
            for i, trade in enumerate(exit_trades):
                # Time
                time_str = trade['timestamp'].strftime('%Y-%m-%d %H:%M') if hasattr(trade['timestamp'], 'strftime') else str(trade['timestamp'])
                self.trades_table.setItem(i, 0, QTableWidgetItem(time_str))
                
                # Action
                self.trades_table.setItem(i, 1, QTableWidgetItem(trade['action']))
                
                # Price
                self.trades_table.setItem(i, 2, QTableWidgetItem(f"{trade['price']:.4f}"))
                
                # Size
                self.trades_table.setItem(i, 3, QTableWidgetItem(f"{trade['size']:.4f}"))
                
                # P&L
                pnl = trade.get('pnl', 0)
                pnl_item = QTableWidgetItem(f"{pnl:.2f}")
                pnl_item.setForeground(Qt.green if pnl > 0 else Qt.red)
                self.trades_table.setItem(i, 4, QTableWidgetItem(pnl_item))
                
                # P&L percentage
                pnl_pct = trade.get('pnl_percentage', 0)
                pnl_pct_item = QTableWidgetItem(f"{pnl_pct:.2f}%")
                pnl_pct_item.setForeground(Qt.green if pnl_pct > 0 else Qt.red)
                self.trades_table.setItem(i, 5, QTableWidgetItem(pnl_pct_item))

class BacktestDialog(QDialog):
    """Backtest settings dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.backtest_thread = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Strategy Backtest")
        self.setMinimumSize(500, 600)
        
        layout = QVBoxLayout()
        
        # Basic settings group
        basic_group = QGroupBox("Basic Settings")
        basic_layout = QFormLayout()
        
        # Trading symbol
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(list(ALL_SYMBOL_MAP.keys()))
        basic_layout.addRow("Trading Symbol:", self.symbol_combo)
        
        # Timeframe
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(list(ALL_INTERVAL_MAP.keys()))
        basic_layout.addRow("Timeframe:", self.interval_combo)
        
        # Data source
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(["MT5", "CryptoCompare"])
        basic_layout.addRow("Data Source:", self.data_source_combo)
        
        # Strategy type
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Professional Strategy", "Improved Strategy"])
        basic_layout.addRow("Strategy Type:", self.strategy_combo)
        
        # Compare strategies
        self.compare_checkbox = QCheckBox("Compare Two Strategies")
        self.compare_checkbox.stateChanged.connect(self.on_compare_changed)
        basic_layout.addRow("", self.compare_checkbox)
        
        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)
        
        # Time range group
        time_group = QGroupBox("Time Range")
        time_layout = QFormLayout()
        
        # Start date
        self.start_date_edit = QDateTimeEdit()
        self.start_date_edit.setDateTime(datetime.now() - timedelta(days=90))
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd hh:mm")
        time_layout.addRow("Start Time:", self.start_date_edit)
        
        # End date
        self.end_date_edit = QDateTimeEdit()
        self.end_date_edit.setDateTime(datetime.now())
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd hh:mm")
        time_layout.addRow("End Time:", self.end_date_edit)
        
        time_group.setLayout(time_layout)
        layout.addWidget(time_group)
        
        # Backtest parameters group
        params_group = QGroupBox("Backtest Parameters")
        params_layout = QFormLayout()
        
        # Initial balance
        self.initial_balance_spin = QDoubleSpinBox()
        self.initial_balance_spin.setRange(1000, 1000000)
        self.initial_balance_spin.setValue(10000)
        self.initial_balance_spin.setSuffix(" USD")
        params_layout.addRow("Initial Balance:", self.initial_balance_spin)
        
        # Commission
        self.commission_spin = QDoubleSpinBox()
        self.commission_spin.setRange(0, 0.01)
        self.commission_spin.setValue(0.001)
        self.commission_spin.setDecimals(4)
        self.commission_spin.setSingleStep(0.0001)
        params_layout.addRow("Commission Rate:", self.commission_spin)
        
        # Slippage
        self.slippage_spin = QDoubleSpinBox()
        self.slippage_spin.setRange(0, 0.01)
        self.slippage_spin.setValue(0.0005)
        self.slippage_spin.setDecimals(4)
        self.slippage_spin.setSingleStep(0.0001)
        params_layout.addRow("Slippage:", self.slippage_spin)
        
        # Save report
        self.save_report_checkbox = QCheckBox("Save Report")
        self.save_report_checkbox.setChecked(True)
        params_layout.addRow("", self.save_report_checkbox)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Progress bar and status
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.run_button = QPushButton("Start Backtest")
        self.run_button.clicked.connect(self.run_backtest)
        button_layout.addWidget(self.run_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def on_compare_changed(self, state):
        """Compare strategies checkbox state changed"""
        if state == Qt.Checked:
            self.strategy_combo.setEnabled(False)
        else:
            self.strategy_combo.setEnabled(True)
    
    def run_backtest(self):
        """Run backtest"""
        # Get parameters
        symbol_display = self.symbol_combo.currentText()
        symbol = ALL_SYMBOL_MAP[symbol_display]
        
        interval_display = self.interval_combo.currentText()
        interval = ALL_INTERVAL_MAP[interval_display]
        
        data_source = self.data_source_combo.currentText()
        
        strategy_type = "professional" if self.strategy_combo.currentText() == "Professional Strategy" else "improved"
        
        start_date = self.start_date_edit.dateTime().toPython()
        end_date = self.end_date_edit.dateTime().toPython()
        
        initial_balance = self.initial_balance_spin.value()
        commission = self.commission_spin.value()
        slippage = self.slippage_spin.value()
        save_report = self.save_report_checkbox.isChecked()
        compare = self.compare_checkbox.isChecked()
        
        # Create parameters dictionary
        params = {
            'symbol': symbol,
            'interval': interval_display,
            'data_source': data_source,
            'strategy_type': strategy_type,
            'start_date': start_date,
            'end_date': end_date,
            'initial_balance': initial_balance,
            'commission': commission,
            'slippage': slippage,
            'save_report': save_report,
            'compare': compare
        }
        
        # Disable UI
        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Preparing backtest...")
        
        # Create and start backtest thread
        self.backtest_thread = BacktestThread(params)
        self.backtest_thread.progress_updated.connect(self.progress_bar.setValue)
        self.backtest_thread.status_updated.connect(self.status_label.setText)
        self.backtest_thread.finished.connect(self.on_backtest_finished)
        self.backtest_thread.error.connect(self.on_backtest_error)
        self.backtest_thread.start()
    
    def on_backtest_finished(self, result):
        """Backtest finished"""
        # Restore UI
        self.run_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Backtest completed")
        
        # Show results
        dialog = BacktestResultsDialog(result, self)
        dialog.exec_()
    
    def on_backtest_error(self, error_msg):
        """Backtest error"""
        # Restore UI
        self.run_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Backtest error: {error_msg}")