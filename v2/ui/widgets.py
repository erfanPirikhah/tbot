# ui/widgets.py

import logging
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QPushButton, QGroupBox, QTextEdit,
                             QTableWidget, QTableWidgetItem, QProgressBar,
                             QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont

from utils.font_manager import FontManager

logger = logging.getLogger(__name__)

class MarketConditionsWidget(QWidget):
    """ویجت نمایش شرایط بازار"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        group = QGroupBox("🌡️ شرایط بازار")
        group.setLayoutDirection(Qt.RightToLeft)
        grid_layout = QGridLayout()
        grid_layout.setVerticalSpacing(8)
        grid_layout.setHorizontalSpacing(15)
        
        self.conditions = {
            'current_price': self.create_condition_label(),
            'trend': self.create_condition_label(),
            'volatility': self.create_condition_label(),
            'momentum': self.create_condition_label(),
            'rsi': self.create_condition_label(),
            'signal_strength': self.create_condition_label(),
            'volume_trend': self.create_condition_label(),
            'buy_score': self.create_condition_label()
        }
        
        conditions_config = [
            ("💰 قیمت فعلی:", 'current_price'),
            ("📈 روند:", 'trend'),
            ("🌊 نوسانات:", 'volatility'),
            ("🚀 مومنتوم:", 'momentum'),
            ("📊 RSI:", 'rsi'),
            ("💪 قدرت سیگنال:", 'signal_strength'),
            ("📦 حجم معاملات:", 'volume_trend'),
            ("🎯 امتیاز خرید:", 'buy_score')
        ]
        
        for i, (label_text, condition_key) in enumerate(conditions_config):
            label = QLabel(label_text)
            label.setFont(FontManager.get_font(size=9))
            label.setAlignment(Qt.AlignRight)
            grid_layout.addWidget(label, i, 1)
            
            value_label = self.conditions[condition_key]
            grid_layout.addWidget(value_label, i, 0)
        
        group.setLayout(grid_layout)
        layout.addWidget(group)
    
    def create_condition_label(self):
        """ایجاد برچسب شرایط"""
        label = QLabel("--")
        label.setFont(FontManager.get_font(size=9))
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(20)
        label.setStyleSheet("""
            QLabel {
                background-color: rgba(255,255,255,0.05);
                border-radius: 4px;
                padding: 2px 6px;
            }
        """)
        return label
    
    def update_conditions(self, signal_info, current_price=None):
        """به روزرسانی شرایط بازار"""
        # قیمت لحظه‌ای
        if current_price:
            self.conditions['current_price'].setText(f"{current_price:,.2f} $")
            self.conditions['current_price'].setStyleSheet("""
                QLabel {
                    color: #2196F3;
                    font-weight: bold;
                    background-color: rgba(33, 150, 243, 0.2);
                    border-radius: 4px;
                    padding: 2px 6px;
                }
            """)
        
        market_conditions = signal_info.get('market_conditions', {})
        
        # روند
        trend = market_conditions.get('trend', 'نامشخص')
        trend_config = {
            "STRONG_UPTREND": ("صعودی قوی", "#00C853"),
            "UPTREND": ("صعودی", "#4CAF50"),
            "SIDEWAYS": ("خنثی", "#FF9800"),
            "DOWNTREND": ("نزولی", "#F44336"),
            "STRONG_DOWNTREND": ("نزولی قوی", "#D50000")
        }
        trend_text, trend_color = trend_config.get(trend, (trend, "#9E9E9E"))
        self.conditions['trend'].setText(trend_text)
        self.conditions['trend'].setStyleSheet(f"""
            QLabel {{
                color: {trend_color};
                font-weight: bold;
                background-color: rgba{trend_color[1:]}, 0.2;
                border-radius: 4px;
                padding: 2px 6px;
            }}
        """)
        
        # نوسانات
        volatility = market_conditions.get('volatility', 'نامشخص')
        vol_config = {
            "HIGH": ("بالا", "#F44336"),
            "MEDIUM": ("متوسط", "#FF9800"),
            "LOW": ("پایین", "#4CAF50")
        }
        vol_text, vol_color = vol_config.get(volatility, (volatility, "#9E9E9E"))
        self.conditions['volatility'].setText(vol_text)
        self.conditions['volatility'].setStyleSheet(f"color: {vol_color}; font-weight: bold;")
        
        # مومنتوم
        momentum = market_conditions.get('momentum', 'نامشخص')
        momentum_config = {
            "STRONG_BULLISH": ("صعودی قوی", "#00C853"),
            "BULLISH": ("صعودی", "#4CAF50"),
            "NEUTRAL": ("خنثی", "#FF9800"),
            "BEARISH": ("نزولی", "#F44336"),
            "STRONG_BEARISH": ("نزولی قوی", "#D50000")
        }
        momentum_text, momentum_color = momentum_config.get(momentum, (momentum, "#9E9E9E"))
        self.conditions['momentum'].setText(momentum_text)
        self.conditions['momentum'].setStyleSheet(f"color: {momentum_color}; font-weight: bold;")
        
        # RSI
        rsi = signal_info.get('rsi', 0)
        rsi_color = "#F44336" if rsi > 70 else "#4CAF50" if rsi < 30 else "#FF9800"
        self.conditions['rsi'].setText(f"{rsi:.1f}")
        self.conditions['rsi'].setStyleSheet(f"""
            QLabel {{
                color: {rsi_color};
                font-weight: bold;
                background-color: rgba{rsi_color[1:]}, 0.2;
                border-radius: 4px;
                padding: 2px 6px;
            }}
        """)
        
        # قدرت سیگنال
        strength = signal_info.get('signal_strength', 'نامشخص')
        strength_config = {
            "VERY_STRONG": ("بسیار قوی", "#00FF00"),
            "STRONG": ("قوی", "#4CAF50"),
            "MEDIUM": ("متوسط", "#FF9800"),
            "WEAK": ("ضعیف", "#F44336"),
            "NEUTRAL": ("خنثی", "#9E9E9E")
        }
        strength_text, strength_color = strength_config.get(strength, (strength, "#9E9E9E"))
        self.conditions['signal_strength'].setText(strength_text)
        self.conditions['signal_strength'].setStyleSheet(f"color: {strength_color}; font-weight: bold;")
        
        # حجم معاملات
        volume_trend = market_conditions.get('volume_trend', 'نامشخص')
        volume_config = {
            "HIGH": ("بالا", "#4CAF50"),
            "ABOVE_AVERAGE": ("بالاتر از میانگین", "#8BC34A"),
            "NEUTRAL": ("معمولی", "#FF9800"),
            "LOW": ("پایین", "#F44336")
        }
        volume_text, volume_color = volume_config.get(volume_trend, (volume_trend, "#9E9E9E"))
        self.conditions['volume_trend'].setText(volume_text)
        self.conditions['volume_trend'].setStyleSheet(f"color: {volume_color};")
        
        # امتیاز خرید
        buy_score = signal_info.get('buy_score', 0)
        score_color = "#4CAF50" if buy_score >= 7 else "#FF9800" if buy_score >= 5 else "#F44336"
        self.conditions['buy_score'].setText(str(buy_score))
        self.conditions['buy_score'].setStyleSheet(f"""
            QLabel {{
                color: {score_color};
                font-weight: bold;
                background-color: rgba{score_color[1:]}, 0.2;
                border-radius: 4px;
                padding: 2px 6px;
            }}
        """)

class PerformanceWidget(QWidget):
    """ویجت نمایش عملکرد معاملاتی"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        group = QGroupBox("📊 عملکرد معاملاتی")
        group.setLayoutDirection(Qt.RightToLeft)
        grid_layout = QGridLayout()
        grid_layout.setVerticalSpacing(8)
        grid_layout.setHorizontalSpacing(15)
        
        # معیارهای عملکرد
        self.metrics = {
            'total_trades': self.create_metric_label(),
            'win_rate': self.create_metric_label(),
            'total_pnl': self.create_metric_label(),
            'current_position': self.create_metric_label(),
            'portfolio_value': self.create_metric_label(),
            'profit_factor': self.create_metric_label(),
            'portfolio_return': self.create_metric_label(),
            'best_trade': self.create_metric_label(),
            'worst_trade': self.create_metric_label(),
            'avg_trade': self.create_metric_label()
        }
        
        metrics_config = [
            ("تعداد معاملات:", 'total_trades', "#2196F3"),
            ("نرخ برد:", 'win_rate', "#4CAF50"),
            ("سود/زیان کل:", 'total_pnl', "#FF9800"),
            ("موقعیت فعلی:", 'current_position', "#9C27B0"),
            ("ارزش پورتفو:", 'portfolio_value', "#009688"),
            ("فاکتور سود:", 'profit_factor', "#3F51B5"),
            ("بازده پورتفو:", 'portfolio_return', "#00BCD4"),
            ("بهترین معامله:", 'best_trade', "#4CAF50"),
            ("بدترین معامله:", 'worst_trade', "#F44336"),
            ("میانگین معامله:", 'avg_trade', "#FF5722")
        ]
        
        for i, (label_text, metric_key, color) in enumerate(metrics_config):
            # برچسب
            label = QLabel(label_text)
            label.setFont(FontManager.get_font(size=9))
            label.setAlignment(Qt.AlignRight)
            label.setStyleSheet(f"color: {color};")
            grid_layout.addWidget(label, i, 1)
            
            # مقدار
            value_label = self.metrics[metric_key]
            value_label.setStyleSheet(f"""
                QLabel {{
                    color: {color};
                    font-weight: bold;
                    background-color: rgba{color[1:]}, 0.1;
                    border-radius: 4px;
                    padding: 2px 6px;
                }}
            """)
            grid_layout.addWidget(value_label, i, 0)
        
        group.setLayout(grid_layout)
        layout.addWidget(group)
    
    def create_metric_label(self):
        """ایجاد برچسب متریک"""
        label = QLabel("0")
        label.setFont(FontManager.get_font(size=9, bold=True))
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(20)
        return label
    
    def update_metrics(self, metrics_dict):
        """به روزرسانی معیارهای عملکرد"""
        if not metrics_dict:
            return
        
        # مقادیر پایه
        self.metrics['total_trades'].setText(str(metrics_dict.get('total_trades', 0)))
        self.metrics['win_rate'].setText(f"{metrics_dict.get('win_rate', 0):.1f}%")
        
        # سود/زیان
        pnl = metrics_dict.get('total_pnl', 0)
        pnl_color = "#4CAF50" if pnl >= 0 else "#F44336"
        self.metrics['total_pnl'].setText(f"{pnl:+.2f} $")
        self.metrics['total_pnl'].setStyleSheet(self.metrics['total_pnl'].styleSheet().replace("#FF9800", pnl_color))
        
        # موقعیت فعلی
        position = metrics_dict.get('current_position', 'OUT')
        position_color = "#4CAF50" if position == "LONG" else "#F44336" if position == "SHORT" else "#9C27B0"
        self.metrics['current_position'].setText(position)
        self.metrics['current_position'].setStyleSheet(self.metrics['current_position'].styleSheet().replace("#9C27B0", position_color))
        
        # مقادیر عددی
        self.metrics['portfolio_value'].setText(f"{metrics_dict.get('current_portfolio_value', 10000):.2f} $")
        
        profit_factor = metrics_dict.get('profit_factor', 0)
        pf_color = "#4CAF50" if profit_factor > 1.5 else "#FF9800" if profit_factor > 1 else "#F44336"
        self.metrics['profit_factor'].setText(f"{profit_factor:.2f}")
        
        portfolio_return = metrics_dict.get('portfolio_return', 0)
        ret_color = "#4CAF50" if portfolio_return >= 0 else "#F44336"
        self.metrics['portfolio_return'].setText(f"{portfolio_return:+.1f}%")
        
        self.metrics['best_trade'].setText(f"{metrics_dict.get('best_trade', 0):.2f} $")
        self.metrics['worst_trade'].setText(f"{metrics_dict.get('worst_trade', 0):.2f} $")
        self.metrics['avg_trade'].setText(f"{metrics_dict.get('average_trade_pnl', 0):.2f} $")

class StrategySettingsWidget(QWidget):
    """ویجت تنظیمات استراتژی"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # گروه تنظیمات RSI
        rsi_group = QGroupBox("📊 تنظیمات RSI")
        rsi_group.setLayoutDirection(Qt.RightToLeft)
        rsi_layout = QGridLayout()
        
        rsi_layout.addWidget(QLabel("دوره RSI:"), 0, 1)
        self.rsi_period = QSpinBox()
        self.rsi_period.setRange(5, 30)
        self.rsi_period.setValue(14)
        rsi_layout.addWidget(self.rsi_period, 0, 0)
        
        rsi_layout.addWidget(QLabel("سطح اشباع خرید:"), 1, 1)
        self.overbought = QSpinBox()
        self.overbought.setRange(60, 85)
        self.overbought.setValue(70)
        rsi_layout.addWidget(self.overbought, 1, 0)
        
        rsi_layout.addWidget(QLabel("سطح اشباع فروش:"), 2, 1)
        self.oversold = QSpinBox()
        self.oversold.setRange(15, 40)
        self.oversold.setValue(30)
        rsi_layout.addWidget(self.oversold, 2, 0)
        
        rsi_group.setLayout(rsi_layout)
        layout.addWidget(rsi_group)
        
        # گروه مدیریت ریسک
        risk_group = QGroupBox("🛡️ مدیریت ریسک")
        risk_group.setLayoutDirection(Qt.RightToLeft)
        risk_layout = QGridLayout()
        
        risk_layout.addWidget(QLabel("ریسک هر معامله (%):"), 0, 1)
        self.risk_per_trade = QDoubleSpinBox()
        self.risk_per_trade.setRange(0.1, 10.0)
        self.risk_per_trade.setValue(2.0)
        self.risk_per_trade.setDecimals(1)
        risk_layout.addWidget(self.risk_per_trade, 0, 0)
        
        risk_layout.addWidget(QLabel("نسبت سود به زیان:"), 1, 1)
        self.rr_ratio = QDoubleSpinBox()
        self.rr_ratio.setRange(1.0, 5.0)
        self.rr_ratio.setValue(2.5)
        self.rr_ratio.setDecimals(1)
        risk_layout.addWidget(self.rr_ratio, 1, 0)
        
        risk_layout.addWidget(QLabel("ضریب استاپ لاس:"), 2, 1)
        self.stop_loss_multiplier = QDoubleSpinBox()
        self.stop_loss_multiplier.setRange(0.5, 3.0)
        self.stop_loss_multiplier.setValue(1.5)
        self.stop_loss_multiplier.setDecimals(1)
        risk_layout.addWidget(self.stop_loss_multiplier, 2, 0)
        
        risk_group.setLayout(risk_layout)
        layout.addWidget(risk_group)
        
        # گروه تنظیمات پیشرفته
        advanced_group = QGroupBox("⚙️ تنظیمات پیشرفته")
        advanced_group.setLayoutDirection(Qt.RightToLeft)
        advanced_layout = QGridLayout()
        
        self.use_trailing_stop = QCheckBox("استفاده از Trailing Stop")
        self.use_trailing_stop.setChecked(True)
        advanced_layout.addWidget(self.use_trailing_stop, 0, 0, 1, 2)
        
        advanced_layout.addWidget(QLabel("حداکثر زمان معامله (ساعت):"), 1, 1)
        self.max_trade_duration = QSpinBox()
        self.max_trade_duration.setRange(1, 168)
        self.max_trade_duration.setValue(72)
        advanced_layout.addWidget(self.max_trade_duration, 1, 0)
        
        advanced_layout.addWidget(QLabel("دوره تشخیص واگرایی:"), 2, 1)
        self.divergence_lookback = QSpinBox()
        self.divergence_lookback.setRange(5, 30)
        self.divergence_lookback.setValue(14)
        advanced_layout.addWidget(self.divergence_lookback, 2, 0)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        # دکمه‌ها
        button_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton("💾 اعمال تنظیمات")
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        self.reset_btn = QPushButton("🔄 بازنشانی")
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #e68900;
            }
        """)
        
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.reset_btn)
        layout.addLayout(button_layout)
        
        layout.addStretch()

class ModernProgressBar(QWidget):
    """نوار پیشرفت مدرن"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(8)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("آماده برای تحلیل...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self.status_label)
        
    def set_value(self, value, text=""):
        self.progress_bar.setValue(value)
        if text:
            self.status_label.setText(text)

class RightAlignedTextEdit(QTextEdit):
    """ویرایشگر متن راست‌چین شده"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_text_edit()
    
    def setup_text_edit(self):
        """تنظیمات ویرایشگر متن"""
        self.setLayoutDirection(Qt.RightToLeft)
        self.setAlignment(Qt.AlignRight)

class RightAlignedTableWidget(QTableWidget):
    """جدول راست‌چین شده پیشرفته"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_table()
    
    def setup_table(self):
        """تنظیمات پیشرفته جدول"""
        self.setLayoutDirection(Qt.RightToLeft)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setEditTriggers(QTableWidget.NoEditTriggers)