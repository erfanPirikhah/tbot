# -*- coding: utf-8 -*-
"""
TradeBot Pro - نرم افزار تحلیل پیشرفته بازار ارزهای دیجیتال
نسخه: ۲.۰.۰
توسعه دهنده: تیم تحلیل بازار
"""

import sys
import logging
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                             QMessageBox, QComboBox, QGroupBox, QTextEdit, 
                             QStatusBar, QProgressBar, QTabWidget, QTableWidget,
                             QTableWidgetItem, QHeaderView, QSplitter, QLineEdit,
                             QDialog, QDialogButtonBox, QFormLayout, QCheckBox,
                             QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QTimer, QSettings, QSize, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QFontDatabase, QIcon, QPalette
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis, QDateTimeAxis
import pandas as pd
import numpy as np

# ماژول‌های داخلی پروژه
from data.data_fetcher import fetch_market_data, set_cryptocompare_api_key, get_current_price
from indicators.rsi import calculate_rsi
from indicators.moving_averages import calculate_moving_averages
from strategies.improved_advanced_rsi_strategy import ImprovedAdvancedRsiStrategy, PositionType, SignalStrength
from utils.plot_chart import plot_price_and_rsi
from config import (DEFAULT_SYMBOL, DEFAULT_INTERVAL, RSI_PERIOD, 
                   CRYPTOCOMPARE_SYMBOL_MAP, CRYPTOCOMPARE_INTERVAL_MAP,
                   CRYPTOCOMPARE_API_KEY, IMPROVED_STRATEGY_PARAMS)

# تنظیمات پیشرفته لاگینگ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ApiKeyDialog(QDialog):
    """دیالوگ تنظیم کلید API"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔑 تنظیمات کلید API")
        self.setLayoutDirection(Qt.RightToLeft)
        self.setMinimumWidth(500)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QFormLayout(self)
        layout.setLabelAlignment(Qt.AlignRight)
        
        # توضیحات
        description = QLabel("برای دریافت کلید API به وبسایت cryptocompare.com مراجعه کنید")
        description.setWordWrap(True)
        description.setStyleSheet("color: #888; font-size: 11px; padding: 10px;")
        layout.addRow(description)
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("کلید API خود را اینجا وارد کنید...")
        self.api_key_input.setEchoMode(QLineEdit.Normal)
        self.api_key_input.setMinimumHeight(35)
        
        layout.addRow("کلید API CryptoCompare:", self.api_key_input)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        
        # ترجمه دکمه‌ها
        buttons.button(QDialogButtonBox.Ok).setText("تأیید")
        buttons.button(QDialogButtonBox.Cancel).setText("انصراف")
        
        layout.addRow(buttons)
        
    def get_api_key(self):
        return self.api_key_input.text().strip()

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

class FontManager:
    """مدیریت حرفه‌ای فونت‌های فارسی"""
    
    PERSIAN_FONTS = [
        "Vazir", "B Nazanin", "B Mitra", "B Yekan", 
        "Iranian Sans", "Tahoma", "Segoe UI"
    ]
    
    @classmethod
    def setup_application_fonts(cls, app):
        """تنظیم فونت‌های برنامه"""
        font_database = QFontDatabase()
        
        # جستجو برای فونت فارسی
        available_font = "Segoe UI"
        for font_name in cls.PERSIAN_FONTS:
            if font_name in font_database.families():
                available_font = font_name
                logger.info(f"فونت فعال: {available_font}")
                break
        
        # تنظیم فونت پیشفرض
        default_font = QFont(available_font, 10)
        default_font.setStyleStrategy(QFont.PreferAntialias)
        app.setFont(default_font)
        
        return default_font
    
    @classmethod
    def get_font(cls, font_name="Vazir", size=10, bold=False, weight=QFont.Normal):
        """ایجاد فونت با مشخصات دقیق"""
        font = QFont(font_name, size)
        font.setBold(bold)
        font.setWeight(weight)
        font.setStyleStrategy(QFont.PreferAntialias)
        return font

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

class RightAlignedTextEdit(QTextEdit):
    """ویرایشگر متن راست‌چین شده"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_text_edit()
    
    def setup_text_edit(self):
        """تنظیمات ویرایشگر متن"""
        self.setLayoutDirection(Qt.RightToLeft)
        self.setAlignment(Qt.AlignRight)

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

class MainWindow(QMainWindow):
    """پنجره اصلی برنامه"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_data()
        self.connect_signals()
        
    def init_ui(self):
        """راه‌اندازی رابط کاربری"""
        self.setWindowTitle("TradeBot Pro - نرم افزار تحلیل پیشرفته بازار")
        self.setGeometry(100, 50, 1600, 1000)
        self.setLayoutDirection(Qt.RightToLeft)
        
        # تنظیمات مرکزی
        self.setup_central_widget()
        self.setup_status_bar()
        self.setup_menus()
        self.apply_styles()
        
        # نمایش پیام خوشامد
        self.status_bar.showMessage("🎉 نرم افزار TradeBot Pro آماده به کار است - توسعه داده شده توسط تیم تحلیل بازار")
        
    def init_data(self):
        """راه‌اندازی داده‌ها و استراتژی"""
        self.settings = QSettings("TradeBotPro", "v2")
        self.api_key = self.settings.value("api_key", CRYPTOCOMPARE_API_KEY)
        
        # استراتژی
        self.strategy = ImprovedAdvancedRsiStrategy(**IMPROVED_STRATEGY_PARAMS)
        
        self.df = None
        self.analysis_count = 0
        self.current_price = 0.0
        self.auto_update_enabled = False
        
        # تایمرهای خودکار
        self.setup_timers()
        
        # تنظیم API
        self.setup_api()
        
    def connect_signals(self):
        """اتصال سیگنال‌ها به توابع"""
        # دکمه‌های اصلی
        self.analyze_btn.clicked.connect(self.analyze_market)
        self.chart_btn.clicked.connect(self.show_chart)
        self.api_key_btn.clicked.connect(self.show_api_key_dialog)
        self.settings_btn.clicked.connect(self.show_settings_dialog)
        self.help_btn.clicked.connect(self.show_help)
        self.auto_update_btn.clicked.connect(self.toggle_auto_update)
        
        # منوها
        self.exit_action.triggered.connect(self.close)
        self.analyze_action.triggered.connect(self.analyze_market)
        self.chart_action.triggered.connect(self.show_chart)
        
        # تنظیمات استراتژی
        self.settings_tab.apply_btn.clicked.connect(self.apply_strategy_settings)
        self.settings_tab.reset_btn.clicked.connect(self.reset_strategy_settings)        
    def setup_central_widget(self):
        """تنظیم ویجت مرکزی"""
        central_widget = QWidget()
        central_widget.setLayoutDirection(Qt.RightToLeft)
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # نوار ابزار بالا
        self.setup_top_toolbar(main_layout)
        
        # اسپلیتر اصلی
        splitter = QSplitter(Qt.Horizontal)
        splitter.setLayoutDirection(Qt.RightToLeft)
        
        # پنل سمت چپ - اطلاعات و کنترل‌ها
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # پنل سمت راست - نتایج و نمودارها
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([400, 1000])
        main_layout.addWidget(splitter, 1)
        
    def setup_top_toolbar(self, layout):
        """نوار ابزار بالایی"""
        toolbar = QWidget()
        toolbar.setFixedHeight(60)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 5, 10, 5)
        
        # عنوان و لوگو
        title = QLabel("💎 TradeBot Pro")
        title.setFont(FontManager.get_font(size=16, bold=True))
        title.setStyleSheet("color: #2196F3;")
        
        # کنترل‌های سریع
        quick_controls = QWidget()
        quick_layout = QHBoxLayout(quick_controls)
        
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(list(CRYPTOCOMPARE_SYMBOL_MAP.keys()))
        self.symbol_combo.setCurrentText(DEFAULT_SYMBOL)
        self.symbol_combo.setMinimumWidth(150)
        
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(list(CRYPTOCOMPARE_INTERVAL_MAP.keys()))
        self.interval_combo.setCurrentText(DEFAULT_INTERVAL)
        
        self.analyze_btn = QPushButton("🚀 تحلیل بازار")
        self.analyze_btn.setMinimumHeight(35)
        
        self.chart_btn = QPushButton("📊 نمایش نمودار")
        self.chart_btn.setMinimumHeight(35)
        self.chart_btn.setEnabled(False)
        
        quick_layout.addWidget(QLabel("ارز:"))
        quick_layout.addWidget(self.symbol_combo)
        quick_layout.addWidget(QLabel("تایم‌فریم:"))
        quick_layout.addWidget(self.interval_combo)
        quick_layout.addWidget(self.analyze_btn)
        quick_layout.addWidget(self.chart_btn)
        quick_layout.addStretch()
        
        toolbar_layout.addWidget(title)
        toolbar_layout.addWidget(quick_controls)
        
        layout.addWidget(toolbar)
        
    def create_left_panel(self):
        """ایجاد پنل سمت چپ"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # ویجت شرایط بازار
        self.market_conditions = MarketConditionsWidget()
        layout.addWidget(self.market_conditions)
        
        # ویجت عملکرد
        self.performance = PerformanceWidget()
        layout.addWidget(self.performance)
        
        # نوار پیشرفت
        self.progress = ModernProgressBar()
        layout.addWidget(self.progress)
        
        # دکمه‌های کنترل
        control_widget = QWidget()
        control_layout = QGridLayout(control_widget)
        
        self.auto_update_btn = QPushButton("⏰ بروزرسانی خودکار: خاموش")
        self.api_key_btn = QPushButton("🔑 تنظیم API")
        self.settings_btn = QPushButton("⚙️ تنظیمات پیشرفته")
        self.help_btn = QPushButton("❓ راهنما")
        
        control_layout.addWidget(self.auto_update_btn, 0, 0)
        control_layout.addWidget(self.api_key_btn, 0, 1)
        control_layout.addWidget(self.settings_btn, 1, 0)
        control_layout.addWidget(self.help_btn, 1, 1)
        
        layout.addWidget(control_widget)
        layout.addStretch()
        
        return widget
        
    def create_right_panel(self):
        """ایجاد پنل سمت راست"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # تب‌های اصلی
        self.tab_widget = QTabWidget()
        self.tab_widget.setLayoutDirection(Qt.RightToLeft)
        
        # تب نتایج تحلیل
        self.results_tab = self.create_results_tab()
        self.tab_widget.addTab(self.results_tab, "📈 نتایج تحلیل")
        
        # تب تاریخچه معاملات
        self.trades_tab = self.create_trades_tab()
        self.tab_widget.addTab(self.trades_tab, "📋 تاریخچه معاملات")
        
        # تب لاگ
        self.log_tab = self.create_log_tab()
        self.tab_widget.addTab(self.log_tab, "📝 گزارش فعالیت")
        
        # تب تنظیمات
        self.settings_tab = StrategySettingsWidget()
        self.tab_widget.addTab(self.settings_tab, "🔧 تنظیمات استراتژی")
        
        layout.addWidget(self.tab_widget)
        return widget
        
    def create_results_tab(self):
        """ایجاد تب نتایج تحلیل"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.results_text = RightAlignedTextEdit()
        self.results_text.setFont(FontManager.get_font("Consolas", 10))
        self.results_text.setPlainText(
            "🔄 نرم افزار آماده تحلیل است...\n\n"
            "برای شروع تحلیل، دکمه 'تحلیل بازار' را فشار دهید.\n"
            "نتایج تحلیل در این بخش نمایش داده خواهد شد."
        )
        
        layout.addWidget(self.results_text)
        return widget
        
    def create_trades_tab(self):
        """ایجاد تب تاریخچه معاملات"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.trades_table = RightAlignedTableWidget()
        self.trades_table.setColumnCount(8)
        self.trades_table.setHorizontalHeaderLabels([
            "دلیل خروج", "مدت", "سود/زیان $", "سود/زیان %", 
            "حجم", "قیمت خروج", "قیمت ورود", "زمان ورود"
        ])
        
        # تنظیم هدر
        header = self.trades_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        
        layout.addWidget(self.trades_table)
        return widget
        
    def create_log_tab(self):
        """ایجاد تب گزارش فعالیت"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.log_text = RightAlignedTextEdit()
        self.log_text.setFont(FontManager.get_font("Consolas", 9))
        self.log_text.setPlainText(
            "📋 گزارش فعالیت TradeBot Pro\n" +
            "="*50 + "\n" +
            f"🚀 برنامه در تاریخ {datetime.now().strftime('%Y/%m/%d %H:%M')} راه‌اندازی شد\n" +
            "✅ سیستم آماده به کار است\n" +
            "="*50 + "\n"
        )
        
        layout.addWidget(self.log_text)
        return widget
        
    def setup_status_bar(self):
        """تنظیم نوار وضعیت"""
        self.status_bar = QStatusBar()
        self.status_bar.setLayoutDirection(Qt.RightToLeft)
        self.setStatusBar(self.status_bar)
        
        # برچسب وضعیت
        self.status_label = QLabel("آماده به کار")
        self.status_label.setFont(FontManager.get_font(size=9))
        self.status_bar.addWidget(self.status_label)
        
        # اطلاعات سیستم
        self.system_info = QLabel(f"ورژن ۲.۰.۰ | توسعه داده شده توسط تیم تحلیل بازار")
        self.system_info.setFont(FontManager.get_font(size=8))
        self.system_info.setStyleSheet("color: #666;")
        self.status_bar.addPermanentWidget(self.system_info)
        
    def setup_menus(self):
        """تنظیم منوها"""
        menubar = self.menuBar()
        menubar.setLayoutDirection(Qt.RightToLeft)
        
        # منوی فایل
        file_menu = menubar.addMenu("📁 فایل")
        
        self.exit_action = file_menu.addAction("خروج")
        self.exit_action.setShortcut("Ctrl+Q")
        
        # منوی تحلیل
        analysis_menu = menubar.addMenu("📈 تحلیل")
        
        self.analyze_action = analysis_menu.addAction("تحلیل بازار")
        self.analyze_action.setShortcut("F5")
        
        self.chart_action = analysis_menu.addAction("نمایش نمودار")
        self.chart_action.setShortcut("F6")
        
        # منوی تنظیمات
        settings_menu = menubar.addMenu("⚙️ تنظیمات")
        settings_menu.addAction("تنظیمات API")
        settings_menu.addAction("تنظیمات استراتژی")
        
        # منوی راهنما
        help_menu = menubar.addMenu("❓ راهنما")
        help_menu.addAction("مستندات")
        help_menu.addAction("درباره برنامه")
        
    def setup_timers(self):
        """تنظیم تایمرهای خودکار"""
        self.auto_update_timer = QTimer()
        self.auto_update_timer.timeout.connect(self.analyze_market)
        self.auto_update_interval = 300000  # 5 دقیقه
        
    def setup_api(self):
        """تنظیم API"""
        if not self.api_key:
            self.show_api_key_dialog()
        else:
            try:
                set_cryptocompare_api_key(self.api_key)
                self.log_message("✅ کلید API با موفقیت تنظیم شد")
            except Exception as e:
                self.log_message(f"❌ خطا در تنظیم API: {str(e)}")
                self.show_api_key_dialog()
                
    def apply_styles(self):
        """اعمال استایل‌های زیبا"""
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1e1e2e, stop:1 #2d1b69);
                color: #ffffff;
                font-family: Vazir, Tahoma;
            }
            
            QWidget {
                background: transparent;
            }
            
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                border: 2px solid #444;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background: rgba(45, 45, 65, 0.7);
                color: #ffffff;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                right: 10px;
                padding: 0 8px 0 8px;
                color: #ffa500;
                font-size: 11px;
            }
            
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4CAF50, stop:1 #45a049);
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
                min-height: 25px;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #45a049, stop:1 #4CAF50);
            }
            
            QPushButton:pressed {
                background: #367c39;
            }
            
            QPushButton:disabled {
                background: #666;
                color: #999;
            }
            
            QComboBox {
                background: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
                min-height: 20px;
            }
            
            QComboBox:hover {
                border-color: #777;
            }
            
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            
            QComboBox QAbstractItemView {
                background: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555;
                selection-background-color: #4CAF50;
            }
            
            QTabWidget::pane {
                border: 1px solid #444;
                background: rgba(40, 40, 60, 0.9);
            }
            
            QTabBar::tab {
                background: #333;
                color: #ccc;
                padding: 8px 15px;
                margin-left: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            
            QTabBar::tab:selected {
                background: #4CAF50;
                color: white;
                font-weight: bold;
            }
            
            QTabBar::tab:hover:!selected {
                background: #444;
            }
            
            QTextEdit, QTableWidget {
                background: #1a1a1a;
                color: #e0e0e0;
                border: 1px solid #444;
                border-radius: 4px;
                font-family: Consolas, Monospace;
            }
            
            QTableWidget::item {
                padding: 4px;
                border-bottom: 1px solid #333;
            }
            
            QTableWidget::item:selected {
                background: #4CAF50;
                color: black;
            }
            
            QHeaderView::section {
                background: #333;
                color: #fff;
                padding: 6px;
                border: 1px solid #444;
                font-weight: bold;
            }
            
            QStatusBar {
                background: #2b2b2b;
                color: #ccc;
                border-top: 1px solid #444;
            }
            
            QProgressBar {
                border: 1px solid #444;
                border-radius: 3px;
                text-align: center;
                background: #2b2b2b;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4CAF50, stop:1 #45a049);
                border-radius: 2px;
            }
            
            QSplitter::handle {
                background: #444;
                margin: 2px;
            }
            
            QSplitter::handle:hover {
                background: #666;
            }
            
            QCheckBox {
                color: #fff;
                spacing: 5px;
            }
            
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            
            QCheckBox::indicator:unchecked {
                border: 2px solid #666;
                background: #333;
                border-radius: 3px;
            }
            
            QCheckBox::indicator:checked {
                border: 2px solid #4CAF50;
                background: #4CAF50;
                border-radius: 3px;
            }
            
            QSpinBox, QDoubleSpinBox {
                background: #2b2b2b;
                color: #fff;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        
    def show_api_key_dialog(self):
        """نمایش دیالوگ تنظیم API"""
        dialog = ApiKeyDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            api_key = dialog.get_api_key()
            if api_key:
                self.api_key = api_key
                self.settings.setValue("api_key", api_key)
                set_cryptocompare_api_key(api_key)
                self.log_message("✅ کلید API ذخیره و تنظیم شد")
            else:
                QMessageBox.warning(self, "هشدار", "لطفاً یک کلید API معتبر وارد کنید.")
                
    def log_message(self, message):
        """ثبت پیام در لاگ"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        current_log = self.log_text.toPlainText()
        if len(current_log) > 10000:
            current_log = "\n".join(current_log.split("\n")[-200:])
            
        self.log_text.setPlainText(log_entry + current_log)
        
    def analyze_market(self):
        """تحلیل بازار"""
        try:
            # دریافت تنظیمات
            symbol_display = self.symbol_combo.currentText()
            interval_display = self.interval_combo.currentText()
            symbol = CRYPTOCOMPARE_SYMBOL_MAP[symbol_display]
            
            # به روزرسانی وضعیت
            self.analyze_btn.setEnabled(False)
            self.progress.set_value(0, "📡 در حال دریافت داده‌ها از CryptoCompare...")
            
            # دریافت قیمت لحظه‌ای
            self.current_price = get_current_price(symbol)
            
            # دریافت داده‌های تاریخی
            self.progress.set_value(30, "📊 در حال محاسبه اندیکاتورها...")
            raw_data = fetch_market_data(symbol, interval_display)
            
            # محاسبه RSI
            data_with_rsi = calculate_rsi(raw_data, period=RSI_PERIOD)
            
            # تولید سیگنال
            self.progress.set_value(70, "🔍 در حال تحلیل سیگنال...")
            signal_info = self.strategy.generate_signal(data_with_rsi)
            
            self.df = data_with_rsi
            self.analysis_count += 1
            
            # نمایش نتایج
            self.progress.set_value(100, "✅ تحلیل با موفقیت انجام شد")
            self.display_results(signal_info, symbol_display)
            self.update_widgets(signal_info)
            
            self.chart_btn.setEnabled(True)
            self.log_message(f"✅ تحلیل #{self.analysis_count} برای {symbol_display} انجام شد")
            
        except Exception as e:
            error_msg = f"خطا در تحلیل: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(self, "خطا", error_msg)
            self.log_message(f"❌ {error_msg}")
            self.progress.set_value(0, "❌ تحلیل با خطا مواجه شد")
        finally:
            self.analyze_btn.setEnabled(True)
            
    def display_results(self, signal_info, symbol):
        """نمایش نتایج تحلیل"""
        action = signal_info['action']
        reason = signal_info['reason']
        rsi_val = signal_info['rsi']
        
        # رنگ‌بندی بر اساس عمل
        if action == "BUY":
            color = "#4CAF50"
            emoji = "🟢"
            title = "سیگنال خرید"
        elif action == "SELL":
            color = "#F44336" 
            emoji = "🔴"
            title = "سیگنال فروش"
        else:
            color = "#FF9800"
            emoji = "🟡"
            title = "سیگنال انتظار"
            
        html = f"""
        <html dir='rtl'>
        <head>
            <style>
                body {{
                    font-family: Vazir, Tahoma;
                    color: #e0e0e0;
                    line-height: 1.6;
                    margin: 0;
                    padding: 15px;
                }}
                .header {{
                    text-align: center;
                    color: {color};
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 20px;
                    padding: 10px;
                    background: rgba{color[1:]}, 0.1;
                    border-radius: 8px;
                    border: 2px solid {color};
                }}
                .info-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                }}
                .info-table td {{
                    padding: 10px;
                    border-bottom: 1px solid #444;
                    vertical-align: top;
                }}
                .label {{
                    font-weight: bold;
                    text-align: right;
                    color: #aaa;
                    width: 40%;
                }}
                .value {{
                    text-align: left;
                    color: #fff;
                }}
                .metric {{
                    background: rgba(255,255,255,0.05);
                    border-radius: 4px;
                    padding: 3px 8px;
                    margin: 2px;
                }}
            </style>
        </head>
        <body>
            <div class='header'>{emoji} {title} {emoji}</div>
            
            <table class='info-table'>
                <tr>
                    <td class='label'>نماد:</td>
                    <td class='value'><span class='metric'>{symbol}</span></td>
                </tr>
                <tr>
                    <td class='label'>موقعیت:</td>
                    <td class='value'><span class='metric'>{signal_info.get('position', 'OUT')}</span></td>
                </tr>
                <tr>
                    <td class='label'>قیمت فعلی:</td>
                    <td class='value'><span class='metric'>{self.current_price:,.2f} $</span></td>
                </tr>
                <tr>
                    <td class='label'>مقدار RSI:</td>
                    <td class='value'><span class='metric'>{rsi_val:.2f}</span></td>
                </tr>
                <tr>
                    <td class='label'>قدرت سیگنال:</td>
                    <td class='value'><span class='metric'>{signal_info.get('signal_strength', 'NEUTRAL')}</span></td>
                </tr>
        """
        
        if action == "BUY":
            html += f"""
                <tr>
                    <td class='label'>حجم پیشنهادی:</td>
                    <td class='value'><span class='metric'>{signal_info.get('position_size', 0):.4f}</span></td>
                </tr>
                <tr>
                    <td class='label'>حد ضرر:</td>
                    <td class='value'><span class='metric'>{signal_info.get('stop_loss', 0):.2f} $</span></td>
                </tr>
                <tr>
                    <td class='label'>حد سود:</td>
                    <td class='value'><span class='metric'>{signal_info.get('take_profit', 0):.2f} $</span></td>
                </tr>
                <tr>
                    <td class='label'>امتیاز خرید:</td>
                    <td class='value'><span class='metric'>{signal_info.get('buy_score', 0)}</span></td>
                </tr>
            """
        elif action == "SELL":
            pnl = signal_info.get('pnl_percentage', 0)
            pnl_color = "#4CAF50" if pnl >= 0 else "#F44336"
            html += f"""
                <tr>
                    <td class='label'>سود/زیان:</td>
                    <td class='value'><span class='metric' style='color: {pnl_color};'>{pnl:+.2f}%</span></td>
                </tr>
                <tr>
                    <td class='label'>دلیل خروج:</td>
                    <td class='value'><span class='metric'>{signal_info.get('exit_reason', 'N/A')}</span></td>
                </tr>
            """
            
        html += f"""
                <tr>
                    <td class='label'>توضیحات:</td>
                    <td class='value'>{reason}</td>
                </tr>
            </table>
        </body>
        </html>
        """
        
        self.results_text.setHtml(html)
        
    def update_widgets(self, signal_info):
        """به روزرسانی ویجت‌ها"""
        self.market_conditions.update_conditions(signal_info, self.current_price)
        
        performance_metrics = signal_info.get('performance_metrics', {})
        self.performance.update_metrics(performance_metrics)
        
        self.update_trades_table()
        
    def update_trades_table(self):
        """به روزرسانی جدول معاملات"""
        trades = self.strategy.trade_history
        self.trades_table.setRowCount(len(trades))
        
        for i, trade in enumerate(reversed(trades)):
            # زمان ورود
            self.trades_table.setItem(i, 7, QTableWidgetItem(
                trade.entry_time.strftime("%Y/%m/%d %H:%M") if trade.entry_time else "---"
            ))
            
            # قیمت ورود
            self.trades_table.setItem(i, 6, QTableWidgetItem(f"{trade.entry_price:.2f}"))
            
            # قیمت خروج
            exit_price = f"{trade.exit_price:.2f}" if trade.exit_price else "باز"
            self.trades_table.setItem(i, 5, QTableWidgetItem(exit_price))
            
            # حجم
            self.trades_table.setItem(i, 4, QTableWidgetItem(f"{trade.quantity:.4f}"))
            
            # سود/زیان درصدی
            pnl_item = QTableWidgetItem()
            if trade.pnl_percentage is not None:
                pnl_text = f"{trade.pnl_percentage:+.2f}%"
                pnl_item.setText(pnl_text)
                pnl_item.setForeground(QColor("#4CAF50" if trade.pnl_percentage >= 0 else "#F44336"))
            else:
                pnl_item.setText("---")
            self.trades_table.setItem(i, 3, pnl_item)
            
            # سود/زیان دلاری
            pnl_amount_item = QTableWidgetItem()
            if trade.pnl_amount is not None:
                pnl_amount_text = f"{trade.pnl_amount:+.2f}$"
                pnl_amount_item.setText(pnl_amount_text)
                pnl_amount_item.setForeground(QColor("#4CAF50" if trade.pnl_amount >= 0 else "#F44336"))
            else:
                pnl_amount_item.setText("---")
            self.trades_table.setItem(i, 2, pnl_amount_item)
            
            # مدت معامله
            duration_item = QTableWidgetItem()
            if trade.entry_time and trade.exit_time:
                duration_hours = (trade.exit_time - trade.entry_time).total_seconds() / 3600
                duration_item.setText(f"{duration_hours:.1f} ساعت")
            else:
                duration_item.setText("---")
            self.trades_table.setItem(i, 1, duration_item)
            
            # دلیل خروج
            reason_item = QTableWidgetItem()
            if trade.exit_reason:
                reason_text = {
                    "TAKE_PROFIT": "حد سود",
                    "STOP_LOSS": "حد ضرر", 
                    "TRAILING_STOP": "تریلینگ استاپ",
                    "SIGNAL_EXIT": "سیگنال خروج",
                    "TIME_EXIT": "اتمام زمان"
                }.get(trade.exit_reason.value, trade.exit_reason.value)
                reason_item.setText(reason_text)
                
                # رنگ‌بندی
                if trade.exit_reason.value == "TAKE_PROFIT":
                    reason_item.setForeground(QColor("#4CAF50"))
                elif trade.exit_reason.value == "STOP_LOSS":
                    reason_item.setForeground(QColor("#F44336"))
                else:
                    reason_item.setForeground(QColor("#FF9800"))
            else:
                reason_item.setText("باز")
                reason_item.setForeground(QColor("#2196F3"))
                
            self.trades_table.setItem(i, 0, reason_item)
    
    def show_chart(self):
        """نمایش نمودار"""
        if self.df is not None:
            try:
                symbol_display = self.symbol_combo.currentText()
                plot_price_and_rsi(self.df, symbol_display)
                self.log_message("📊 نمودار با موفقیت نمایش داده شد")
            except Exception as e:
                error_msg = f"خطا در نمایش نمودار: {str(e)}"
                QMessageBox.critical(self, "خطا", error_msg)
                self.log_message(f"❌ {error_msg}")
        else:
            QMessageBox.warning(self, "هشدار", "لطفاً ابتدا تحلیل بازار را انجام دهید")
            
    def toggle_auto_update(self):
        """تغییر وضعیت بروزرسانی خودکار"""
        self.auto_update_enabled = not self.auto_update_enabled
        
        if self.auto_update_enabled:
            self.auto_update_timer.start(self.auto_update_interval)
            self.auto_update_btn.setText("⏰ بروزرسانی خودکار: روشن")
            self.auto_update_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                }
            """)
            self.log_message("✅ بروزرسانی خودکار فعال شد")
        else:
            self.auto_update_timer.stop()
            self.auto_update_btn.setText("⏰ بروزرسانی خودکار: خاموش")
            self.auto_update_btn.setStyleSheet("")
            self.log_message("⏸️ بروزرسانی خودکار غیرفعال شد")
            
    def show_settings_dialog(self):
        """نمایش دیالوگ تنظیمات"""
        self.tab_widget.setCurrentWidget(self.settings_tab)
        self.log_message("⚙️ باز شدن تب تنظیمات استراتژی")
        
    def show_help(self):
        """نمایش راهنما"""
        help_text = """
        📖 راهنمای TradeBot Pro
        
        ۱. **تحلیل بازار:**
           - ارز و تایم‌فریم مورد نظر را انتخاب کنید
           - دکمه "تحلیل بازار" را بزنید
           - نتایج در تب "نتایج تحلیل" نمایش داده می‌شود
        
        ۲. **نمایش نمودار:**
           - پس از تحلیل، دکمه "نمایش نمودار" را بزنید
           - نمودار قیمت و RSI نمایش داده می‌شود
        
        ۳. **تنظیمات API:**
           - به وبسایت cryptocompare.com مراجعه کنید
           - کلید API رایگان دریافت کنید
           - در دیالوگ تنظیمات وارد کنید
        
        ۴. **تنظیمات استراتژی:**
           - در تب "تنظیمات استراتژی" پارامترها را تغییر دهید
           - دکمه "اعمال تنظیمات" را بزنید
        
        ۵. **بروزرسانی خودکار:**
           - دکمه "بروزرسانی خودکار" را فعال کنید
           - برنامه هر ۵ دقیقه به صورت خودکار تحلیل می‌کند
        
        ⚠️ نکته مهم: این نرم‌افزار فقط برای تحلیل است و لطفاً برای تصمیم‌گیری نهایی از منابع دیگر نیز استفاده کنید.
        """
        
        QMessageBox.information(self, "راهنما", help_text)
        
    def apply_strategy_settings(self):
        """اعمال تنظیمات استراتژی"""
        try:
            # دریافت مقادیر از ویجت
            new_params = {
                'overbought': self.settings_tab.overbought.value(),
                'oversold': self.settings_tab.oversold.value(),
                'rsi_period': self.settings_tab.rsi_period.value(),
                'risk_per_trade': self.settings_tab.risk_per_trade.value() / 100,
                'stop_loss_atr_multiplier': self.settings_tab.stop_loss_multiplier.value(),
                'take_profit_ratio': self.settings_tab.rr_ratio.value(),
                'use_trailing_stop': self.settings_tab.use_trailing_stop.isChecked(),
                'max_trade_duration': self.settings_tab.max_trade_duration.value(),
                'divergence_lookback': self.settings_tab.divergence_lookback.value()
            }
            
            # ایجاد استراتژی جدید
            self.strategy = ImprovedAdvancedRsiStrategy(**new_params)
            
            QMessageBox.information(self, "موفقیت", "تنظیمات استراتژی با موفقیت اعمال شد")
            self.log_message("✅ تنظیمات استراتژی به‌روزرسانی شد")
            
        except Exception as e:
            error_msg = f"خطا در اعمال تنظیمات: {str(e)}"
            QMessageBox.critical(self, "خطا", error_msg)
            self.log_message(f"❌ {error_msg}")
            
    def reset_strategy_settings(self):
        """بازنشانی تنظیمات استراتژی"""
        reply = QMessageBox.question(
            self, 
            "تأیید", 
            "آیا از بازنشانی تنظیمات اطمینان دارید؟",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                self.strategy = ImprovedAdvancedRsiStrategy(**IMPROVED_STRATEGY_PARAMS)
                
                # بازنشانی مقادیر در ویجت
                self.settings_tab.rsi_period.setValue(14)
                self.settings_tab.overbought.setValue(70)
                self.settings_tab.oversold.setValue(30)
                self.settings_tab.risk_per_trade.setValue(2.0)
                self.settings_tab.rr_ratio.setValue(2.5)
                self.settings_tab.stop_loss_multiplier.setValue(1.5)
                self.settings_tab.use_trailing_stop.setChecked(True)
                self.settings_tab.max_trade_duration.setValue(72)
                self.settings_tab.divergence_lookback.setValue(14)
                
                QMessageBox.information(self, "موفقیت", "تنظیمات به مقادیر پیش‌فرض بازنشانی شد")
                self.log_message("🔄 تنظیمات استراتژی بازنشانی شد")
                
            except Exception as e:
                error_msg = f"خطا در بازنشانی تنظیمات: {str(e)}"
                QMessageBox.critical(self, "خطا", error_msg)
                self.log_message(f"❌ {error_msg}")

def main():
    """تابع اصلی برنامه"""
    app = QApplication(sys.argv)
    app.setApplicationName("TradeBot Pro")
    app.setApplicationVersion("2.0.0")
    
    # تنظیم فونت برنامه
    FontManager.setup_application_fonts(app)
    
    # ایجاد و نمایش پنجره اصلی
    window = MainWindow()
    window.show()
    
    # اجرای برنامه
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()