# ui/main_window.py

import logging
from datetime import datetime
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QGridLayout, QLabel, QPushButton, QComboBox, 
                             QTabWidget, QStatusBar, QSplitter, QMessageBox,
                             QMenuBar, QAction, QMenu, QTextEdit, QTableWidget,
                             QHeaderView)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

from ui.dialogs import ApiKeyDialog, MT5SettingsDialog
from ui.widgets import (MarketConditionsWidget, PerformanceWidget, 
                       StrategySettingsWidget, ModernProgressBar,
                       RightAlignedTextEdit, RightAlignedTableWidget)
from controllers.analysis_controller import AnalysisController
from controllers.data_controller import DataController
from utils.font_manager import FontManager
from config import MT5_SYMBOL_MAP, CRYPTOCOMPARE_SYMBOL_MAP, MT5_INTERVAL_MAP, CRYPTOCOMPARE_INTERVAL_MAP

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """پنجره اصلی برنامه"""
    
    def __init__(self, config_manager):
        super().__init__()
        self.config = config_manager
        self.analysis_controller = AnalysisController()
        self.data_controller = DataController()
        
        self.init_ui()
        self.init_data()
        self.connect_signals()
        
    def init_ui(self):
        """راه‌اندازی رابط کاربری"""
        self.setWindowTitle("TradeBot Pro - نرم افزار تحلیل پیشرفته بازار ارزهای دیجیتال و فارکس")
        self.setGeometry(100, 50, 1600, 1000)
        self.setLayoutDirection(Qt.RightToLeft)
        
        self.setup_central_widget()
        self.setup_status_bar()
        self.setup_menus()
        self.apply_styles()
        
        self.status_bar.showMessage("🎉 نرم افزار TradeBot Pro آماده به کار است - توسعه داده شده توسط تیم تحلیل بازار")
        
    def init_data(self):
        """راه‌اندازی داده‌ها"""
        self.df = None
        self.analysis_count = 0
        self.current_price = 0.0
        self.auto_update_enabled = False
        
        self.setup_timers()
        self.check_mt5_status()
        
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
        toolbar.setFixedHeight(80)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 5, 10, 5)
        
        # عنوان و لوگو
        title = QLabel("💎 TradeBot Pro - نسخه چندمنبعی")
        title.setFont(FontManager.get_font(size=16, bold=True))
        title.setStyleSheet("color: #2196F3;")
        
        # کنترل‌های سریع
        quick_controls = QWidget()
        quick_layout = QGridLayout(quick_controls)
        quick_layout.setVerticalSpacing(5)
        
        # ردیف اول: انتخاب منبع داده
        quick_layout.addWidget(QLabel("منبع داده:"), 0, 0)
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(["MetaTrader5", "CryptoCompare"])
        self.data_source_combo.setCurrentText("MetaTrader5")
        self.data_source_combo.setMinimumWidth(120)
        quick_layout.addWidget(self.data_source_combo, 0, 1)
        
        # ردیف دوم: نماد و تایم‌فریم
        quick_layout.addWidget(QLabel("نماد:"), 1, 0)
        self.symbol_combo = QComboBox()
        self.update_symbols_list()
        self.symbol_combo.setMinimumWidth(150)
        quick_layout.addWidget(self.symbol_combo, 1, 1)
        
        quick_layout.addWidget(QLabel("تایم‌فریم:"), 1, 2)
        self.interval_combo = QComboBox()
        self.update_intervals_list()
        quick_layout.addWidget(self.interval_combo, 1, 3)
        
        # دکمه‌های عمل
        self.analyze_btn = QPushButton("🚀 تحلیل بازار")
        self.analyze_btn.setMinimumHeight(35)
        quick_layout.addWidget(self.analyze_btn, 0, 4, 2, 1)
        
        self.chart_btn = QPushButton("📊 نمایش نمودار")
        self.chart_btn.setMinimumHeight(35)
        self.chart_btn.setEnabled(False)
        quick_layout.addWidget(self.chart_btn, 0, 5, 2, 1)
        
        toolbar_layout.addWidget(title)
        toolbar_layout.addStretch()
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
        self.mt5_settings_btn = QPushButton("🖥️ تنظیمات MT5")
        self.settings_btn = QPushButton("⚙️ تنظیمات پیشرفته")
        self.help_btn = QPushButton("❓ راهنما")
        
        control_layout.addWidget(self.auto_update_btn, 0, 0)
        control_layout.addWidget(self.api_key_btn, 0, 1)
        control_layout.addWidget(self.mt5_settings_btn, 1, 0)
        control_layout.addWidget(self.settings_btn, 1, 1)
        control_layout.addWidget(self.help_btn, 2, 0, 1, 2)
        
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
        
        # پیام راه‌اندازی
        from data.mt5_data import MT5_AVAILABLE, mt5_fetcher
        mt5_status = "✅ فعال" if MT5_AVAILABLE and mt5_fetcher and mt5_fetcher.connected else "❌ غیرفعال"
        startup_info = f"""
📋 گزارش فعالیت TradeBot Pro نسخه ۳.۰.۰
{"="*60}
🚀 برنامه در تاریخ {datetime.now().strftime('%Y/%m/%d %H:%M')} راه‌اندازی شد
✅ سیستم آماده به کار است

💽 منابع داده:
  • MetaTrader5: {mt5_status}
  • CryptoCompare: ✅ فعال

📊 نمادهای پشتیبانی شده:
  • {len(MT5_SYMBOL_MAP)} نماد فارکس و طلا
  • {len(CRYPTOCOMPARE_SYMBOL_MAP)} نماد ارز دیجیتال

{"="*60}
"""
        self.log_text.setPlainText(startup_info)
        
        layout.addWidget(self.log_text)
        return widget
        
    def setup_status_bar(self):
        """تنظیم نوار وضعیت"""
        self.status_bar = QStatusBar()
        self.status_bar.setLayoutDirection(Qt.RightToLeft)
        self.setStatusBar(self.status_bar)
        
        # وضعیت MT5
        from data.mt5_data import MT5_AVAILABLE, mt5_fetcher
        if MT5_AVAILABLE and mt5_fetcher and mt5_fetcher.connected:
            mt5_status = "✅ MT5"
            mt5_style = "color: #4CAF50;"
        else:
            mt5_status = "❌ MT5"
            mt5_style = "color: #F44336;"
        
        self.mt5_status_label = QLabel(mt5_status)
        self.mt5_status_label.setFont(FontManager.get_font(size=9))
        self.mt5_status_label.setStyleSheet(mt5_style)
        self.status_bar.addWidget(self.mt5_status_label)
        
        # برچسب وضعیت
        self.status_label = QLabel("آماده به کار")
        self.status_label.setFont(FontManager.get_font(size=9))
        self.status_bar.addWidget(self.status_label)
        
        # اطلاعات سیستم
        self.system_info = QLabel(f"ورژن ۳.۰.۰ | توسعه داده شده توسط تیم تحلیل بازار")
        self.system_info.setFont(FontManager.get_font(size=8))
        self.system_info.setStyleSheet("color: #666;")
        self.status_bar.addPermanentWidget(self.system_info)
        
    def setup_menus(self):
        """تنظیم منوها"""
        menubar = self.menuBar()
        menubar.setLayoutDirection(Qt.RightToLeft)
        
        # منوی فایل
        file_menu = menubar.addMenu("📁 فایل")
        
        self.exit_action = QAction("خروج", self)
        self.exit_action.setShortcut("Ctrl+Q")
        file_menu.addAction(self.exit_action)
        
        # منوی تحلیل
        analysis_menu = menubar.addMenu("📈 تحلیل")
        
        self.analyze_action = QAction("تحلیل بازار", self)
        self.analyze_action.setShortcut("F5")
        analysis_menu.addAction(self.analyze_action)
        
        self.chart_action = QAction("نمایش نمودار", self)
        self.chart_action.setShortcut("F6")
        analysis_menu.addAction(self.chart_action)
        
        # منوی تنظیمات
        settings_menu = menubar.addMenu("⚙️ تنظیمات")
        
        self.api_settings_action = QAction("تنظیمات API", self)
        settings_menu.addAction(self.api_settings_action)
        
        self.mt5_settings_action = QAction("تنظیمات MT5", self)
        settings_menu.addAction(self.mt5_settings_action)
        
        self.strategy_settings_action = QAction("تنظیمات استراتژی", self)
        settings_menu.addAction(self.strategy_settings_action)
        
        # منوی راهنما
        help_menu = menubar.addMenu("❓ راهنما")
        
        self.docs_action = QAction("مستندات", self)
        help_menu.addAction(self.docs_action)
        
        self.about_action = QAction("درباره برنامه", self)
        help_menu.addAction(self.about_action)
        
    def setup_timers(self):
        """تنظیم تایمرهای خودکار"""
        self.auto_update_timer = QTimer()
        self.auto_update_timer.timeout.connect(self.analyze_market)
        self.auto_update_interval = 300000  # 5 دقیقه
        
    def connect_signals(self):
        """اتصال سیگنال‌ها"""
        # دکمه‌های اصلی
        self.analyze_btn.clicked.connect(self.analyze_market)
        self.chart_btn.clicked.connect(self.show_chart)
        self.api_key_btn.clicked.connect(self.show_api_key_dialog)
        self.mt5_settings_btn.clicked.connect(self.show_mt5_settings_dialog)
        self.settings_btn.clicked.connect(self.show_settings_dialog)
        self.help_btn.clicked.connect(self.show_help)
        self.auto_update_btn.clicked.connect(self.toggle_auto_update)
        
        # منوها
        self.exit_action.triggered.connect(self.close)
        self.analyze_action.triggered.connect(self.analyze_market)
        self.chart_action.triggered.connect(self.show_chart)
        self.mt5_settings_action.triggered.connect(self.show_mt5_settings_dialog)
        self.api_settings_action.triggered.connect(self.show_api_key_dialog)
        self.strategy_settings_action.triggered.connect(self.show_settings_dialog)
        
        # کنترل‌های داده
        self.data_source_combo.currentTextChanged.connect(self.on_data_source_changed)
        
        # تنظیمات استراتژی
        self.settings_tab.apply_btn.clicked.connect(self.apply_strategy_settings)
        self.settings_tab.reset_btn.clicked.connect(self.reset_strategy_settings)
        
    def update_symbols_list(self):
        """به‌روزرسانی لیست نمادها بر اساس منبع داده انتخاب شده"""
        self.symbol_combo.clear()
        
        data_source = self.data_source_combo.currentText()
        if data_source == "MetaTrader5":
            symbols = list(MT5_SYMBOL_MAP.keys())
            default_symbol = "طلا (XAUUSD)"
        else:
            symbols = list(CRYPTOCOMPARE_SYMBOL_MAP.keys())
            default_symbol = "بیت‌کوین (BTC)"
        
        self.symbol_combo.addItems(symbols)
        self.symbol_combo.setCurrentText(default_symbol)
    
    def update_intervals_list(self):
        """به‌روزرسانی لیست تایم‌فریم‌ها بر اساس منبع داده"""
        self.interval_combo.clear()
        
        data_source = self.data_source_combo.currentText()
        if data_source == "MetaTrader5":
            intervals = list(MT5_INTERVAL_MAP.keys())
        else:
            intervals = list(CRYPTOCOMPARE_INTERVAL_MAP.keys())
        
        self.interval_combo.addItems(intervals)
        self.interval_combo.setCurrentText("۱ ساعت")
    
    def on_data_source_changed(self):
        """هنگام تغییر منبع داده"""
        self.update_symbols_list()
        self.update_intervals_list()
        self.log_message(f"🔁 تغییر منبع داده به: {self.data_source_combo.currentText()}")
        
    def analyze_market(self):
        """تحلیل بازار - نسخه تصحیح شده"""
        try:
            # دریافت تنظیمات
            data_source = self.data_source_combo.currentText()
            symbol_display = self.symbol_combo.currentText()
            interval_display = self.interval_combo.currentText()
            
            # به روزرسانی وضعیت
            self.analyze_btn.setEnabled(False)
            self.progress.set_value(0, f"📡 در حال اتصال به {data_source}...")
            
            # دریافت کد نماد
            symbol_code = self.get_symbol_code(symbol_display, data_source)
            logger.info(f"🔍 تحلیل برای: {symbol_display} -> {symbol_code} از {data_source}")
            
            if not symbol_code:
                raise ValueError(f"کد نماد برای {symbol_display} یافت نشد")
            
            # تحلیل بازار
            self.progress.set_value(30, "📊 در حال تحلیل داده‌ها...")
            signal_info = self.analysis_controller.analyze_market(
                data_source, symbol_display, interval_display
            )
            
            # دریافت قیمت لحظه‌ای
            self.progress.set_value(70, "💰 در حال دریافت قیمت لحظه‌ای...")
            self.current_price = self.data_controller.get_current_price(
                symbol_code, 
                data_source
            )
            
            logger.info(f"💰 قیمت دریافتی برای {symbol_code}: {self.current_price}")
            
            # اگر قیمت صفر است، از آخرین قیمت تاریخی استفاده کن
            if self.current_price == 0:
                current_data = self.analysis_controller.get_current_data()
                if current_data is not None and not current_data.empty and 'close' in current_data.columns:
                    self.current_price = current_data['close'].iloc[-1]
                    logger.info(f"🔁 استفاده از قیمت تاریخی: {self.current_price}")
            
            self.analysis_count += 1
            
            # نمایش نتایج
            self.progress.set_value(100, "✅ تحلیل با موفقیت انجام شد")
            self.display_results(signal_info, symbol_display, data_source)
            self.update_widgets(signal_info)
            
            self.chart_btn.setEnabled(True)
            self.log_message(f"✅ تحلیل #{self.analysis_count} برای {symbol_display} انجام شد - قیمت: {self.current_price:.2f}$")
            
        except Exception as e:
            error_msg = f"خطا در تحلیل: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(self, "خطا", error_msg)
            self.log_message(f"❌ {error_msg}")
            self.progress.set_value(0, "❌ تحلیل با خطا مواجه شد")
        finally:
            self.analyze_btn.setEnabled(True)
            
    def get_symbol_code(self, symbol_display, data_source):
        """دریافت کد نماد از نمایش آن"""
        if data_source == "MetaTrader5":
            return MT5_SYMBOL_MAP.get(symbol_display)
        else:
            return CRYPTOCOMPARE_SYMBOL_MAP.get(symbol_display)
            
    def display_results(self, signal_info, symbol, data_source):
        """نمایش نتایج تحلیل"""
        # پیاده‌سازی مشابه کد اصلی (HTML formatting)
        action = signal_info['action']
        reason = signal_info['reason']
        rsi_val = signal_info.get('rsi', 0)
        
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
                .source {{
                    color: #2196F3;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class='header'>{emoji} {title} {emoji}</div>
            
            <table class='info-table'>
                <tr>
                    <td class='label'>نماد:</td>
                    <td class='value'>
                        <span class='metric'>{symbol}</span>
                        <span class='source'> (از {data_source})</span>
                    </td>
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
        
        performance_metrics = self.analysis_controller.get_performance_metrics()
        self.performance.update_metrics(performance_metrics)
        
        self.update_trades_table()
        
    def update_trades_table(self):
        """به روزرسانی جدول معاملات"""
        trades = self.analysis_controller.get_trade_history()
        self.trades_table.setRowCount(len(trades))
        
        # پیاده‌سازی مشابه کد اصلی برای پر کردن جدول
        
    def show_chart(self):
        """نمایش نمودار"""
        try:
            from utils.plot_chart import plot_price_and_rsi
            current_data = self.analysis_controller.get_current_data()
            if current_data is not None:
                symbol_display = self.symbol_combo.currentText()
                plot_price_and_rsi(current_data, symbol_display)
                self.log_message("📊 نمودار با موفقیت نمایش داده شد")
            else:
                QMessageBox.warning(self, "هشدار", "لطفاً ابتدا تحلیل بازار را انجام دهید")
        except Exception as e:
            error_msg = f"خطا در نمایش نمودار: {str(e)}"
            QMessageBox.critical(self, "خطا", error_msg)
            self.log_message(f"❌ {error_msg}")
            
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
            
    def show_api_key_dialog(self):
        """نمایش دیالوگ تنظیم API"""
        dialog = ApiKeyDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            api_key = dialog.get_api_key()
            if api_key:
                self.config.api_key = api_key
                self.config.save_settings()
                self.log_message("✅ کلید API ذخیره و تنظیم شد")
            else:
                QMessageBox.warning(self, "هشدار", "لطفاً یک کلید API معتبر وارد کنید.")
                
    def show_mt5_settings_dialog(self):
        """نمایش دیالوگ تنظیمات MT5"""
        dialog = MT5SettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # به‌روزرسانی وضعیت MT5 در نوار وضعیت
            from data.mt5_data import MT5_AVAILABLE, mt5_fetcher
            if MT5_AVAILABLE and mt5_fetcher and mt5_fetcher.connected:
                mt5_status = "✅ MT5"
                mt5_style = "color: #4CAF50;"
            else:
                mt5_status = "❌ MT5"
                mt5_style = "color: #F44336;"
            
            self.mt5_status_label.setText(mt5_status)
            self.mt5_status_label.setStyleSheet(mt5_style)
            self.log_message("✅ تنظیمات MT5 به‌روزرسانی شد")
                
    def log_message(self, message):
        """ثبت پیام در لاگ"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        current_log = self.log_text.toPlainText()
        if len(current_log) > 10000:
            current_log = "\n".join(current_log.split("\n")[-200:])
            
        self.log_text.setPlainText(log_entry + current_log)
        
    def show_settings_dialog(self):
        """نمایش دیالوگ تنظیمات"""
        self.tab_widget.setCurrentWidget(self.settings_tab)
        self.log_message("⚙️ باز شدن تب تنظیمات استراتژی")
        
    def show_help(self):
        """نمایش راهنما"""
        help_text = """
        📖 راهنمای TradeBot Pro نسخه ۳.۰.۰
        
        ۱. **انتخاب منبع داده:**
           - MetaTrader5: برای تحلیل طلا، جفت‌ارزها، شاخص‌ها
           - CryptoCompare: برای تحلیل ارزهای دیجیتال
        
        ۲. **تحلیل بازار:**
           - منبع داده، نماد و تایم‌فریم مورد نظر را انتخاب کنید
           - دکمه "تحلیل بازار" را بزنید
           - نتایج در تب "نتایج تحلیل" نمایش داده می‌شود
        
        ۳. **نمایش نمودار:**
           - پس از تحلیل، دکمه "نمایش نمودار" را بزنید
           - نمودار قیمت و RSI نمایش داده می‌شود
        
        ۴. **تنظیمات MT5:**
           - مطمئن شوید MetaTrader5 نصب و اجرا است
           - از منوی تنظیمات، "تنظیمات MT5" را انتخاب کنید
           - دکمه "تست اتصال" را برای بررسی اتصال بزنید
        
        ۵. **تنظیمات API:**
           - به وبسایت cryptocompare.com مراجعه کنید
           - کلید API رایگان دریافت کنید
           - در دیالوگ تنظیمات وارد کنید
        
        ۶. **تنظیمات استراتژی:**
           - در تب "تنظیمات استراتژی" پارامترها را تغییر دهید
           - دکمه "اعمال تنظیمات" را بزنید
        
        ۷. **بروزرسانی خودکار:**
           - دکمه "بروزرسانی خودکار" را فعال کنید
           - برنامه هر ۵ دقیقه به صورت خودکار تحلیل می‌کند
        
        ⚠️ نکته مهم: این نرم‌افزار فقط برای تحلیل است و لطفاً برای تصمیم‌گیری نهایی از منابع دیگر نیز استفاده کنید.
        
        📞 پشتیبانی: در صورت بروز مشکل با تیم توسعه‌دهنده تماس بگیرید.
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
            
            # ذخیره تنظیمات
            for key, value in new_params.items():
                self.config.set_strategy_param(key, value)
            self.config.save_settings()
            
            # به‌روزرسانی استراتژی
            self.analysis_controller.update_strategy_params(new_params)
            
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
                self.config.reset_to_defaults()
                
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
                
                # به‌روزرسانی استراتژی
                self.analysis_controller.update_strategy_params(self.config.get_all_strategy_params())
                
                QMessageBox.information(self, "موفقیت", "تنظیمات به مقادیر پیش‌فرض بازنشانی شد")
                self.log_message("🔄 تنظیمات استراتژی بازنشانی شد")
                
            except Exception as e:
                error_msg = f"خطا در بازنشانی تنظیمات: {str(e)}"
                QMessageBox.critical(self, "خطا", error_msg)
                self.log_message(f"❌ {error_msg}")
                
    def check_mt5_status(self):
        """بررسی وضعیت MT5"""
        from utils.mt5_connection_helper import MT5ConnectionHelper
        mt5_available, mt5_message = MT5ConnectionHelper.check_mt5_requirements()
        
        if not mt5_available:
            self.log_message(f"⚠️ {mt5_message}")
            
            # اگر منبع داده روی MT5 است اما در دسترس نیست، به CryptoCompare تغییر دهید
            if self.data_source_combo.currentText() == "MetaTrader5":
                self.data_source_combo.setCurrentText("CryptoCompare")
                self.log_message("🔁 تغییر خودکار منبع داده به CryptoCompare")
        
        # اگر MT5 نصب است اما متصل نیست
        elif mt5_available and not self.data_controller.check_mt5_connection()[0]:
            self.log_message("⚠️ MT5 نصب است اما متصل نیست. از منوی تنظیمات اتصال را تست کنید.")
            
    def apply_styles(self):
        """اعمال استایل‌های زیبا"""
        from ui.styles import get_main_stylesheet
        self.setStyleSheet(get_main_stylesheet())
        
    def cleanup(self):
        """تمیزکاری منابع"""
        if hasattr(self, 'auto_update_timer'):
            self.auto_update_timer.stop()
        self.analysis_controller.cleanup()
        self.data_controller.cleanup()