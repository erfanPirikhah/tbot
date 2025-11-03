# ui/dialogs.py

import logging
from PyQt5.QtWidgets import (QDialog, QFormLayout, QLabel, QLineEdit, 
                             QDialogButtonBox, QPushButton, QMessageBox)
from PyQt5.QtCore import Qt

from data.mt5_data import MT5_AVAILABLE, mt5_fetcher
from utils.mt5_connection_helper import MT5ConnectionHelper

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

class MT5SettingsDialog(QDialog):
    """دیالوگ تنظیمات اتصال به MetaTrader5"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🖥️ تنظیمات اتصال به MetaTrader5")
        self.setLayoutDirection(Qt.RightToLeft)
        self.setMinimumWidth(500)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QFormLayout(self)
        layout.setLabelAlignment(Qt.AlignRight)
        
        # توضیحات
        description = QLabel("برای اتصال به MetaTrader5، لطفاً مطمئن شوید که متاتریدر روی سیستم شما نصب و اجرا است.")
        description.setWordWrap(True)
        description.setStyleSheet("color: #888; font-size: 11px; padding: 10px;")
        layout.addRow(description)
        
        # وضعیت اتصال
        self.connection_status = QLabel("در حال بررسی اتصال...")
        self.connection_status.setStyleSheet("color: #FF9800; font-weight: bold;")
        layout.addRow("وضعیت اتصال:", self.connection_status)
        
        # اطلاعات سرور
        self.server_input = QLineEdit()
        self.server_input.setPlaceholderText("خالی بگذارید برای استفاده از پیشفرض")
        layout.addRow("سرور:", self.server_input)
        
        self.login_input = QLineEdit()
        self.login_input.setPlaceholderText("شماره حساب (اختیاری)")
        layout.addRow("شماره حساب:", self.login_input)
        
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("رمز (اختیاری)")
        self.password_input.setEchoMode(QLineEdit.Password)
        layout.addRow("رمز:", self.password_input)
        
        # دکمه تست اتصال
        self.test_btn = QPushButton("🔗 تست اتصال")
        self.test_btn.clicked.connect(self.test_connection)
        layout.addRow(self.test_btn)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        
        buttons.button(QDialogButtonBox.Ok).setText("تأیید")
        buttons.button(QDialogButtonBox.Cancel).setText("انصراف")
        
        layout.addRow(buttons)
        
        # بررسی اولیه اتصال
        self.check_initial_connection()
        
    def check_initial_connection(self):
        """بررسی وضعیت اولیه اتصال"""
        if MT5_AVAILABLE and mt5_fetcher and mt5_fetcher.connected:
            self.connection_status.setText("✅ متصل")
            self.connection_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.connection_status.setText("❌ قطع")
            self.connection_status.setStyleSheet("color: #F44336; font-weight: bold;")
    
    def test_connection(self):
        """تست اتصال به MT5"""
        try:
            self.test_btn.setEnabled(False)
            self.test_btn.setText("🔗 در حال اتصال...")
            
            if not MT5_AVAILABLE:
                QMessageBox.warning(self, "خطا", "MetaTrader5 نصب نیست. لطفاً با دستور 'pip install MetaTrader5' نصب کنید.")
                return
                
            # تلاش برای اتصال مجدد
            if mt5_fetcher.initialize_mt5():
                self.connection_status.setText("✅ متصل")
                self.connection_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
                QMessageBox.information(self, "موفقیت", "اتصال به MetaTrader5 با موفقیت برقرار شد")
            else:
                self.connection_status.setText("❌ قطع")
                self.connection_status.setStyleSheet("color: #F44336; font-weight: bold;")
                QMessageBox.warning(self, "خطا", "اتصال به MetaTrader5 برقرار نشد. لطفاً مطمئن شوید که متاتریدر اجرا است.")
                
        except Exception as e:
            self.connection_status.setText("❌ خطا")
            self.connection_status.setStyleSheet("color: #F44336; font-weight: bold;")
            QMessageBox.critical(self, "خطا", f"خطا در اتصال: {str(e)}")
        finally:
            self.test_btn.setEnabled(True)
            self.test_btn.setText("🔗 تست اتصال")