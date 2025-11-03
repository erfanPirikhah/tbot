# utils/font_manager.py

from PyQt5.QtGui import QFont, QFontDatabase

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