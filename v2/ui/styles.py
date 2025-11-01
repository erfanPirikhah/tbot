# ui/styles.py

def get_main_stylesheet():
    """دریافت استایل‌شیت اصلی"""
    return """
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
    """