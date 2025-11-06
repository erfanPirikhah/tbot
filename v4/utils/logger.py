# utils/logger.py

import logging
import sys
import os
from datetime import datetime
from typing import Optional

def setup_logger(
    name: str = "trading_bot",
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_dir: Optional[str] = None
) -> logging.Logger:
    """
    تنظیمات پیشرفته برای لاگ‌گیری
    
    Args:
        name: نام لاگر
        level: سطح لاگ‌گیری
        log_to_file: ذخیره در فایل
        log_to_console: نمایش در کنسول
        log_dir: مسیر پوشه لاگ‌ها (در صورت عدم تنظیم، 'logs' استفاده می‌شود)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Force UTF-8 for stdout to avoid UnicodeEncodeError on Windows cp1252 consoles
    try:
        import os
        os.environ["PYTHONIOENCODING"] = "UTF-8"
        os.environ["PYTHONUTF8"] = "1"
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # جلوگیری از ایجاد هندلرهای تکراری
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # هندلر کنسول (UTF-8 safe)
    if log_to_console:
        stdout_stream = sys.stdout
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            import io
            if not (getattr(stdout_stream, "encoding", "") or "").lower().startswith("utf"):
                stdout_stream = io.TextIOWrapper(stdout_stream.buffer, encoding="utf-8", errors="replace")
        except Exception:
            pass
        console_handler = logging.StreamHandler(stdout_stream)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # هندلر فایل
    if log_to_file:
        resolved_dir = log_dir or "logs"
        os.makedirs(resolved_dir, exist_ok=True)
        
        log_file = os.path.join(
            resolved_dir,
            f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_performance_logger(log_dir: Optional[str] = None) -> logging.Logger:
    """لاگر مخصوص عملکرد"""
    return setup_logger("performance", logging.INFO, log_to_file=True, log_to_console=True, log_dir=log_dir)

def get_trade_logger(log_dir: Optional[str] = None) -> logging.Logger:
    """لاگر مخصوص معاملات"""
    return setup_logger("trades", logging.INFO, log_to_file=True, log_to_console=True, log_dir=log_dir)

def get_error_logger(log_dir: Optional[str] = None) -> logging.Logger:
    """لاگر مخصوص خطاها"""
    return setup_logger("errors", logging.ERROR, log_to_file=True, log_to_console=True, log_dir=log_dir)