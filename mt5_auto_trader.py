# mt5_auto_trader_professional.py
"""
🚀 MT5 Professional Auto Trader - Elite Edition
نسخه حرفه‌ای معامله‌گر خودکار با قابلیت‌های پیشرفته

ویژگی‌های کلیدی:
✅ اجرای خودکار معاملات در MT5 (حساب Demo/Real)
✅ مانیتورینگ همزمان 10+ ارز
✅ مدیریت ریسک هوشمند
✅ گزارش‌دهی زنده و تعاملی
✅ سیستم هشدار و نوتیفیکیشن
✅ توقف اضطراری و بازیابی خودکار
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sys
import os
from pathlib import Path
import threading
from collections import defaultdict

# اضافه کردن مسیر پروژه
sys.path.append(os.path.dirname(__file__))

from strategies.adaptive_elite_rsi_strategy import ProfessionalAdvancedRsiStrategy
from indicators.rsi import calculate_rsi

# ===============================================
# تنظیمات لاگینگ پیشرفته
# ===============================================
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(
            log_dir / f'mt5_trader_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===============================================
# کلاس‌های پایه
# ===============================================

class OrderType(Enum):
    """نوع سفارش"""
    BUY = "BUY"
    SELL = "SELL"
    CLOSE_BUY = "CLOSE_BUY"
    CLOSE_SELL = "CLOSE_SELL"

class OrderStatus(Enum):
    """وضعیت سفارش"""
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"
    CLOSED = "CLOSED"

@dataclass
class Position:
    """اطلاعات پوزیشن"""
    ticket: int
    symbol: str
    type: str
    volume: float
    open_price: float
    stop_loss: float
    take_profit: float
    open_time: datetime
    profit: float = 0.0
    swap: float = 0.0
    commission: float = 0.0
    
    def to_dict(self):
        return {
            **asdict(self),
            'open_time': self.open_time.isoformat()
        }

@dataclass
class TradeResult:
    """نتیجه معامله"""
    success: bool
    order_type: OrderType
    symbol: str
    price: float
    volume: float
    ticket: Optional[int] = None
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def to_dict(self):
        return asdict(self)

# ===============================================
# نمایش پیشرفته وضعیت
# ===============================================

class LiveDashboard:
    """داشبورد زنده برای نمایش وضعیت معاملات"""
    
    def __init__(self):
        self.is_running = False
        self.thread = None
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'current_balance': 0.0,
            'open_positions': 0,
            'symbols_monitoring': [],
            'last_update': datetime.now()
        }
        self.lock = threading.Lock()
    
    def start(self):
        """شروع داشبورد"""
        self.is_running = True
        self.thread = threading.Thread(target=self._run_dashboard, daemon=True)
        self.thread.start()
    
    def stop(self):
        """توقف داشبورد"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def update_stats(self, **kwargs):
        """بروزرسانی آمار"""
        with self.lock:
            self.stats.update(kwargs)
            self.stats['last_update'] = datetime.now()
    
    def _run_dashboard(self):
        """اجرای داشبورد"""
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        idx = 0
        
        while self.is_running:
            with self.lock:
                stats = self.stats.copy()
            
            # محاسبه Win Rate
            total = stats['total_trades']
            win_rate = (stats['winning_trades'] / total * 100) if total > 0 else 0
            
            # نمایش
            print(f'\r{spinner[idx]} ', end='')
            print(f"💰 Balance: {stats['current_balance']:.2f} | ", end='')
            print(f"📊 Profit: {stats['total_profit']:.2f} | ", end='')
            print(f"📈 Trades: {total} ({win_rate:.1f}% Win) | ", end='')
            print(f"🔓 Open: {stats['open_positions']} | ", end='')
            print(f"🕐 {stats['last_update'].strftime('%H:%M:%S')}", end='', flush=True)
            
            idx = (idx + 1) % len(spinner)
            time.sleep(0.3)

# ===============================================
# مدیریت اتصال MT5
# ===============================================

class MT5ConnectionManager:
    """مدیریت اتصال به MT5 با قابلیت بازیابی خودکار"""
    
    def __init__(self, account: int = None, password: str = None, server: str = None):
        self.account = account
        self.password = password
        self.server = server
        self.is_connected = False
        self.last_connection_check = datetime.now()
        self.connection_retry_interval = 30  # ثانیه
    
    def connect(self, max_retries: int = 3) -> bool:
        """اتصال به MT5 با تلاش مجدد"""
        for attempt in range(max_retries):
            try:
                logger.info(f"🔌 Connecting to MT5 (Attempt {attempt + 1}/{max_retries})...")
                
                # بستن اتصالات قبلی
                try:
                    mt5.shutdown()
                except:
                    pass
                
                # اتصال جدید
                if not mt5.initialize():
                    error = mt5.last_error()
                    logger.error(f"❌ MT5 initialization failed: {error}")
                    time.sleep(2)
                    continue
                
                # لاگین (در صورت وجود اطلاعات)
                if self.account and self.password and self.server:
                    if not mt5.login(self.account, password=self.password, server=self.server):
                        error = mt5.last_error()
                        logger.error(f"❌ MT5 login failed: {error}")
                        time.sleep(2)
                        continue
                    logger.info(f"✅ Logged in to account {self.account}")
                
                # بررسی اطلاعات ترمینال
                terminal_info = mt5.terminal_info()
                if not terminal_info:
                    logger.error("❌ Cannot retrieve terminal info")
                    time.sleep(2)
                    continue
                
                account_info = mt5.account_info()
                if account_info:
                    logger.info(f"✅ Connected to MT5 - Balance: {account_info.balance:.2f}")
                    logger.info(f"📊 Account Type: {'Demo' if account_info.trade_mode == 0 else 'Real'}")
                
                self.is_connected = True
                self.last_connection_check = datetime.now()
                return True
                
            except Exception as e:
                logger.error(f"❌ Connection error: {e}")
                time.sleep(2)
        
        logger.error("❌ Failed to connect to MT5 after all retries")
        return False
    
    def ensure_connected(self) -> bool:
        """اطمینان از اتصال فعال"""
        # بررسی دوره‌ای اتصال
        if (datetime.now() - self.last_connection_check).seconds > self.connection_retry_interval:
            self.is_connected = self._check_connection()
            self.last_connection_check = datetime.now()
        
        if not self.is_connected:
            logger.warning("⚠️ Connection lost, reconnecting...")
            return self.connect()
        
        return True
    
    def _check_connection(self) -> bool:
        """بررسی وضعیت اتصال"""
        try:
            account_info = mt5.account_info()
            return account_info is not None
        except:
            return False
    
    def disconnect(self):
        """قطع اتصال"""
        try:
            mt5.shutdown()
            self.is_connected = False
            logger.info("🔌 Disconnected from MT5")
        except Exception as e:
            logger.error(f"❌ Disconnect error: {e}")

# ===============================================
# مدیریت نمادها
# ===============================================

class SymbolManager:
    """مدیریت نمادهای معاملاتی"""
    
    # لیست نمادهای پیشنهادی برای معامله
    RECOMMENDED_SYMBOLS = [
        "XAUUSD",   # طلا
        "EURUSD",   # یورو/دلار
        "GBPUSD",   # پوند/دلار
        "USDJPY",   # دلار/ین
        "AUDUSD",   # دلار استرالیا
        "USDCAD",   # دلار کانادا
        "NZDUSD",   # دلار نیوزلند
        "USDCHF",   # دلار/فرانک
        "EURJPY",   # یورو/ین
        "GBPJPY",   # پوند/ین
        "EURGBP",   # یورو/پوند
        "AUDJPY",   # دلار استرالیا/ین
    ]
    
    def __init__(self):
        self.available_symbols = []
        self.symbol_info = {}
    
    def discover_symbols(self, preferred_symbols: List[str] = None) -> List[str]:
        """کشف نمادهای موجود در MT5"""
        try:
            logger.info("🔍 Discovering available symbols...")
            
            # دریافت همه نمادها
            all_symbols = mt5.symbols_get()
            if not all_symbols:
                logger.error("❌ No symbols found")
                return []
            
            available = []
            
            # اگر نمادهای ترجیحی داده شده، ابتدا آنها را بررسی کن
            symbols_to_check = preferred_symbols if preferred_symbols else self.RECOMMENDED_SYMBOLS
            
            for symbol_name in symbols_to_check:
                symbol_info = mt5.symbol_info(symbol_name)
                if symbol_info and symbol_info.visible:
                    available.append(symbol_name)
                    self.symbol_info[symbol_name] = symbol_info
                elif symbol_info:
                    # اگر نماد وجود دارد ولی visible نیست، فعالش کن
                    if mt5.symbol_select(symbol_name, True):
                        available.append(symbol_name)
                        self.symbol_info[symbol_name] = mt5.symbol_info(symbol_name)
            
            self.available_symbols = available
            logger.info(f"✅ Found {len(available)} available symbols")
            for sym in available[:10]:  # نمایش 10 نماد اول
                logger.info(f"  📊 {sym}")
            
            return available
            
        except Exception as e:
            logger.error(f"❌ Error discovering symbols: {e}")
            return []
    
    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """دریافت اطلاعات نماد"""
        if symbol not in self.symbol_info:
            info = mt5.symbol_info(symbol)
            if info:
                self.symbol_info[symbol] = info
        
        return self.symbol_info.get(symbol)
    
    def calculate_pip_value(self, symbol: str, lot_size: float = 1.0) -> float:
        """محاسبه ارزش هر پیپ"""
        try:
            info = self.get_symbol_info(symbol)
            if not info:
                return 10.0  # مقدار پیش‌فرض
            
            # برای اکثر جفت‌ارزها
            if "JPY" in symbol:
                pip_value = 0.01 * lot_size * info.trade_contract_size
            else:
                pip_value = 0.0001 * lot_size * info.trade_contract_size
            
            return pip_value
            
        except Exception as e:
            logger.error(f"❌ Error calculating pip value: {e}")
            return 10.0

# ===============================================
# مدیریت ریسک پیشرفته
# ===============================================

class RiskManager:
    """مدیریت ریسک هوشمند"""
    
    def __init__(self, 
                 max_risk_per_trade: float = 0.02,
                 max_daily_loss: float = 0.05,
                 max_positions: int = 5,
                 max_symbol_exposure: float = 0.03):
        
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_positions = max_positions
        self.max_symbol_exposure = max_symbol_exposure
        
        self.daily_pnl = 0.0
        self.daily_start_balance = 0.0
        self.trades_today = 0
        self.last_reset_date = datetime.now().date()
    
    def reset_daily_stats(self, current_balance: float):
        """ریست آمار روزانه"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_start_balance = current_balance
            self.trades_today = 0
            self.last_reset_date = today
            logger.info("🔄 Daily stats reset")
    
    def can_open_trade(self, current_balance: float, open_positions: int) -> Tuple[bool, str]:
        """بررسی امکان باز کردن معامله جدید"""
        
        # بررسی تعداد پوزیشن‌های باز
        if open_positions >= self.max_positions:
            return False, f"Maximum positions reached ({self.max_positions})"
        
        # بررسی ضرر روزانه
        if self.daily_start_balance > 0:
            daily_loss_pct = abs(self.daily_pnl) / self.daily_start_balance
            if self.daily_pnl < 0 and daily_loss_pct >= self.max_daily_loss:
                return False, f"Daily loss limit reached ({daily_loss_pct*100:.1f}%)"
        
        return True, "OK"
    
    def calculate_position_size(self,
                                symbol: str,
                                entry_price: float,
                                stop_loss: float,
                                account_balance: float,
                                symbol_manager: SymbolManager) -> float:
        """محاسبه حجم پوزیشن بر اساس ریسک"""
        try:
            # محاسبه ریسک به دلار
            risk_amount = account_balance * self.max_risk_per_trade
            
            # محاسبه فاصله استاپ به پیپ
            symbol_info = symbol_manager.get_symbol_info(symbol)
            if not symbol_info:
                return 0.01
            
            point = symbol_info.point
            stop_distance = abs(entry_price - stop_loss)
            stop_distance_pips = stop_distance / (point * 10)
            
            if stop_distance_pips == 0:
                return 0.01
            
            # محاسبه حجم
            pip_value = symbol_manager.calculate_pip_value(symbol, 1.0)
            required_lots = risk_amount / (stop_distance_pips * pip_value)
            
            # اعمال محدودیت‌ها
            min_lot = symbol_info.volume_min
            max_lot = symbol_info.volume_max
            lot_step = symbol_info.volume_step
            
            # رند کردن به lot_step
            required_lots = round(required_lots / lot_step) * lot_step
            required_lots = max(min_lot, min(required_lots, max_lot))
            
            # محدودیت اضافی: حداکثر 10% از بالانس
            max_allowed_lots = (account_balance * 0.1) / (entry_price * symbol_info.trade_contract_size)
            required_lots = min(required_lots, max_allowed_lots)
            
            logger.info(f"📊 Position Size Calculation:")
            logger.info(f"  💰 Risk Amount: ${risk_amount:.2f}")
            logger.info(f"  📏 Stop Distance: {stop_distance_pips:.1f} pips")
            logger.info(f"  📦 Calculated Lots: {required_lots:.2f}")
            
            return required_lots
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            return 0.01

# ===============================================
# سیستم اجرای سفارشات
# ===============================================

class OrderExecutor:
    """سیستم حرفه‌ای اجرای سفارشات"""
    
    def __init__(self, symbol_manager: SymbolManager):
        self.symbol_manager = symbol_manager
        self.order_history = []
        self.magic_number = 20240101
    
    def open_position(self,
                     symbol: str,
                     order_type: OrderType,
                     volume: float,
                     stop_loss: float = 0.0,
                     take_profit: float = 0.0,
                     comment: str = "") -> TradeResult:
        """باز کردن پوزیشن"""
        try:
            # دریافت قیمت فعلی
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return TradeResult(
                    success=False,
                    order_type=order_type,
                    symbol=symbol,
                    price=0.0,
                    volume=volume,
                    error_message="Cannot get current price"
                )
            
            # تعیین نوع و قیمت
            if order_type == OrderType.BUY:
                mt5_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            else:  # SELL
                mt5_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            
            # آماده‌سازی درخواست
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5_type,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 20,
                "magic": self.magic_number,
                "comment": comment or f"RSI_Pro_{order_type.value}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            logger.info(f"📤 Sending order: {order_type.value} {volume} {symbol} @ {price:.5f}")
            
            # ارسال سفارش
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Order failed: {result.retcode} - {result.comment}"
                logger.error(f"❌ {error_msg}")
                
                return TradeResult(
                    success=False,
                    order_type=order_type,
                    symbol=symbol,
                    price=price,
                    volume=volume,
                    error_code=result.retcode,
                    error_message=error_msg
                )
            
            # موفقیت
            logger.info(f"✅ Order executed successfully!")
            logger.info(f"  🎫 Ticket: {result.order}")
            logger.info(f"  💵 Price: {result.price:.5f}")
            logger.info(f"  🛡️ SL: {stop_loss:.5f}")
            logger.info(f"  🎯 TP: {take_profit:.5f}")
            
            trade_result = TradeResult(
                success=True,
                order_type=order_type,
                symbol=symbol,
                price=result.price,
                volume=volume,
                ticket=result.order,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            self.order_history.append(trade_result)
            return trade_result
            
        except Exception as e:
            error_msg = f"Exception during order execution: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            return TradeResult(
                success=False,
                order_type=order_type,
                symbol=symbol,
                price=0.0,
                volume=volume,
                error_message=error_msg
            )
    
    def close_position(self, position: Position) -> TradeResult:
        """بستن پوزیشن"""
        try:
            # دریافت قیمت فعلی
            tick = mt5.symbol_info_tick(position.symbol)
            if not tick:
                return TradeResult(
                    success=False,
                    order_type=OrderType.CLOSE_BUY if position.type == "BUY" else OrderType.CLOSE_SELL,
                    symbol=position.symbol,
                    price=0.0,
                    volume=position.volume,
                    error_message="Cannot get current price"
                )
            
            # تعیین نوع معکوس
            if position.type == mt5.ORDER_TYPE_BUY or position.type == "BUY":
                close_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
                order_type_enum = OrderType.CLOSE_BUY
            else:
                close_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
                order_type_enum = OrderType.CLOSE_SELL
            
            # درخواست بستن
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": close_type,
                "position": position.ticket,
                "price": price,
                "deviation": 20,
                "magic": self.magic_number,
                "comment": f"Close_{position.ticket}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            logger.info(f"📤 Closing position: #{position.ticket} {position.symbol}")
            
            # ارسال
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Close failed: {result.retcode}"
                logger.error(f"❌ {error_msg}")
                return TradeResult(
                    success=False,
                    order_type=order_type_enum,
                    symbol=position.symbol,
                    price=price,
                    volume=position.volume,
                    error_code=result.retcode,
                    error_message=error_msg
                )
            
            logger.info(f"✅ Position closed successfully!")
            logger.info(f"  💰 Profit: ${position.profit:.2f}")
            
            return TradeResult(
                success=True,
                order_type=order_type_enum,
                symbol=position.symbol,
                price=result.price,
                volume=position.volume,
                ticket=position.ticket
            )
            
        except Exception as e:
            error_msg = f"Exception closing position: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return TradeResult(
                success=False,
                order_type=OrderType.CLOSE_BUY,
                symbol=position.symbol,
                price=0.0,
                volume=0.0,
                error_message=error_msg
            )

# ===============================================
# معامله‌گر خودکار اصلی
# ===============================================

class TradeLogger:
    """ذخیره‌سازی نتایج معاملات در فایل CSV"""
    
    def __init__(self, output_dir: str = "backtest_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.live_trades_file = self.output_dir / "live_trades.csv"
        self.positions_file = self.output_dir / "open_positions.csv"
        self.daily_summary_file = self.output_dir / "daily_summary.csv"
        
        # ایجاد فایل‌ها اگر وجود ندارند
        self._initialize_files()
    
    def _initialize_files(self):
        """ایجاد فایل‌های CSV با هدر"""
        # فایل معاملات
        if not self.live_trades_file.exists():
            pd.DataFrame(columns=[
                'timestamp', 'symbol', 'action', 'order_type', 'ticket',
                'entry_price', 'exit_price', 'volume', 'stop_loss', 'take_profit',
                'profit', 'profit_pct', 'duration_minutes', 'exit_reason'
            ]).to_csv(self.live_trades_file, index=False)
        
        # فایل پوزیشن‌های باز
        if not self.positions_file.exists():
            pd.DataFrame(columns=[
                'timestamp', 'ticket', 'symbol', 'type', 'volume',
                'open_price', 'current_price', 'stop_loss', 'take_profit',
                'profit', 'duration_minutes'
            ]).to_csv(self.positions_file, index=False)
        
        # فایل خلاصه روزانه
        if not self.daily_summary_file.exists():
            pd.DataFrame(columns=[
                'date', 'balance', 'equity', 'profit', 'trades', 'winning_trades',
                'losing_trades', 'win_rate', 'open_positions'
            ]).to_csv(self.daily_summary_file, index=False)
    
    def log_trade(self, trade_data: Dict):
        """ثبت یک معامله"""
        try:
            df = pd.DataFrame([trade_data])
            df.to_csv(self.live_trades_file, mode='a', header=False, index=False)
            logger.info(f"💾 Trade logged to CSV: {trade_data.get('ticket', 'N/A')}")
        except Exception as e:
            logger.error(f"❌ Error logging trade: {e}")
    
    def log_positions(self, positions: List[Dict]):
        """ثبت وضعیت پوزیشن‌های باز"""
        try:
            if positions:
                df = pd.DataFrame(positions)
                df.to_csv(self.positions_file, index=False)
        except Exception as e:
            logger.error(f"❌ Error logging positions: {e}")
    
    def log_daily_summary(self, summary: Dict):
        """ثبت خلاصه روزانه"""
        try:
            df = pd.DataFrame([summary])
            
            # بررسی وجود رکورد امروز
            if self.daily_summary_file.exists():
                existing = pd.read_csv(self.daily_summary_file)
                today = summary['date']
                existing = existing[existing['date'] != today]
                df = pd.concat([existing, df], ignore_index=True)
            
            df.to_csv(self.daily_summary_file, index=False)
        except Exception as e:
            logger.error(f"❌ Error logging daily summary: {e}")

class MT5ProfessionalAutoTrader:
    """
    معامله‌گر خودکار حرفه‌ای MT5
    
    قابلیت‌ها:
    - مانیتورینگ همزمان چندین نماد
    - مدیریت ریسک پیشرفته
    - اجرای خودکار سیگنال‌ها
    - گزارش‌دهی زنده
    - مدیریت هوشمند پوزیشن‌های باز
    - ذخیره‌سازی نتایج در CSV
    """
    
    def __init__(self,
                 account: int = None,
                 password: str = None,
                 server: str = None,
                 symbols: List[str] = None,
                 timeframe: str = "H1",
                 max_risk_per_trade: float = 0.02,
                 max_positions: int = 5,
                 enable_trailing_stop: bool = True,
                 trailing_stop_pips: float = 50):
        
        logger.info("=" * 70)
        logger.info("🚀 Initializing MT5 Professional Auto Trader")
        logger.info("=" * 70)
        
        # مدیریت اتصال
        self.connection_manager = MT5ConnectionManager(account, password, server)
        if not self.connection_manager.connect():
            raise ConnectionError("❌ Cannot connect to MT5")
        
        # مدیریت نمادها
        self.symbol_manager = SymbolManager()
        self.symbols = self.symbol_manager.discover_symbols(symbols)
        if not self.symbols:
            raise ValueError("❌ No valid symbols found")
        
        # مدیریت ریسک
        self.risk_manager = RiskManager(
            max_risk_per_trade=max_risk_per_trade,
            max_positions=max_positions
        )
        
        # اجراکننده سفارشات
        self.order_executor = OrderExecutor(self.symbol_manager)
        
        # استراتژی برای هر نماد
        self.strategies = {}
        for symbol in self.symbols:
            self.strategies[symbol] = ProfessionalAdvancedRsiStrategy(
                enable_short_trades=True,
                risk_per_trade=max_risk_per_trade
            )
        
        # تایم‌فریم
        self.timeframe = self._parse_timeframe(timeframe)
        self.timeframe_str = timeframe
        
        # داشبورد
        self.dashboard = LiveDashboard()
        
        # ذخیره‌ساز نتایج
        self.trade_logger = TradeLogger()
        
        # تنظیمات Trailing Stop
        self.enable_trailing_stop = enable_trailing_stop
        self.trailing_stop_pips = trailing_stop_pips
        
        # وضعیت
        self.is_running = False
        self.total_cycles = 0
        self.last_positions_check = {}
        self.position_entry_data = {}  # ذخیره اطلاعات ورود به معامله
        
        # آمار
        self.stats = {
            'trades_executed': 0,
            'trades_closed': 0,
            'total_profit': 0.0,
            'winning_trades': 0,
            'losing_trades': 0
        }
        
        logger.info(f"✅ Trader initialized successfully")
        logger.info(f"📊 Monitoring {len(self.symbols)} symbols")
        logger.info(f"⏱️ Timeframe: {timeframe}")
        logger.info(f"🎯 Max Risk: {max_risk_per_trade*100}%")
        logger.info(f"🔢 Max Positions: {max_positions}")
        logger.info(f"🎯 Trailing Stop: {'Enabled' if enable_trailing_stop else 'Disabled'}")
    
    def _parse_timeframe(self, tf_str: str) -> int:
        """تبدیل تایم‌فریم"""
        timeframes = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1
        }
        return timeframes.get(tf_str, mt5.TIMEFRAME_H1)
    
    def get_market_data(self, symbol: str, count: int = 100) -> Optional[pd.DataFrame]:
        """دریافت داده‌های بازار"""
        try:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"❌ Symbol {symbol} not available")
                return None
            
            rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0, count)
            if rates is None or len(rates) == 0:
                logger.warning(f"⚠️ No data for {symbol}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.rename(columns={
                'time': 'open_time',
                'tick_volume': 'volume'
            })
            df = df.set_index('open_time')
            df = df.sort_index()
            
            # محاسبه RSI
            df = calculate_rsi(df, period=14)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting data for {symbol}: {e}")
            return None
    
    def get_open_positions(self, symbol: str = None) -> List[Position]:
        """دریافت پوزیشن‌های باز"""
        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
            
            if not positions:
                return []
            
            result = []
            for pos in positions:
                result.append(Position(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    type="BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                    volume=pos.volume,
                    open_price=pos.price_open,
                    stop_loss=pos.sl,
                    take_profit=pos.tp,
                    open_time=datetime.fromtimestamp(pos.time),
                    profit=pos.profit,
                    swap=pos.swap,
                    commission=pos.commission
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error getting positions: {e}")
            return []
    
    def manage_open_positions(self, symbol: str, data: pd.DataFrame) -> List[Dict]:
        """
        مدیریت هوشمند پوزیشن‌های باز
        
        شامل:
        - بررسی شرایط خروج بر اساس RSI
        - Trailing Stop
        - Time-based Exit
        - Manual Stop Loss/Take Profit Check
        """
        results = []
        
        try:
            positions = self.get_open_positions(symbol)
            if not positions:
                return results
            
            current_price = data['close'].iloc[-1]
            current_rsi = data['RSI'].iloc[-1]
            current_time = datetime.now()
            
            for position in positions:
                should_close = False
                close_reason = ""
                
                # 1. بررسی RSI Exit Signals
                if position.type == "BUY":
                    # خروج از LONG در اشباع خرید
                    if current_rsi > 70:
                        should_close = True
                        close_reason = "RSI_OVERBOUGHT"
                        logger.info(f"📊 {symbol}: RSI Exit signal for LONG ({current_rsi:.1f})")
                
                else:  # SHORT/SELL
                    # خروج از SHORT در اشباع فروش
                    if current_rsi < 30:
                        should_close = True
                        close_reason = "RSI_OVERSOLD"
                        logger.info(f"📊 {symbol}: RSI Exit signal for SHORT ({current_rsi:.1f})")
                
                # 2. بررسی Time-Based Exit (مثلاً بیش از 48 ساعت)
                duration = (current_time - position.open_time).total_seconds() / 3600
                if duration > 48:  # 48 ساعت
                    should_close = True
                    close_reason = "TIME_EXIT"
                    logger.info(f"⏰ {symbol}: Time-based exit ({duration:.1f} hours)")
                
                # 3. Trailing Stop
                if self.enable_trailing_stop and not should_close:
                    if self._check_trailing_stop(position, current_price):
                        should_close = True
                        close_reason = "TRAILING_STOP"
                        logger.info(f"🎯 {symbol}: Trailing stop triggered")
                
                # 4. بررسی سود خوب (مثلاً 3% سود)
                if not should_close:
                    profit_pct = (position.profit / (position.open_price * position.volume)) * 100
                    if profit_pct > 3.0:  # 3% سود
                        should_close = True
                        close_reason = "PROFIT_TARGET"
                        logger.info(f"💰 {symbol}: Profit target reached ({profit_pct:.2f}%)")
                
                # اجرای بستن پوزیشن
                if should_close:
                    logger.info(f"🔒 Closing position #{position.ticket} - Reason: {close_reason}")
                    
                    close_result = self.order_executor.close_position(position)
                    
                    if close_result.success:
                        # محاسبه آمار
                        duration_minutes = (current_time - position.open_time).total_seconds() / 60
                        profit_pct = (position.profit / (position.open_price * position.volume)) * 100
                        
                        # ثبت در آمار
                        self.stats['trades_closed'] += 1
                        self.stats['total_profit'] += position.profit
                        
                        if position.profit > 0:
                            self.stats['winning_trades'] += 1
                        else:
                            self.stats['losing_trades'] += 1
                        
                        # ذخیره در CSV
                        trade_data = {
                            'timestamp': current_time.isoformat(),
                            'symbol': symbol,
                            'action': 'CLOSE',
                            'order_type': position.type,
                            'ticket': position.ticket,
                            'entry_price': position.open_price,
                            'exit_price': current_price,
                            'volume': position.volume,
                            'stop_loss': position.stop_loss,
                            'take_profit': position.take_profit,
                            'profit': position.profit,
                            'profit_pct': profit_pct,
                            'duration_minutes': duration_minutes,
                            'exit_reason': close_reason
                        }
                        
                        self.trade_logger.log_trade(trade_data)
                        
                        results.append({
                            'action': 'CLOSED',
                            'ticket': position.ticket,
                            'symbol': symbol,
                            'profit': position.profit,
                            'reason': close_reason
                        })
                        
                        logger.info(f"✅ Position closed: Profit ${position.profit:.2f}")
                    else:
                        logger.error(f"❌ Failed to close position #{position.ticket}")
                        results.append({
                            'action': 'CLOSE_FAILED',
                            'ticket': position.ticket,
                            'error': close_result.error_message
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error managing positions for {symbol}: {e}", exc_info=True)
            return results
    
    def _check_trailing_stop(self, position: Position, current_price: float) -> bool:
        """بررسی Trailing Stop"""
        try:
            # محاسبه سود فعلی به پیپ
            symbol_info = self.symbol_manager.get_symbol_info(position.symbol)
            if not symbol_info:
                return False
            
            point = symbol_info.point
            
            if position.type == "BUY":
                # برای LONG: اگر قیمت از بالاترین نقطه به اندازه trailing_stop افت کرد
                entry_key = f"{position.ticket}_highest"
                
                if entry_key not in self.position_entry_data:
                    self.position_entry_data[entry_key] = position.open_price
                
                highest = max(self.position_entry_data[entry_key], current_price)
                self.position_entry_data[entry_key] = highest
                
                drop_pips = (highest - current_price) / (point * 10)
                
                if drop_pips >= self.trailing_stop_pips:
                    return True
            
            else:  # SHORT
                # برای SHORT: اگر قیمت از پایین‌ترین نقطه به اندازه trailing_stop صعود کرد
                entry_key = f"{position.ticket}_lowest"
                
                if entry_key not in self.position_entry_data:
                    self.position_entry_data[entry_key] = position.open_price
                
                lowest = min(self.position_entry_data[entry_key], current_price)
                self.position_entry_data[entry_key] = lowest
                
                rise_pips = (current_price - lowest) / (point * 10)
                
                if rise_pips >= self.trailing_stop_pips:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking trailing stop: {e}")
            return False
        """دریافت پوزیشن‌های باز"""
        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
            
            if not positions:
                return []
            
            result = []
            for pos in positions:
                result.append(Position(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    type="BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                    volume=pos.volume,
                    open_price=pos.price_open,
                    stop_loss=pos.sl,
                    take_profit=pos.tp,
                    open_time=datetime.fromtimestamp(pos.time),
                    profit=pos.profit,
                    swap=pos.swap,
                    commission=pos.commission
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error getting positions: {e}")
            return []
    
    def process_symbol(self, symbol: str) -> Dict:
        """پردازش یک نماد"""
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 Processing: {symbol}")
            logger.info(f"{'='*60}")
            
            # دریافت داده‌های بازار
            data = self.get_market_data(symbol, 100)
            if data is None or len(data) < 50:
                logger.warning(f"⚠️ Insufficient data for {symbol}")
                return {'symbol': symbol, 'action': 'SKIP', 'reason': 'Insufficient data'}
            
            # 🔥 بررسی و مدیریت پوزیشن‌های باز
            open_positions = self.get_open_positions(symbol)
            
            if open_positions:
                logger.info(f"🔓 Managing {len(open_positions)} open position(s) for {symbol}")
                
                # مدیریت پوزیشن‌های باز
                close_results = self.manage_open_positions(symbol, data)
                
                if close_results:
                    logger.info(f"✅ Managed positions: {len(close_results)} actions taken")
                    return {
                        'symbol': symbol,
                        'action': 'POSITION_MANAGED',
                        'close_results': close_results,
                        'positions_before': len(open_positions)
                    }
                else:
                    # هنوز پوزیشن باز است، فعلاً نگه‌داری
                    current_profit = sum(p.profit for p in open_positions)
                    logger.info(f"👁️ Monitoring open positions - Current P/L: ${current_profit:.2f}")
                    
                    return {
                        'symbol': symbol,
                        'action': 'MONITOR',
                        'has_position': True,
                        'positions': len(open_positions),
                        'current_profit': current_profit
                    }
            
            # اگر پوزیشن باز نداریم، دنبال سیگنال جدید باش
            logger.info(f"🔍 Looking for new signal for {symbol}")
            
            # تولید سیگنال
            signal = self.strategies[symbol].generate_signal(data, len(data) - 1)
            
            logger.info(f"🎯 Signal: {signal.get('action', 'HOLD')}")
            if signal.get('reason'):
                logger.info(f"💡 Reason: {signal['reason']}")
            
            # اگر سیگنال معتبر نیست
            if signal['action'] not in ['BUY', 'SHORT']:
                return {
                    'symbol': symbol,
                    'action': signal['action'],
                    'reason': signal.get('reason', 'No signal')
                }
            
            # بررسی امکان باز کردن معامله
            account_info = mt5.account_info()
            if not account_info:
                logger.error("❌ Cannot get account info")
                return {'symbol': symbol, 'action': 'ERROR', 'reason': 'No account info'}
            
            current_balance = account_info.balance
            total_open = len(self.get_open_positions())
            
            can_trade, reason = self.risk_manager.can_open_trade(current_balance, total_open)
            if not can_trade:
                logger.warning(f"⚠️ Cannot open trade: {reason}")
                return {
                    'symbol': symbol,
                    'action': 'BLOCKED',
                    'reason': reason
                }
            
            # محاسبه حجم پوزیشن
            current_price = data['close'].iloc[-1]
            stop_loss = signal.get('stop_loss', 0)
            take_profit = signal.get('take_profit', 0)
            
            if stop_loss == 0:
                # محاسبه استاپ لاس بر اساس ATR
                atr = self.strategies[symbol].calculate_atr(data.tail(20))
                if signal['action'] == 'BUY':
                    stop_loss = current_price - (atr * 2)
                    take_profit = current_price + (atr * 4)
                else:
                    stop_loss = current_price + (atr * 2)
                    take_profit = current_price - (atr * 4)
            
            volume = self.risk_manager.calculate_position_size(
                symbol=symbol,
                entry_price=current_price,
                stop_loss=stop_loss,
                account_balance=current_balance,
                symbol_manager=self.symbol_manager
            )
            
            # اجرای سفارش
            order_type = OrderType.BUY if signal['action'] == 'BUY' else OrderType.SELL
            
            logger.info(f"\n🎯 Opening Position:")
            logger.info(f"  Symbol: {symbol}")
            logger.info(f"  Type: {order_type.value}")
            logger.info(f"  Volume: {volume:.2f}")
            logger.info(f"  Current Price: {current_price:.5f}")
            logger.info(f"  Stop Loss: {stop_loss:.5f}")
            logger.info(f"  Take Profit: {take_profit:.5f}")
            
            result = self.order_executor.open_position(
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=f"RSI_Elite_{symbol}"
            )
            
            if result.success:
                self.stats['trades_executed'] += 1
                
                # ذخیره اطلاعات ورود
                entry_data = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'action': 'OPEN',
                    'order_type': order_type.value,
                    'ticket': result.ticket,
                    'entry_price': result.price,
                    'exit_price': None,
                    'volume': volume,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'profit': None,
                    'profit_pct': None,
                    'duration_minutes': None,
                    'exit_reason': None
                }
                
                self.trade_logger.log_trade(entry_data)
                
                logger.info(f"✅ Trade opened successfully!")
                return {
                    'symbol': symbol,
                    'action': 'EXECUTED',
                    'order_type': order_type.value,
                    'ticket': result.ticket,
                    'price': result.price,
                    'volume': volume,
                    'sl': stop_loss,
                    'tp': take_profit
                }
            else:
                logger.error(f"❌ Failed to open trade: {result.error_message}")
                return {
                    'symbol': symbol,
                    'action': 'FAILED',
                    'reason': result.error_message
                }
            
        except Exception as e:
            logger.error(f"❌ Error processing {symbol}: {e}", exc_info=True)
            return {
                'symbol': symbol,
                'action': 'ERROR',
                'reason': str(e)
            }
    
    def update_dashboard_stats(self):
        """بروزرسانی آمار داشبورد"""
        try:
            account_info = mt5.account_info()
            if account_info:
                positions = self.get_open_positions()
                
                self.dashboard.update_stats(
                    total_trades=self.stats['trades_executed'],
                    winning_trades=self.stats['winning_trades'],
                    losing_trades=self.stats['losing_trades'],
                    total_profit=self.stats['total_profit'],
                    current_balance=account_info.balance,
                    open_positions=len(positions),
                    symbols_monitoring=self.symbols
                )
        except Exception as e:
            logger.error(f"❌ Error updating dashboard: {e}")
    
    def run_trading_cycle(self) -> Dict[str, any]:
        """اجرای یک سیکل کامل معاملاتی"""
        try:
            self.total_cycles += 1
            logger.info(f"\n{'#'*70}")
            logger.info(f"🔄 Trading Cycle #{self.total_cycles} - {datetime.now()}")
            logger.info(f"{'#'*70}\n")
            
            # بررسی اتصال
            if not self.connection_manager.ensure_connected():
                logger.error("❌ Connection lost and cannot reconnect")
                return {'status': 'CONNECTION_ERROR'}
            
            # ریست آمار روزانه
            account_info = mt5.account_info()
            if account_info:
                self.risk_manager.reset_daily_stats(account_info.balance)
            
            # پردازش هر نماد
            results = {}
            for i, symbol in enumerate(self.symbols):
                logger.info(f"\n[{i+1}/{len(self.symbols)}] Processing {symbol}...")
                result = self.process_symbol(symbol)
                results[symbol] = result
                
                # فاصله بین نمادها
                time.sleep(1)
            
            # 💾 ذخیره وضعیت پوزیشن‌های باز
            self._save_positions_snapshot()
            
            # 💾 ذخیره خلاصه روزانه
            self._save_daily_summary()
            
            # بروزرسانی آمار
            self.update_dashboard_stats()
            
            # خلاصه سیکل
            executed = sum(1 for r in results.values() if r.get('action') == 'EXECUTED')
            monitoring = sum(1 for r in results.values() if r.get('action') == 'MONITOR')
            managed = sum(1 for r in results.values() if r.get('action') == 'POSITION_MANAGED')
            
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 Cycle Summary:")
            logger.info(f"  ✅ New Positions: {executed}")
            logger.info(f"  🔒 Positions Closed: {managed}")
            logger.info(f"  👁️ Still Monitoring: {monitoring}")
            logger.info(f"  ⏭️ Skipped: {len(results) - executed - monitoring - managed}")
            logger.info(f"{'='*60}\n")
            
            return {
                'status': 'COMPLETED',
                'cycle': self.total_cycles,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"❌ Error in trading cycle: {e}", exc_info=True)
            return {'status': 'ERROR', 'error': str(e)}
    
    def _save_positions_snapshot(self):
        """ذخیره وضعیت فعلی پوزیشن‌های باز"""
        try:
            positions = self.get_open_positions()
            if not positions:
                return
            
            positions_data = []
            current_time = datetime.now()
            
            for pos in positions:
                # دریافت قیمت فعلی
                tick = mt5.symbol_info_tick(pos.symbol)
                current_price = tick.bid if pos.type == "BUY" else tick.ask
                
                duration = (current_time - pos.open_time).total_seconds() / 60
                
                positions_data.append({
                    'timestamp': current_time.isoformat(),
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': pos.type,
                    'volume': pos.volume,
                    'open_price': pos.open_price,
                    'current_price': current_price if tick else pos.open_price,
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit,
                    'profit': pos.profit,
                    'duration_minutes': duration
                })
            
            self.trade_logger.log_positions(positions_data)
            
        except Exception as e:
            logger.error(f"❌ Error saving positions snapshot: {e}")
    
    def _save_daily_summary(self):
        """ذخیره خلاصه روزانه"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return
            
            positions = self.get_open_positions()
            
            win_rate = 0
            if self.stats['trades_closed'] > 0:
                win_rate = (self.stats['winning_trades'] / self.stats['trades_closed']) * 100
            
            summary = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'balance': account_info.balance,
                'equity': account_info.equity,
                'profit': account_info.equity - account_info.balance,
                'trades': self.stats['trades_executed'],
                'winning_trades': self.stats['winning_trades'],
                'losing_trades': self.stats['losing_trades'],
                'win_rate': win_rate,
                'open_positions': len(positions)
            }
            
            self.trade_logger.log_daily_summary(summary)
            
        except Exception as e:
            logger.error(f"❌ Error saving daily summary: {e}")
    
    def start(self, 
              interval_minutes: int = 15,
              max_cycles: int = None,
              run_once: bool = False):
        """شروع معامله‌گر خودکار"""
        
        logger.info("\n" + "="*70)
        logger.info("🚀 Starting MT5 Professional Auto Trader")
        logger.info("="*70)
        logger.info(f"⏱️ Interval: {interval_minutes} minutes")
        logger.info(f"🔢 Max Cycles: {max_cycles if max_cycles else '∞'}")
        logger.info(f"📊 Symbols: {', '.join(self.symbols[:5])}{'...' if len(self.symbols) > 5 else ''}")
        logger.info(f"💰 Risk per Trade: {self.risk_manager.max_risk_per_trade*100}%")
        logger.info("="*70 + "\n")
        
        # شروع داشبورد
        self.dashboard.start()
        self.is_running = True
        
        try:
            cycle_count = 0
            
            while self.is_running:
                # بررسی محدودیت سیکل
                if max_cycles and cycle_count >= max_cycles:
                    logger.info(f"✅ Reached max cycles ({max_cycles})")
                    break
                
                cycle_count += 1
                
                # اجرای سیکل
                result = self.run_trading_cycle()
                
                # اگر فقط یکبار اجرا شود
                if run_once:
                    logger.info("✅ Single cycle completed")
                    break
                
                # انتظار تا سیکل بعدی
                logger.info(f"⏳ Waiting {interval_minutes} minutes until next cycle...")
                
                for remaining in range(interval_minutes * 60, 0, -10):
                    if not self.is_running:
                        break
                    time.sleep(min(10, remaining))
                
        except KeyboardInterrupt:
            logger.info("\n⚠️ Stopped by user (Ctrl+C)")
        except Exception as e:
            logger.error(f"❌ Critical error: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self):
        """توقف معامله‌گر"""
        logger.info("\n🛑 Stopping trader...")
        self.is_running = False
        self.dashboard.stop()
        
        # نمایش آمار نهایی
        self.print_final_stats()
        
        # قطع اتصال
        self.connection_manager.disconnect()
        logger.info("✅ Trader stopped successfully\n")
    
    def print_final_stats(self):
        """نمایش آمار نهایی"""
        try:
            account_info = mt5.account_info()
            positions = self.get_open_positions()
            
            logger.info("\n" + "="*70)
            logger.info("📊 FINAL STATISTICS")
            logger.info("="*70)
            
            if account_info:
                logger.info(f"💰 Final Balance: ${account_info.balance:.2f}")
                logger.info(f"📈 Equity: ${account_info.equity:.2f}")
                logger.info(f"💵 Profit: ${account_info.equity - account_info.balance:.2f}")
            
            logger.info(f"\n📊 Trading Stats:")
            logger.info(f"  Total Cycles: {self.total_cycles}")
            logger.info(f"  Trades Executed: {self.stats['trades_executed']}")
            logger.info(f"  Trades Closed: {self.stats['trades_closed']}")
            logger.info(f"  Open Positions: {len(positions)}")
            
            if self.stats['trades_closed'] > 0:
                win_rate = (self.stats['winning_trades'] / self.stats['trades_closed']) * 100
                logger.info(f"  Win Rate: {win_rate:.1f}%")
                logger.info(f"  Total Profit: ${self.stats['total_profit']:.2f}")
            
            logger.info("="*70 + "\n")
            
        except Exception as e:
            logger.error(f"❌ Error printing stats: {e}")
    
    def emergency_close_all(self):
        """بستن اضطراری تمام پوزیشن‌ها"""
        logger.warning("⚠️ EMERGENCY: Closing all positions...")
        
        positions = self.get_open_positions()
        closed = 0
        
        for pos in positions:
            result = self.order_executor.close_position(pos)
            if result.success:
                closed += 1
                logger.info(f"✅ Closed position #{pos.ticket}")
            else:
                logger.error(f"❌ Failed to close #{pos.ticket}")
        
        logger.info(f"✅ Emergency close completed: {closed}/{len(positions)} positions closed")

# ===============================================
# تابع اصلی اجرا
# ===============================================

def main():
    """تابع اصلی برای اجرای معامله‌گر"""
    
    print("\n" + "="*70)
    print("🚀 MT5 PROFESSIONAL AUTO TRADER - ELITE EDITION")
    print("="*70)
    print("📋 Configuration:")
    print("-"*70)
    
    # تنظیمات
    config = {
        # اتصال MT5 (None برای استفاده از تنظیمات پیش‌فرض)
        "account": None,
        "password": None,
        "server": None,
        
        # نمادها (None برای استفاده از لیست پیش‌فرض)
        "symbols": [
            "XAUUSD",   # طلا
            "EURUSD",   # یورو/دلار
            "GBPUSD",   # پوند/دلار
            "USDJPY",   # دلار/ین
            "AUDUSD",   # دلار استرالیا
            "USDCAD",   # دلار کانادا
            "NZDUSD",   # دلار نیوزلند
            "USDCHF",   # دلار/فرانک
            "EURJPY",   # یورو/ین
            "GBPJPY",   # پوند/ین
        ],
        
        # تنظیمات معاملاتی
        "timeframe": "H1",
        "max_risk_per_trade": 0.02,  # 2% ریسک هر معامله
        "max_positions": 5,
        
        # تنظیمات اجرا
        "interval_minutes": 15,
        "max_cycles": None,  # None = اجرای بی‌نهایت
        "run_once": False,  # True = فقط یک سیکل
    }
    
    print(f"  📊 Symbols: {len(config['symbols'])} instruments")
    print(f"  ⏱️ Timeframe: {config['timeframe']}")
    print(f"  💰 Risk per Trade: {config['max_risk_per_trade']*100}%")
    print(f"  🔢 Max Positions: {config['max_positions']}")
    print(f"  ⏰ Check Interval: {config['interval_minutes']} minutes")
    print("="*70)
    print("\n⚠️  WARNING: This will execute REAL trades in MT5!")
    print("   Make sure you're using a DEMO account for testing.\n")
    
    response = input("Continue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("❌ Cancelled by user")
        return
    
    try:
        # ایجاد معامله‌گر
        trader = MT5ProfessionalAutoTrader(
            account=config["account"],
            password=config["password"],
            server=config["server"],
            symbols=config["symbols"],
            timeframe=config["timeframe"],
            max_risk_per_trade=config["max_risk_per_trade"],
            max_positions=config["max_positions"]
        )
        
        # شروع معامله‌گری
        trader.start(
            interval_minutes=config["interval_minutes"],
            max_cycles=config["max_cycles"],
            run_once=config["run_once"]
        )
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Critical error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
    finally:
        print("\n✅ Program terminated")

if __name__ == "__main__":
    main()