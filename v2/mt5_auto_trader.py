# mt5_auto_trader.py

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import sys
import os

# اضافه کردن مسیر پروژه
sys.path.append(os.path.dirname(__file__))

from strategies.professional_advanced_rsi_strategy import ProfessionalAdvancedRsiStrategy
from indicators.rsi import calculate_rsi

# تنظیمات لاگینگ پیشرفته
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mt5_auto_trading.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MT5AutoTrader:
    """
    کلاس حرفه‌ای برای اجرای خودکار استراتژی RSI در MetaTrader 5
    """
    
    def __init__(self, 
                 account: int = None, 
                 password: str = None, 
                 server: str = None,
                 symbol: str = "XAUUSD",
                 timeframe: str = "H1",
                 lot_size: float = 0.1,
                 max_risk_per_trade: float = 0.02):
        
        self.symbol = symbol
        self.timeframe = self._get_timeframe(timeframe)
        self.lot_size = lot_size
        self.max_risk_per_trade = max_risk_per_trade
        self.strategy = ProfessionalAdvancedRsiStrategy(
            enable_short_trades=True,
            use_adx_filter=True,
            use_partial_exits=True,
            use_break_even=True,
            min_signal_score=7.0
        )
        
        # اتصال به MT5
        self._connect_mt5(account, password, server)
        
        # اطلاعات حساب
        self.account_info = mt5.account_info()
        self.initial_balance = self.account_info.balance
        
        logger.info(f"✅ MT5 AutoTrader initialized for {symbol}")
        logger.info(f"💰 Account Balance: {self.initial_balance:.2f}")
        logger.info(f"📊 Timeframe: {timeframe}")
    
    def _get_timeframe(self, tf_str: str) -> int:
        """تبدیل تایم‌فریم رشته‌ای به کد MT5"""
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
    
    def _connect_mt5(self, account: int, password: str, server: str) -> bool:
        """اتصال به MetaTrader 5"""
        try:
            if not mt5.initialize():
                logger.error("❌ Failed to initialize MT5")
                return False
            
            # اگر اطلاعات لاگین ارائه شده، لاگین کن
            if account and password and server:
                authorized = mt5.login(account, password=password, server=server)
                if not authorized:
                    logger.error(f"❌ Failed to login to account {account}")
                    return False
                logger.info(f"✅ Logged in to account {account}")
            else:
                logger.info("✅ Connected to MT5 with default settings")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            return False
    
    def get_market_data(self, count: int = 100) -> Optional[pd.DataFrame]:
        """دریافت داده‌های بازار از MT5"""
        try:
            # اطمینان از انتخاب نماد
            if not mt5.symbol_select(self.symbol, True):
                logger.error(f"❌ Symbol {self.symbol} not found")
                return None
            
            # دریافت داده‌های تاریخی
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, count)
            if rates is None:
                logger.error(f"❌ No data received for {self.symbol}")
                return None
            
            # تبدیل به DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.rename(columns={
                'time': 'open_time',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            })
            df = df.set_index('open_time')
            df = df.sort_index()
            
            # محاسبه RSI
            df = calculate_rsi(df)
            
            logger.info(f"📊 Market data retrieved: {len(df)} candles for {self.symbol}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting market data: {e}")
            return None
    
    def calculate_position_size(self, stop_loss_pips: float) -> float:
        """محاسبه حجم پوزیشن بر اساس مدیریت ریسک"""
        try:
            # دریافت قیمت فعلی
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                return self.lot_size
            
            current_price = tick.bid
            
            # محاسبه ارزش هر پیپ
            symbol_info = mt5.symbol_info(self.symbol)
            point_value = symbol_info.trade_tick_value
            
            # محاسبه ریسک بر اساس درصد حساب
            account_balance = self.account_info.balance
            risk_amount = account_balance * self.max_risk_per_trade
            
            # محاسبه حجم
            if stop_loss_pips > 0:
                risk_per_lot = stop_loss_pips * point_value
                calculated_lots = risk_amount / risk_per_lot
                
                # محدودیت‌های حجم
                min_lot = symbol_info.volume_min
                max_lot = symbol_info.volume_max
                calculated_lots = max(min_lot, min(calculated_lots, max_lot))
                
                logger.info(f"📏 Calculated lot size: {calculated_lots:.2f}")
                return round(calculated_lots, 2)
            else:
                return self.lot_size
                
        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            return self.lot_size
    
    def place_order(self, signal: Dict, stop_loss_pips: float = 200) -> bool:
        """ارسال سفارش به MT5"""
        try:
            action = signal['action']
            current_price = signal.get('price', 0)
            
            if action not in ['BUY', 'SHORT']:
                logger.info("🟡 No trade signal")
                return False
            
            # تعیین نوع سفارش
            order_type = mt5.ORDER_TYPE_BUY if action == 'BUY' else mt5.ORDER_TYPE_SELL
            
            # محاسبه قیمت‌ها
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                logger.error("❌ Cannot get current tick")
                return False
            
            price = tick.ask if action == 'BUY' else tick.bid
            point = mt5.symbol_info(self.symbol).point
            
            # محاسبه استاپ لاس و تیک پروفیت
            if action == 'BUY':
                stop_loss = price - (stop_loss_pips * point)
                take_profit = price + (stop_loss_pips * 2 * point)  # ریسک به ریوارد 1:2
            else:  # SHORT
                stop_loss = price + (stop_loss_pips * point)
                take_profit = price - (stop_loss_pips * 2 * point)
            
            # محاسبه حجم
            volume = self.calculate_position_size(stop_loss_pips)
            
            # آماده‌سازی درخواست سفارش
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 20,
                "magic": 2023,
                "comment": f"RSI_Pro_{action}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # ارسال سفارش
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"❌ Order failed: {result.retcode}")
                return False
            
            logger.info(f"✅ Order executed: {action} {volume} lots at {price:.4f}")
            logger.info(f"🛡️ SL: {stop_loss:.4f}, 🎯 TP: {take_profit:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error placing order: {e}")
            return False
    
    def close_all_positions(self) -> bool:
        """بستن تمام پوزیشن‌های باز"""
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if not positions:
                logger.info("📭 No open positions")
                return True
            
            for position in positions:
                # تعیین نوع معکوس برای بستن پوزیشن
                if position.type == mt5.ORDER_TYPE_BUY:
                    close_type = mt5.ORDER_TYPE_SELL
                    price = mt5.symbol_info_tick(self.symbol).bid
                else:
                    close_type = mt5.ORDER_TYPE_BUY
                    price = mt5.symbol_info_tick(self.symbol).ask
                
                # درخواست بستن پوزیشن
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": position.ticket,
                    "symbol": self.symbol,
                    "volume": position.volume,
                    "type": close_type,
                    "price": price,
                    "deviation": 20,
                    "magic": 2023,
                    "comment": "Close_All",
                    "type_time": mt5.ORDER_TIME_GTC,
                }
                
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"✅ Position closed: {position.ticket}")
                else:
                    logger.error(f"❌ Failed to close position: {result.retcode}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error closing positions: {e}")
            return False
    
    def get_account_status(self) -> Dict:
        """دریافت وضعیت حساب"""
        try:
            account_info = mt5.account_info()
            positions = mt5.positions_get(symbol=self.symbol)
            balance = account_info.balance
            equity = account_info.equity
            profit = equity - balance
            
            status = {
                "balance": balance,
                "equity": equity,
                "profit": profit,
                "open_positions": len(positions) if positions else 0,
                "profit_percentage": (profit / balance * 100) if balance > 0 else 0
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error getting account status: {e}")
            return {}
    
    def run_trading_cycle(self) -> Dict:
        """اجرای یک سیکل کامل معاملاتی"""
        try:
            logger.info(f"🔄 Starting trading cycle for {self.symbol}")
            
            # دریافت داده‌های بازار
            data = self.get_market_data(100)
            if data is None or data.empty:
                return {"error": "No market data"}
            
            # تولید سیگنال
            signal = self.strategy.generate_signal(data)
            
            # بررسی پوزیشن‌های باز
            open_positions = mt5.positions_get(symbol=self.symbol)
            has_open_position = len(open_positions) > 0 if open_positions else False
            
            # اگر پوزیشن باز داریم، بررسی شرایط خروج
            if has_open_position:
                logger.info("📊 Checking exit conditions for open position")
                exit_signal = self.strategy.check_exit_conditions(data)
                if exit_signal:
                    logger.info(f"🔄 Exit signal detected: {exit_signal.get('reason', 'Unknown')}")
                    # در اینجا می‌توانید منطق خروج خودکار را اضافه کنید
            
            # اگر پوزیشن باز نداریم و سیگنال معتبر داریم
            if not has_open_position and signal['action'] in ['BUY', 'SHORT']:
                logger.info(f"🎯 Valid signal: {signal['action']}")
                
                # محاسبه استاپ لاس بر اساس ATR
                atr = self.strategy.calculate_atr(data.tail(20))
                stop_loss_pips = int(atr * 10000)  # تبدیل به پیپ
                stop_loss_pips = max(100, min(stop_loss_pips, 500))  # محدودیت 100-500 پیپ
                
                # ارسال سفارش
                order_result = self.place_order(signal, stop_loss_pips)
                signal['order_executed'] = order_result
                signal['stop_loss_pips'] = stop_loss_pips
            else:
                signal['order_executed'] = False
            
            # افزودن وضعیت حساب
            account_status = self.get_account_status()
            signal['account_status'] = account_status
            
            logger.info(f"✅ Trading cycle completed: {signal['action']}")
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error in trading cycle: {e}")
            return {"error": str(e)}
    
    def start_auto_trading(self, interval_minutes: int = 60, max_cycles: int = None):
        """شروع معامله‌گری خودکار"""
        logger.info(f"🚀 Starting auto trading for {self.symbol}")
        logger.info(f"⏰ Check interval: {interval_minutes} minutes")
        logger.info(f"💰 Max risk per trade: {self.max_risk_per_trade*100}%")
        
        cycle_count = 0
        
        try:
            while True:
                if max_cycles and cycle_count >= max_cycles:
                    logger.info("🛑 Max cycles reached")
                    break
                
                cycle_count += 1
                logger.info(f"🔄 Cycle #{cycle_count} - {datetime.now()}")
                
                # اجرای سیکل معاملاتی
                result = self.run_trading_cycle()
                
                # نمایش نتایج
                if 'action' in result:
                    action = result['action']
                    strength = result.get('signal_strength', 'N/A')
                    score = result.get('signal_score', 'N/A')
                    executed = result.get('order_executed', False)
                    
                    status_msg = f"📊 Signal: {action} | Strength: {strength} | Score: {score}"
                    if executed:
                        status_msg += " | ✅ EXECUTED"
                    else:
                        status_msg += " | 🟡 NOT EXECUTED"
                    
                    logger.info(status_msg)
                
                # نمایش وضعیت حساب
                if 'account_status' in result:
                    status = result['account_status']
                    logger.info(f"💰 Balance: {status.get('balance', 0):.2f} | "
                               f"Equity: {status.get('equity', 0):2f} | "
                               f"Profit: {status.get('profit', 0):.2f} "
                               f"({status.get('profit_percentage', 0):.1f}%)")
                
                # انتظار برای سیکل بعدی
                logger.info(f"⏳ Waiting {interval_minutes} minutes for next cycle...")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            logger.info("🛑 Auto trading stopped by user")
        except Exception as e:
            logger.error(f"❌ Auto trading error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """تمیزکاری و بستن اتصال"""
        try:
            mt5.shutdown()
            logger.info("🔌 MT5 connection closed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

# 🎯 تابع اصلی اجرا
def main():
    """تابع اصلی برای اجرای معامله‌گر خودکار"""
    
    # تنظیمات معامله‌گری
    config = {
        "symbol": "XAUUSD",      # نماد معاملاتی
        "timeframe": "H1",       # تایم‌فریم
        "lot_size": 0.1,         # حجم پیش‌فرض
        "max_risk": 0.02,        # حداکثر ریسک 2%
        "interval": 60,          # بررسی هر 60 دقیقه
        "max_cycles": 24         # حداکثر 24 سیکل (24 ساعت)
    }
    
    print("=" * 60)
    print("🚀 MT5 Professional RSI Auto Trader")
    print("=" * 60)
    print(f"📊 Symbol: {config['symbol']}")
    print(f"⏰ Timeframe: {config['timeframe']}")
    print(f"💰 Lot Size: {config['lot_size']}")
    print(f"🎯 Max Risk: {config['max_risk']*100}%")
    print(f"🔄 Check Interval: {config['interval']} minutes")
    print("=" * 60)
    
    # ایجاد معامله‌گر
    trader = MT5AutoTrader(
        symbol=config["symbol"],
        timeframe=config["timeframe"],
        lot_size=config["lot_size"],
        max_risk_per_trade=config["max_risk"]
    )
    
    # شروع معامله‌گری خودکار
    try:
        trader.start_auto_trading(
            interval_minutes=config["interval"],
            max_cycles=config["max_cycles"]
        )
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        trader.cleanup()

if __name__ == "__main__":
    main()