# mt5_auto_trader.py

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import sys
import os
import threading

# اضافه کردن مسیر پروژه
sys.path.append(os.path.dirname(__file__))

from strategies.adaptive_elite_rsi_strategy import ProfessionalAdvancedRsiStrategy
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

class ProgressIndicator:
    """کلاس برای نمایش وضعیت پیشرفت و فعالیت اسکریپت"""
    
    def __init__(self):
        self.is_running = False
        self.thread = None
        self.current_activity = "Initializing..."
    
    def start(self):
        """شروع نمایش وضعیت"""
        self.is_running = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """توقف نمایش وضعیت"""
        self.is_running = False
        if self.thread:
            self.thread.join()
    
    def update_activity(self, activity: str):
        """بروزرسانی فعالیت جاری"""
        self.current_activity = activity
    
    def _animate(self):
        """انیمیشن نوار پیشرفت"""
        symbols = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷']
        idx = 0
        while self.is_running:
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f'\r{symbols[idx]} {self.current_activity} | 🕒 {current_time}', end='', flush=True)
            idx = (idx + 1) % len(symbols)
            time.sleep(0.1)

class MT5AutoTrader:
    """
    کلاس حرفه‌ای برای اجرای خودکار استراتژی RSI در MetaTrader 5
    """
    
    def __init__(self, 
                 account: int = None, 
                 password: str = None, 
                 server: str = None,
                 symbols: List[str] = None,
                 timeframe: str = "H1",
                 lot_size: float = 0.1,
                 max_risk_per_trade: float = 0.02):
        
        # لیست نمادها
        self.symbols = symbols if symbols else ["XAUUSD"]
        self.current_symbol_index = 0
        self.symbol = self.symbols[0]  # نماد فعلی برای نمایش
        
        self.timeframe = self._get_timeframe(timeframe)
        self.lot_size = lot_size
        self.max_risk_per_trade = max_risk_per_trade
        
        # استراتژی برای هر نماد
        self.strategies = {}
        for symbol in self.symbols:
            self.strategies[symbol] = ProfessionalAdvancedRsiStrategy(
                enable_short_trades=True,
                use_adx_filter=True,
                use_partial_exits=True,
                use_break_even=True,
                min_signal_score=7.0
            )
        
        # نمایشگر وضعیت
        self.progress = ProgressIndicator()
        self.progress.update_activity("Connecting to MT5...")
        self.progress.start()
        
        # اتصال به MT5
        self._connect_mt5(account, password, server)
        
        # اطلاعات حساب
        self.account_info = mt5.account_info()
        self.initial_balance = self.account_info.balance
        
        self.progress.update_activity(f"Trading {len(self.symbols)} symbols on {timeframe}")
        logger.info(f"✅ MT5 AutoTrader initialized for {len(self.symbols)} symbols")
        logger.info(f"💰 Account Balance: {self.initial_balance:.2f}")
        logger.info(f"📊 Timeframe: {timeframe}")
        logger.info(f"📈 Symbols: {', '.join(self.symbols)}")
    
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
            self.progress.update_activity("Initializing MT5 connection...")
            if not mt5.initialize():
                logger.error("❌ Failed to initialize MT5")
                self.progress.update_activity("MT5 connection failed!")
                return False
            
            # اگر اطلاعات لاگین ارائه شده، لاگین کن
            if account and password and server:
                self.progress.update_activity(f"Logging in to account {account}...")
                authorized = mt5.login(account, password=password, server=server)
                if not authorized:
                    logger.error(f"❌ Failed to login to account {account}")
                    self.progress.update_activity("Login failed!")
                    return False
                logger.info(f"✅ Logged in to account {account}")
            else:
                logger.info("✅ Connected to MT5 with default settings")
            
            self.progress.update_activity("MT5 connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            self.progress.update_activity(f"Connection error: {e}")
            return False
    
    def get_market_data(self, symbol: str, count: int = 100) -> Optional[pd.DataFrame]:
        """دریافت داده‌های بازار از MT5"""
        try:
            self.progress.update_activity(f"Fetching market data for {symbol}...")
            
            # اطمینان از انتخاب نماد
            if not mt5.symbol_select(symbol, True):
                logger.error(f"❌ Symbol {symbol} not found")
                self.progress.update_activity(f"Symbol {symbol} not found!")
                return None
            
            # دریافت داده‌های تاریخی
            rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0, count)
            if rates is None:
                logger.error(f"❌ No data received for {symbol}")
                self.progress.update_activity(f"No data for {symbol}!")
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
            
            self.progress.update_activity(f"Market data: {len(df)} candles for {symbol}")
            logger.info(f"📊 Market data retrieved: {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting market data: {e}")
            self.progress.update_activity(f"Data error: {e}")
            return None
    
    def calculate_position_size(self, symbol: str, stop_loss_pips: float) -> float:
        """محاسبه حجم پوزیشن بر اساس مدیریت ریسک"""
        try:
            self.progress.update_activity("Calculating position size...")
            
            # دریافت قیمت فعلی
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return self.lot_size
            
            current_price = tick.bid
            
            # محاسبه ارزش هر پیپ
            symbol_info = mt5.symbol_info(symbol)
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
                
                logger.info(f"📏 Calculated lot size: {calculated_lots:.2f} for {symbol}")
                self.progress.update_activity(f"Position size: {calculated_lots:.2f} lots for {symbol}")
                return round(calculated_lots, 2)
            else:
                return self.lot_size
                
        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            self.progress.update_activity("Position size calculation failed")
            return self.lot_size
    
    def place_order(self, symbol: str, signal: Dict, stop_loss_pips: float = 200) -> bool:
        """ارسال سفارش به MT5"""
        try:
            action = signal['action']
            current_price = signal.get('price', 0)
            
            if action not in ['BUY', 'SHORT']:
                logger.info("🟡 No trade signal")
                return False
            
            self.progress.update_activity(f"Placing {action} order for {symbol}...")
            
            # تعیین نوع سفارش
            order_type = mt5.ORDER_TYPE_BUY if action == 'BUY' else mt5.ORDER_TYPE_SELL
            
            # محاسبه قیمت‌ها
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                logger.error("❌ Cannot get current tick")
                self.progress.update_activity(f"Cannot get current price for {symbol}!")
                return False
            
            price = tick.ask if action == 'BUY' else tick.bid
            point = mt5.symbol_info(symbol).point
            
            # محاسبه استاپ لاس و تیک پروفیت
            if action == 'BUY':
                stop_loss = price - (stop_loss_pips * point)
                take_profit = price + (stop_loss_pips * 2 * point)  # ریسک به ریوارد 1:2
            else:  # SHORT
                stop_loss = price + (stop_loss_pips * point)
                take_profit = price - (stop_loss_pips * 2 * point)
            
            # محاسبه حجم
            volume = self.calculate_position_size(symbol, stop_loss_pips)
            
            # آماده‌سازی درخواست سفارش
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 20,
                "magic": 2023,
                "comment": f"RSI_Pro_{action}_{symbol}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # ارسال سفارش
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"❌ Order failed for {symbol}: {result.retcode}")
                self.progress.update_activity(f"Order failed for {symbol}: {result.retcode}")
                return False
            
            logger.info(f"✅ Order executed: {action} {volume} lots of {symbol} at {price:.4f}")
            logger.info(f"🛡️ SL: {stop_loss:.4f}, 🎯 TP: {take_profit:.4f}")
            self.progress.update_activity(f"✅ {action} order executed for {symbol}!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error placing order: {e}")
            self.progress.update_activity(f"Order error: {e}")
            return False
    
    def close_all_positions(self, symbol: str = None) -> bool:
        """بستن تمام پوزیشن‌های باز"""
        try:
            symbols_to_check = [symbol] if symbol else self.symbols
            closed_positions = 0
            
            for sym in symbols_to_check:
                self.progress.update_activity(f"Closing positions for {sym}...")
                positions = mt5.positions_get(symbol=sym)
                if not positions:
                    continue
                
                for position in positions:
                    # تعیین نوع معکوس برای بستن پوزیشن
                    if position.type == mt5.ORDER_TYPE_BUY:
                        close_type = mt5.ORDER_TYPE_SELL
                        price = mt5.symbol_info_tick(sym).bid
                    else:
                        close_type = mt5.ORDER_TYPE_BUY
                        price = mt5.symbol_info_tick(sym).ask
                    
                    # درخواست بستن پوزیشن
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "position": position.ticket,
                        "symbol": sym,
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
                        logger.info(f"✅ Position closed: {position.ticket} ({sym})")
                        closed_positions += 1
                    else:
                        logger.error(f"❌ Failed to close position: {result.retcode} ({sym})")
            
            if closed_positions > 0:
                self.progress.update_activity(f"Closed {closed_positions} positions")
            else:
                self.progress.update_activity("No positions to close")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error closing positions: {e}")
            self.progress.update_activity("Error closing positions")
            return False
    
    def get_account_status(self) -> Dict:
        """دریافت وضعیت حساب"""
        try:
            self.progress.update_activity("Checking account status...")
            account_info = mt5.account_info()
            positions = mt5.positions_get()
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
            
            self.progress.update_activity(f"Account: {balance:.2f} | Equity: {equity:.2f}")
            return status
            
        except Exception as e:
            logger.error(f"❌ Error getting account status: {e}")
            self.progress.update_activity("Account status error")
            return {}
    
    def run_trading_cycle(self, symbol: str) -> Dict:
        """اجرای یک سیکل کامل معاملاتی برای یک نماد"""
        try:
            logger.info(f"🔄 Starting trading cycle for {symbol}")
            self.progress.update_activity(f"Running trading cycle for {symbol}...")
            
            # دریافت داده‌های بازار
            data = self.get_market_data(symbol, 100)
            if data is None or data.empty:
                self.progress.update_activity(f"No market data available for {symbol}")
                return {"error": "No market data", "symbol": symbol}
            
            # تولید سیگنال
            self.progress.update_activity(f"Generating trading signal for {symbol}...")
            signal = self.strategies[symbol].generate_signal(data)
            
            # بررسی پوزیشن‌های باز
            open_positions = mt5.positions_get(symbol=symbol)
            has_open_position = len(open_positions) > 0 if open_positions else False
            
            # اگر پوزیشن باز داریم، بررسی شرایط خروج
            if has_open_position:
                logger.info(f"📊 Checking exit conditions for open position in {symbol}")
                self.progress.update_activity(f"Checking exit conditions for {symbol}...")
                exit_signal = self.strategies[symbol].check_exit_conditions(data)
                if exit_signal:
                    logger.info(f"🔄 Exit signal detected for {symbol}: {exit_signal.get('reason', 'Unknown')}")
                    self.progress.update_activity(f"Exit signal for {symbol}: {exit_signal.get('reason', 'Unknown')}")
                    # در اینجا می‌توانید منطق خروج خودکار را اضافه کنید
            
            # اگر پوزیشن باز نداریم و سیگنال معتبر داریم
            if not has_open_position and signal['action'] in ['BUY', 'SHORT']:
                logger.info(f"🎯 Valid signal for {symbol}: {signal['action']}")
                self.progress.update_activity(f"Valid {signal['action']} signal detected for {symbol}")
                
                # محاسبه استاپ لاس بر اساس ATR
                atr = self.strategies[symbol].calculate_atr(data.tail(20))
                stop_loss_pips = int(atr * 10000)  # تبدیل به پیپ
                stop_loss_pips = max(100, min(stop_loss_pips, 500))  # محدودیت 100-500 پیپ
                
                # ارسال سفارش
                order_result = self.place_order(symbol, signal, stop_loss_pips)
                signal['order_executed'] = order_result
                signal['stop_loss_pips'] = stop_loss_pips
            else:
                signal['order_executed'] = False
                if has_open_position:
                    self.progress.update_activity(f"Position already open for {symbol} - waiting")
                else:
                    self.progress.update_activity(f"No valid signal for {symbol} - waiting")
            
            # افزودن وضعیت حساب
            account_status = self.get_account_status()
            signal['account_status'] = account_status
            signal['symbol'] = symbol
            
            logger.info(f"✅ Trading cycle completed for {symbol}: {signal['action']}")
            self.progress.update_activity(f"Cycle completed for {symbol}: {signal['action']}")
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error in trading cycle for {symbol}: {e}")
            self.progress.update_activity(f"Trading cycle error for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    def start_auto_trading(self, interval_minutes: int = 5, max_cycles: int = None):
        """شروع معامله‌گری خودکار برای چند نماد"""
        logger.info(f"🚀 Starting auto trading for {len(self.symbols)} symbols")
        logger.info(f"⏰ Check interval: {interval_minutes} minutes")
        logger.info(f"💰 Max risk per trade: {self.max_risk_per_trade*100}%")
        
        self.progress.update_activity(f"Auto trading started - {interval_minutes}min intervals")
        
        cycle_count = 0
        
        try:
            while True:
                if max_cycles and cycle_count >= max_cycles:
                    logger.info("🛑 Max cycles reached")
                    self.progress.update_activity("Max cycles reached - stopping")
                    break
                
                cycle_count += 1
                logger.info(f"🔄 Cycle #{cycle_count} - {datetime.now()}")
                self.progress.update_activity(f"Cycle #{cycle_count} - Processing...")
                
                # اجرای سیکل معاملاتی برای هر نماد
                all_results = []
                for symbol in self.symbols:
                    result = self.run_trading_cycle(symbol)
                    all_results.append(result)
                    
                    # نمایش نتایج
                    if 'action' in result:
                        action = result['action']
                        strength = result.get('signal_strength', 'N/A')
                        score = result.get('signal_score', 'N/A')
                        executed = result.get('order_executed', False)
                        
                        status_msg = f"📊 {symbol}: {action} | Strength: {strength} | Score: {score}"
                        if executed:
                            status_msg += " | ✅ EXECUTED"
                        else:
                            status_msg += " | 🟡 NOT EXECUTED"
                        
                        logger.info(status_msg)
                
                # نمایش وضعیت حساب
                if all_results and 'account_status' in all_results[0]:
                    status = all_results[0]['account_status']
                    logger.info(f"💰 Balance: {status.get('balance', 0):.2f} | "
                               f"Equity: {status.get('equity', 0):.2f} | "
                               f"Profit: {status.get('profit', 0):.2f} "
                               f"({status.get('profit_percentage', 0):.1f}%)")
                
                # نمایش زمان باقی‌مانده تا سیکل بعدی
                for i in range(interval_minutes * 60, 0, -1):
                    minutes, seconds = divmod(i, 60)
                    self.progress.update_activity(
                        f"Next cycle in: {minutes:02d}:{seconds:02d} | "
                        f"Cycle: {cycle_count}/{max_cycles if max_cycles else '∞'} | "
                        f"Symbols: {len(self.symbols)}"
                    )
                    time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("🛑 Auto trading stopped by user")
            self.progress.update_activity("Stopped by user")
        except Exception as e:
            logger.error(f"❌ Auto trading error: {e}")
            self.progress.update_activity(f"Auto trading error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """تمیزکاری و بستن اتصال"""
        try:
            self.progress.update_activity("Shutting down...")
            self.progress.stop()
            mt5.shutdown()
            print("\r✅ MT5 connection closed" + " " * 50)  # پاک کردن خط آخر
            logger.info("🔌 MT5 connection closed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

# 🎯 تابع اصلی اجرا
def main():
    """تابع اصلی برای اجرای معامله‌گر خودکار"""
    
    # لیست نمادهای معاملاتی
    trading_symbols = [
        "XAUUSD",   # طلا
        "XAGUSD",   # نقره
        "EURUSD",   # یورو/دلار
        "GBPUSD",   # پوند/دلار
        "USDJPY",   # دلار/ین
        "USDCHF",   # دلار/فرانک
        "USDCAD",   # دلار/دلار کانادا
        "AUDUSD",   # دلار استرالیا/دلار
        "EURJPY",   # یورو/ین
        "BTCUSD",   # بیت‌کوین
        "ETHUSD",   # اتریوم
        "XTIUSD",   # نفت
        "US30",     # شاخص داوجونز
        "NAS100",   # ناسداک
        "SPX500"    # S&P 500
    ]
    
    # تنظیمات معامله‌گری
    config = {
        "symbols": trading_symbols,  # لیست نمادهای معاملاتی
        "timeframe": "H1",          # تایم‌فریم
        "lot_size": 0.1,            # حجم پیش‌فرض
        "max_risk": 0.02,           # حداکثر ریسک 2%
        "interval": 5,              # بررسی هر 5 دقیقه
        "max_cycles": 24            # حداکثر 24 سیکل (2 ساعت)
    }
    
    print("=" * 60)
    print("🚀 MT5 Professional RSI Auto Trader")
    print("=" * 60)
    print(f"📊 Symbols: {len(config['symbols'])} instruments")
    print(f"⏰ Timeframe: {config['timeframe']}")
    print(f"💰 Lot Size: {config['lot_size']}")
    print(f"🎯 Max Risk: {config['max_risk']*100}%")
    print(f"🔄 Check Interval: {config['interval']} minutes")
    print("=" * 60)
    print("📈 Live Progress:")
    print("-" * 60)
    
    # ایجاد معامله‌گر
    trader = MT5AutoTrader(
        symbols=config["symbols"],
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
        print(f"\r❌ Error: {e}" + " " * 50)
    finally:
        trader.cleanup()

if __name__ == "__main__":
    main()