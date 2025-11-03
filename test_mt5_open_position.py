import MetaTrader5 as mt5

SYMBOL = "XAUUSD"  # یا EURUSD / BTCUSD
VOLUME = 0.01

print("🚀 Connecting to MT5...")
mt5.initialize()

info = mt5.symbol_info(SYMBOL)
if not info:
    print(f"❌ Symbol {SYMBOL} not found")
    mt5.shutdown()
    exit()

print(f"✅ Symbol: {SYMBOL}")
print(f"Trade mode: {info.trade_mode}")
print(f"Filling mode: {info.filling_mode}")

# قیمت جاری
tick = mt5.symbol_info_tick(SYMBOL)
price = tick.ask

# انتخاب حالت مجاز پر کردن سفارش
filling = info.filling_mode if info.filling_mode in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN] else mt5.ORDER_FILLING_RETURN

# درخواست معامله
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": SYMBOL,
    "volume": VOLUME,
    "type": mt5.ORDER_TYPE_BUY,
    "price": price,
    "deviation": 20,
    "magic": 1001,
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": filling,
}

print("\n📤 Sending order...")
result = mt5.order_send(request)
print(result)

mt5.shutdown()
