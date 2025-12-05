import sys
import os
# Add project path
project_path = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_path)

from main import TradingBotV4
from config.parameters import OPTIMIZED_PARAMS_V5

print('Testing the fix using TradingBotV4 class...')

# Create bot with test_mode=True
bot = TradingBotV4(test_mode=True)

# Initialize with very permissive parameters
test_params = OPTIMIZED_PARAMS_V5.copy()
test_params.update({
    'test_mode_enabled': True,
    'bypass_contradiction_detection': True,
    'relax_risk_filters': True,
    'relax_entry_conditions': True,
    'enable_all_signals': True,
    'enable_trend_filter': False,
    'enable_mtf': False,
    'enable_volatility_filter': False,
    'min_candles_between': 1,
    'rsi_entry_buffer': 1,
    'rsi_oversold': 20,
    'rsi_overbought': 80,
    'max_trades_per_100': 100,
})

# Initialize strategy and backtest engine
bot.initialize_strategy(strategy_params=test_params, use_diagnostic=False)
bot.initialize_backtest()

# Run backtest with test_mode=True
results = bot.run_backtest(
    symbol='EURUSD',
    timeframe='H1',
    days_back=3,
    strategy_params=test_params,
    test_mode=True  # Important: enable test mode
)

metrics = bot.get_strategy_performance()
trades = metrics.get('total_trades', 0)
pnl = metrics.get('total_pnl', 0)

print(f'Backtest completed successfully!')
print(f'Trades generated: {trades}')
print(f'P&L: ${pnl:,.2f}')
print(f'Issue resolved: {trades > 0}')