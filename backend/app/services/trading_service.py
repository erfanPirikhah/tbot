import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

from lib.data.data_fetcher import DataFetcher
from lib.strategies.enhanced_rsi_strategy_v5 import EnhancedRsiStrategyV5, PositionType
from lib.config.parameters import LIVE_TRADING_PARAMS

logger = logging.getLogger("TradingService")

class TradingService:
    def __init__(self):
        self.is_running = False
        self.strategy = None
        self.data_fetcher = DataFetcher()
        self.current_position = "OUT"
        self.portfolio_value = 10000.0
        self.active_symbols = []
        self.start_time = None
        self.trade_history = []
        self.open_positions = {}
        self._trading_task = None
        
    async def start_trading(self, symbol: str, timeframe: str, strategy_params: Optional[Dict] = None, duration_hours: Optional[int] = None):
        """Start live trading"""
        if self.is_running:
            raise ValueError("Trading is already running")
        
        try:
            # Use LIVE_TRADING_PARAMS as base, then override with any custom params
            params = LIVE_TRADING_PARAMS.copy()
            if strategy_params:
                params.update(strategy_params)
            
            logger.info(f"🚀 Starting live trading with LIVE_TRADING_PARAMS")
            logger.info(f"   RSI thresholds: {params.get('rsi_oversold')}/{params.get('rsi_overbought')}")
            logger.info(f"   Filters: MTF={params.get('enable_mtf')}, Trend={params.get('enable_trend_filter')}")
            logger.info(f"   Test Mode: {params.get('test_mode_enabled')}")
            
            self.strategy = EnhancedRsiStrategyV5(**params)
            self.active_symbols = [symbol]
            self.is_running = True
            self.start_time = datetime.now()
            
            # Start trading loop in background
            self._trading_task = asyncio.create_task(
                self._trading_loop(symbol, timeframe, duration_hours)
            )
            
            logger.info(f"Trading started for {symbol} on {timeframe}")
            return {"status": "started", "symbol": symbol, "timeframe": timeframe}
            
        except Exception as e:
            logger.error(f"Error starting trading: {e}")
            self.is_running = False
            raise
    
    async def stop_trading(self):
        """Stop live trading"""
        if not self.is_running:
            raise ValueError("Trading is not running")
        
        self.is_running = False
        if self._trading_task:
            self._trading_task.cancel()
            try:
                await self._trading_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Trading stopped")
        return {"status": "stopped"}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current trading status"""
        uptime = None
        if self.start_time:
            uptime = int((datetime.now() - self.start_time).total_seconds())
        
        return {
            "is_running": self.is_running,
            "current_position": self.current_position,
            "portfolio_value": self.portfolio_value,
            "active_symbols": self.active_symbols,
            "uptime_seconds": uptime
        }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get open positions"""
        return list(self.open_positions.values())
    
    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history"""
        return self.trade_history[-limit:]
    
    async def _trading_loop(self, symbol: str, timeframe: str, duration_hours: Optional[int]):
        """Main trading loop"""
        iteration = 0
        signals_generated = {"BUY": 0, "SELL": 0, "HOLD": 0, "EXIT": 0}
        
        try:
            while self.is_running:
                iteration += 1
                
                # Fetch data - ML needs 200+ candles
                logger.info(f"📊 Iteration {iteration}: Fetching data for {symbol}...")
                data = self.data_fetcher.fetch_market_data(symbol, timeframe, limit=300)
                
                if data.empty:
                    logger.warning(f"⚠️ No data received for {symbol}")
                    await asyncio.sleep(60)
                    continue
                
                logger.info(f"✅ Got {len(data)} candles. Price: {data['close'].iloc[-1]:.5f}")
                
                # Generate signal
                signal = self.strategy.generate_signal(data, iteration)
                action = signal.get('action', 'HOLD')
                signals_generated[action] = signals_generated.get(action, 0) + 1
                
                # Log every signal for debugging
                logger.info(f"📈 Signal: {action} | Reason: {signal.get('reason', 'N/A')[:100]}")
                logger.info(f"📊 Signal Stats: {signals_generated}")
                
                if action in ['BUY', 'SELL']:
                    logger.info(f"🎯 TRADE SIGNAL! {action} @ {data['close'].iloc[-1]:.5f}")
                    logger.info(f"   SL: {signal.get('stop_loss', 'N/A')}, TP: {signal.get('take_profit', 'N/A')}")
                    logger.info(f"   Size: {signal.get('position_size', 'N/A')}")
                
                # Process signal
                await self._process_signal(signal, data, symbol)
                
                # Dynamic sleep based on timeframe
                sleep_time = 60  # Default 1 minute for 1m timeframe
                if timeframe == "5m":
                    sleep_time = 60 * 2
                elif timeframe == "15m":
                    sleep_time = 60 * 5
                elif timeframe in ["1h", "H1"]:
                    sleep_time = 60 * 15
                
                logger.info(f"💤 Sleeping {sleep_time}s until next check...")
                await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.is_running = False
    
    async def _process_signal(self, signal: Dict[str, Any], data: pd.DataFrame, symbol: str):
        """Process trading signal"""
        action = signal.get('action', 'HOLD')
        current_price = data['close'].iloc[-1]
        
        trade_record = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "price": current_price
        }
        
        if action in ['BUY', 'SELL']:
            self.current_position = "LONG" if action == "BUY" else "SHORT"
            
            # Create position
            self.open_positions[symbol] = {
                "symbol": symbol,
                "side": self.current_position,
                "entry_price": current_price,
                "current_price": current_price,
                "quantity": signal.get('position_size', 1.0),
                "pnl_percentage": 0.0,
                "pnl_amount": 0.0,
                "stop_loss": signal.get('stop_loss', current_price * 0.98),
                "take_profit": signal.get('take_profit', current_price * 1.02)
            }
            
            trade_record.update({
                "side": self.current_position,
                "action": "ENTRY",
                "quantity": signal.get('position_size'),
                "reason": signal.get('reason', '')
            })
            
        elif action == 'EXIT':
            if symbol in self.open_positions:
                pos = self.open_positions.pop(symbol)
                pnl_pct = signal.get('pnl_percentage', 0)
                
                trade_record.update({
                    "side": pos['side'],
                    "action": "EXIT",
                    "pnl_percentage": pnl_pct,
                    "pnl_amount": signal.get('pnl_amount', 0),
                    "reason": signal.get('exit_reason', '')
                })
                
                self.current_position = "OUT"
        
        # Update history
        if action != 'HOLD':
            self.trade_history.append(trade_record)

# Global instance
trading_service = TradingService()
