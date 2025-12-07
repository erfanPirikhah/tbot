# Trading Bot API Backend

FastAPI-based backend for the Intelligent Crypto Trading Bot.

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Run Server**:
   ```bash
   python backend/run.py
   ```

3. **API Documentation**: `http://localhost:8000/docs`

## API Endpoints

### System
- `GET /api/system/health` - Health check

### Market Data
- `GET /api/market/analysis/{symbol}` - Market analysis (price, regime, direction)
- `GET /api/market/ohlcv/{symbol}` - OHLCV data
- `GET /api/market/indicators/{symbol}` - Technical indicators

### Backtesting
- `POST /api/backtest/run` - Start backtest
- `GET /api/backtest/{id}/results` - Get results
- `GET /api/backtest/{id}/equity-curve` - Equity curve

### Live Trading
- `POST /api/trading/start` - Start live trading
- `POST /api/trading/stop` - Stop trading
- `GET /api/trading/status` - Trading status
- `GET /api/trading/positions` - Open positions
- `GET /api/trading/history` - Trade history

### Strategy Management
- `GET /api/strategies/list` - List strategies
- `GET /api/strategies/ml-status` - ML model status
- `POST /api/strategies/configure` - Configure strategy
- `GET /api/strategies/parameters` - Get parameters
- `POST /api/strategies/ml/retrain` - Retrain ML model

### Reports
- `GET /api/reports/performance` - Performance metrics
- `GET /api/reports/trades` - Trade history (filtered)
- `GET /api/reports/daily-stats` - Daily statistics

### Configuration
- `GET /api/config/symbols` - Available symbols
- `GET /api/config/timeframes` - Supported timeframes
- `GET /api/config/risk` - Risk configuration
- `PUT /api/config/risk` - Update risk settings
