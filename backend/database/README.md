# ORM Configuration for Crypto Trading Bot

## Overview
This project uses SQLAlchemy as the ORM (Object-Relational Mapping) tool to interact with the PostgreSQL database. The ORM provides a Pythonic way to work with database records using Python classes instead of raw SQL queries.

## Components

### 1. Database Configuration (`database/database.py`)
- **Engine**: Creates the connection to the PostgreSQL database with connection pooling
- **Session**: Provides a factory for database sessions with automatic rollback on errors
- **Base**: The declarative base class for all models
- **get_db()**: FastAPI dependency for injecting database sessions into endpoints

### 2. Models (`database/models.py`)
- Contains SQLAlchemy model classes that map to database tables
- Includes relationships between models
- Uses proper field types and constraints

### 3. Repository Pattern (`database/repositories/`)
- **BaseRepository**: Generic repository with common CRUD operations
- **Specific Repositories**: Model-specific repositories with custom query methods
- Provides abstraction between business logic and database operations

### 4. Services (`services/`)
- Business logic layer that uses repositories to perform operations
- Orchestrates complex operations that might involve multiple models
- Decouples business logic from data access logic

## Key Features

### Connection Pooling
- Configured with `pool_size=10` and `max_overflow=20`
- `pool_pre_ping=True` to verify connections before use
- `pool_recycle=3600` to recycle connections after 1 hour

### Error Handling
- Automatic rollback on database errors
- Proper exception handling in repositories
- Logging for debugging purposes

### Security
- Passwords properly encoded in the connection string
- SQL injection protection through SQLAlchemy's query building

## Usage Examples

### In FastAPI Endpoints:
```python
from fastapi import Depends
from database.database import get_db
from services.trading_service import TradingService

@app.post("/trades/")
def create_trade(trade_data: TradeCreate, db: Session = Depends(get_db)):
    service = TradingService(db)
    return service.create_trade(
        trading_session_id=trade_data.session_id,
        symbol=trade_data.symbol,
        side=trade_data.side,
        quantity=trade_data.quantity,
        price=trade_data.price,
        order_id=trade_data.order_id
    )
```

### Direct Repository Usage:
```python
from database.repositories.trade_repository import TradeRepository

def some_function(db_session):
    trade_repo = TradeRepository(db_session)
    trades = trade_repo.get_trades_by_symbol("BTCUSDT")
    return trades
```

## Models Included
- User: User account information
- TradingAccount: Exchange account credentials
- TradingSession: Active trading session data
- Trade: Individual trade records
- MarketData: Historical market price data
- StrategyConfig: Trading strategy configurations
- Portfolio: Portfolio holdings tracking

This ORM setup provides a robust, scalable, and maintainable database layer for the crypto trading bot backend.