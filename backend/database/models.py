from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database.database import Base


class User(Base):
    """
    User model for storing user information
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship to trading accounts
    trading_accounts = relationship("TradingAccount", back_populates="user")


class TradingAccount(Base):
    """
    Trading account model for storing exchange account information
    """
    __tablename__ = "trading_accounts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    account_name = Column(String, index=True)
    exchange_name = Column(String, index=True)  # e.g., Binance, Coinbase, etc.
    api_key = Column(String)
    api_secret = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship to user
    user = relationship("User", back_populates="trading_accounts")
    # Relationship to trading sessions
    trading_sessions = relationship("TradingSession", back_populates="trading_account")


class TradingSession(Base):
    """
    Trading session model for storing active trading session information
    """
    __tablename__ = "trading_sessions"

    id = Column(Integer, primary_key=True, index=True)
    trading_account_id = Column(Integer, ForeignKey("trading_accounts.id"))
    strategy_name = Column(String, index=True)
    is_active = Column(Boolean, default=True)
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True), nullable=True)
    initial_balance = Column(Float)
    current_balance = Column(Float)
    profit_loss = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship to trading account
    trading_account = relationship("TradingAccount", back_populates="trading_sessions")
    # Relationship to trades
    trades = relationship("Trade", back_populates="trading_session")


class Trade(Base):
    """
    Trade model for storing individual trade records
    """
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    trading_session_id = Column(Integer, ForeignKey("trading_sessions.id"))
    symbol = Column(String, index=True)
    side = Column(String)  # 'buy' or 'sell'
    quantity = Column(Float)
    price = Column(Float)
    fee = Column(Float, default=0.0)
    status = Column(String, default="completed")  # pending, completed, cancelled
    order_id = Column(String, index=True)  # Exchange order ID
    executed_at = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship to trading session
    trading_session = relationship("TradingSession", back_populates="trades")


class MarketData(Base):
    """
    Market data model for storing historical price data
    """
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    timeframe = Column(String, index=True)  # 1m, 5m, 15m, 1h, 4h, 1d, etc.
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    timestamp = Column(DateTime(timezone=True), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class StrategyConfig(Base):
    """
    Strategy configuration model for storing strategy parameters
    """
    __tablename__ = "strategy_configs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text)
    parameters = Column(Text)  # JSON string of strategy parameters
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Portfolio(Base):
    """
    Portfolio model for tracking portfolio holdings
    """
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String, index=True)
    quantity = Column(Float)
    avg_buy_price = Column(Float)
    current_price = Column(Float)
    value = Column(Float)  # quantity * current_price
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
