# Data Storage Architecture for Algorithmic Trading System

## Current State Analysis (Log-based Storage)

### Existing Data Persistence Approach
The current system relies primarily on log-based storage with data being maintained only in RAM during runtime and limited logging to MongoDB for basic events. Critical trading entities are lost on system restarts, creating significant operational and financial risks.

### Data Currently Only Being Logged
1. **Trade executions** - Only basic logging without persistent storage of position details
2. **Performance metrics** - Calculated on-demand from in-memory data
3. **Error and system logs** - Basic operational logging only
4. **Bot heartbeat events** - Runtime monitoring without persistent state tracking

### Data Not Being Stored at All
1. **Open positions** - Critical financial data stored only in RAM
2. **Complete trade history** - Historical data lost on restarts
3. **Bot runtime state** - Bot status and configuration lost on restart
4. **Strategy parameter history** - No versioning of strategy configurations
5. **Risk management state** - Daily/weekly drawdown limits not persisted
6. **Decision traces** - No storage of strategy decisions and signals

## Data Gaps & Risks

### Financial Risks
- **Position Loss**: Open positions are lost on system restart, potentially causing significant financial losses
- **No P&L Tracking**: Historical profit/loss data unavailable for analysis
- **Order State Loss**: Exchange order states not persisted, leading to potential conflicts

### Operational Risks
- **Service Continuity**: Bot cannot resume operations automatically after restart
- **Configuration Loss**: Strategy and risk management parameters lost on restart
- **Manual Recovery**: Requires manual intervention to restore trading state
- **No Failover**: No mechanism to restore operations on backup systems

### Compliance & Auditability Risks
- **No Audit Trail**: Complete lack of transaction audit capability
- **Regulatory Compliance**: Insufficient data for compliance reporting
- **Tax Reporting**: Missing trade history for tax calculations
- **No Data Governance**: No systematic tracking of trading decisions

### Performance & Analytics Risks
- **No Historical Analysis**: Inability to analyze strategy performance over time
- **No A/B Testing**: Cannot compare different strategy versions
- **No Risk Metrics**: Unable to track system-level risk metrics
- **Limited Backtesting**: No persistent backtest result storage

## Target Storage Architecture

### Technology Stack
- **Primary Database**: PostgreSQL 14+ for structured, relational data storage
- **Time Series Data**: PostgreSQL with TimescaleDB extension for high-frequency metrics
- **Caching Layer**: Redis for real-time state and performance caching
- **Event Storage**: PostgreSQL for audit trails and event sourcing

### Architecture Principles
1. **ACID Compliance**: Ensure data integrity for all trading operations
2. **High Performance**: Optimize for high-frequency trading operations
3. **Auditability**: Maintain complete audit trails for all operations
4. **Scalability**: Support for horizontal and vertical scaling
5. **Resilience**: Built-in redundancy and failover capabilities

### Storage Categories
1. **State Storage**: Persistent state that must survive restarts
2. **Event Storage**: Immutable events for audit and replay capability
3. **Historical Storage**: Long-term storage for analytics and reporting
4. **Cache Storage**: High-frequency data for performance optimization

## Detailed Data Models (Tables/Entities)

### Core Trading Entities

#### 1. `trades` table - Trade Execution Records
**Purpose**: Store all executed trades with complete execution details
**Type**: High-frequency, Historical
**Usage**: Live Trading, Paper Trading, Backtesting

```sql
CREATE TABLE trades (
    id BIGSERIAL PRIMARY KEY,
    trade_id UUID UNIQUE NOT NULL,
    bot_id VARCHAR(50) NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    position_id UUID REFERENCES positions(position_id) ON DELETE SET NULL,
    
    -- Trading details
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    action VARCHAR(20) NOT NULL CHECK (action IN ('ENTRY', 'PARTIAL_EXIT', 'FULL_EXIT', 'ADJUSTMENT')),
    
    -- Execution details
    order_id VARCHAR(100) NOT NULL, -- Exchange order ID
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP', 'TAKE_PROFIT', 'STOP_LOSS')),
    price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    cost DECIMAL(20, 8) NOT NULL, -- price * quantity
    fees DECIMAL(20, 8) NOT NULL DEFAULT 0,
    
    -- Performance metrics
    pnl DECIMAL(20, 8) DEFAULT NULL, -- Realized P&L for exits only
    pnl_pct DECIMAL(8, 4) DEFAULT NULL, -- Realized P&L percentage
    
    -- Execution timestamps
    exchange_timestamp TIMESTAMP WITH TIME ZONE NOT NULL, -- Exchange reported time
    local_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(), -- Local processing time
    
    -- Status tracking
    status VARCHAR(20) NOT NULL CHECK (status IN ('FILLED', 'PARTIALLY_FILLED', 'CANCELLED', 'REJECTED')),
    
    -- Context
    reason VARCHAR(100), -- Reason for trade (signal, SL, TP, manual)
    slippage DECIMAL(8, 4), -- Difference between expected and actual price
    
    -- Versioning and audit
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    version INTEGER NOT NULL DEFAULT 1,
    
    -- Metadata
    metadata JSONB -- Flexible additional data (exchange-specific info, risk metrics)
);
```

**Indexing Strategy**:
- `CREATE INDEX idx_trades_bot_timestamp ON trades(bot_id, local_timestamp DESC);`
- `CREATE INDEX idx_trades_symbol_timestamp ON trades(symbol, local_timestamp DESC);`
- `CREATE INDEX idx_trades_position_id ON trades(position_id);`
- `CREATE INDEX idx_trades_strategy_id ON trades(strategy_id);`
- `CREATE INDEX idx_trades_order_id ON trades(order_id);`

**Partitioning**: Partition by date (monthly) for high-volume scenarios

#### 2. `positions` table - Open Position Management  
**Purpose**: Track all positions (open, closing, closed) with complete lifecycle
**Type**: Medium-frequency, Real-time + Historical
**Usage**: Live Trading, Paper Trading, Backtesting

```sql
CREATE TABLE positions (
    id BIGSERIAL PRIMARY KEY,
    position_id UUID UNIQUE NOT NULL,
    bot_id VARCHAR(50) NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    
    -- Basic position data
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    entry_price DECIMAL(20, 8) NOT NULL,
    avg_entry_price DECIMAL(20, 8), -- For multiple entries
    quantity DECIMAL(20, 8) NOT NULL,
    
    -- Current state
    current_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
    unrealized_pnl_pct DECIMAL(8, 4) DEFAULT 0,
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    
    -- Risk management
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    trailing_stop_active BOOLEAN DEFAULT FALSE,
    trailing_stop_price DECIMAL(20, 8),
    liquidation_price DECIMAL(20, 8), -- For leveraged positions
    leverage INTEGER DEFAULT 1,
    margin_used DECIMAL(20, 8),
    
    -- Position status
    status VARCHAR(20) NOT NULL CHECK (status IN ('ACTIVE', 'CLOSING', 'CLOSED', 'LIQUIDATED')) DEFAULT 'ACTIVE',
    
    -- Timestamps
    entry_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    close_timestamp TIMESTAMP WITH TIME ZONE,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Concurrency control
    version INTEGER NOT NULL DEFAULT 1,
    
    -- Metadata
    metadata JSONB
);
```

**Indexing Strategy**:
- `CREATE INDEX idx_positions_bot_status ON positions(bot_id, status);`
- `CREATE INDEX idx_positions_symbol_status ON positions(symbol, status);`
- `CREATE INDEX idx_positions_strategy_status ON positions(strategy_id, status);`
- `CREATE INDEX idx_positions_status_timestamp ON positions(status, entry_timestamp DESC);`

#### 3. `bots` table - Bot Runtime State
**Purpose**: Store bot configuration and runtime state that survives restarts
**Type**: Low-frequency, State
**Usage**: Live Trading, Paper Trading

```sql
CREATE TABLE bots (
    id BIGSERIAL PRIMARY KEY,
    bot_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    
    -- Bot configuration
    bot_type VARCHAR(20) NOT NULL CHECK (bot_type IN ('LIVE', 'PAPER', 'BACKTEST')),
    mode VARCHAR(20) NOT NULL CHECK (mode IN ('AUTOMATED', 'MANUAL', 'LIMITED')) DEFAULT 'AUTOMATED',
    exchange VARCHAR(50) NOT NULL,
    
    -- Runtime state
    status VARCHAR(20) NOT NULL CHECK (status IN ('RUNNING', 'PAUSED', 'STOPPED', 'ERROR', 'MAINTENANCE')) DEFAULT 'STOPPED',
    start_time TIMESTAMP WITH TIME ZONE,
    stop_time TIMESTAMP WITH TIME ZONE,
    last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    uptime_seconds BIGINT DEFAULT 0,
    
    -- Portfolio state
    current_position VARCHAR(10) CHECK (current_position IN ('LONG', 'SHORT', 'NEUTRAL')),
    portfolio_value DECIMAL(20, 8),
    initial_capital DECIMAL(20, 8),
    
    -- Active settings
    active_symbols TEXT[], -- Array of symbols bot is trading
    strategy_config JSONB, -- Active strategy parameters
    risk_config JSONB, -- Active risk parameters
    
    -- Performance metrics (real-time)
    performance_metrics JSONB,
    
    -- Error tracking
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    last_error_time TIMESTAMP WITH TIME ZONE,
    
    -- Audit trail
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    version INTEGER NOT NULL DEFAULT 1,
    
    -- Metadata
    metadata JSONB
);
```

**Indexing Strategy**:
- `CREATE INDEX idx_bots_status ON bots(status);`
- `CREATE INDEX idx_bots_type_status ON bots(bot_type, status);`
- `CREATE INDEX idx_bots_heartbeat ON bots(last_heartbeat);`
- `CREATE INDEX idx_bots_exchange ON bots(exchange);`

#### 4. `bot_configs` table - Configuration Versioning
**Purpose**: Store versioned bot configurations for audit and rollback
**Type**: Low-frequency, Historical
**Usage**: Live Trading, Paper Trading

```sql
CREATE TABLE bot_configs (
    id BIGSERIAL PRIMARY KEY,
    config_id UUID UNIQUE NOT NULL,
    bot_id VARCHAR(50) NOT NULL REFERENCES bots(bot_id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    
    -- Configuration sections
    strategy_params JSONB,
    risk_params JSONB,
    trading_params JSONB,
    exchange_params JSONB,
    
    -- State tracking
    active BOOLEAN NOT NULL DEFAULT FALSE, -- Only one config per bot can be active
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(50) NOT NULL,
    reason TEXT,
    applied_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    metadata JSONB
);
```

**Indexing Strategy**:
- `CREATE INDEX idx_bot_configs_bot_id ON bot_configs(bot_id);`
- `CREATE INDEX idx_bot_configs_bot_version ON bot_configs(bot_id, version_number DESC);`
- `CREATE INDEX idx_bot_configs_active ON bot_configs(bot_id, active);`

#### 5. `strategies` table - Strategy Configuration Management
**Purpose**: Store strategy definitions and parameters with versioning
**Type**: Low-frequency, State
**Usage**: All environments

```sql
CREATE TABLE strategies (
    id BIGSERIAL PRIMARY KEY,
    strategy_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    class_name VARCHAR(100) NOT NULL,
    description TEXT,
    
    -- Strategy configuration
    parameters JSONB NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    version VARCHAR(20) NOT NULL, -- Semantic versioning
    
    -- Strategy metadata
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH')),
    target_symbols TEXT[],
    timeframes TEXT[], -- e.g., ['1m', '5m', '1h', '4h']
    tags TEXT[],
    
    -- Performance metrics
    backtest_results JSONB,
    
    -- Audit
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(50) NOT NULL,
    
    -- Metadata
    metadata JSONB
);
```

**Indexing Strategy**:
- `CREATE INDEX idx_strategies_active ON strategies(is_active, risk_level);`
- `CREATE INDEX idx_strategies_name ON strategies(name);`
- `CREATE INDEX idx_strategies_created_by ON strategies(created_by);`
- `CREATE INDEX idx_strategies_tags ON strategies USING GIN(tags);`

### Decision & Strategy Traceability

#### 6. `strategy_decisions` table - Decision Log
**Purpose**: Store all strategy decisions and signals for auditability
**Type**: High-frequency, Event
**Usage**: All environments

```sql
CREATE TABLE strategy_decisions (
    id BIGSERIAL PRIMARY KEY,
    decision_id UUID UNIQUE NOT NULL,
    bot_id VARCHAR(50) NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    
    -- Decision context
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    signal_type VARCHAR(20) NOT NULL CHECK (signal_type IN ('BUY', 'SELL', 'HOLD', 'EXIT', 'ADJUST')),
    
    -- Decision data
    decision_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    decision_price DECIMAL(20, 8) NOT NULL,
    decision_reason TEXT,
    confidence_score DECIMAL(5, 4), -- 0.0000 to 1.0000
    
    -- Input data snapshot
    market_data JSONB, -- Current market conditions
    indicator_values JSONB, -- Current indicator values
    regime_state JSONB, -- Market regime detector output
    trend_filter_result JSONB, -- Trend filter output
    contradiction_analysis JSONB, -- Contradiction detection results
    
    -- Execution reference
    executed_trade_id UUID REFERENCES trades(trade_id) ON DELETE SET NULL,
    position_impact VARCHAR(50), -- How this decision affected position
    
    -- Metadata
    metadata JSONB
);
```

**Indexing Strategy**:
- `CREATE INDEX idx_decision_bot_timestamp ON strategy_decisions(bot_id, decision_timestamp DESC);`
- `CREATE INDEX idx_decision_strategy_timestamp ON strategy_decisions(strategy_id, decision_timestamp DESC);`
- `CREATE INDEX idx_decision_signal_type ON strategy_decisions(signal_type);`

#### 7. `indicator_snapshots` table - Indicator State Storage
**Purpose**: Store indicator values at decision points for analysis
**Type**: High-frequency, Historical
**Usage**: All environments

```sql
CREATE TABLE indicator_snapshots (
    id BIGSERIAL PRIMARY KEY,
    snapshot_id UUID UNIQUE NOT NULL,
    bot_id VARCHAR(50) NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    
    -- Time context
    timeframe VARCHAR(10) NOT NULL,
    kline_timestamp TIMESTAMP WITH TIME ZONE NOT NULL, -- When kline closed
    
    -- Indicator values
    rsi_value DECIMAL(8, 4),
    moving_averages JSONB, -- Different MA values
    bollinger_bands JSONB, -- Upper, middle, lower
    macd_values JSONB, -- MACD line, signal line, histogram
    atr_value DECIMAL(20, 8),
    volatility DECIMAL(8, 4),
    volume_indicators JSONB,
    
    -- Multi-timeframe data
    mtf_data JSONB, -- Data from other timeframes
    
    -- Market regime
    regime_type VARCHAR(20), -- 'TRENDING', 'RANGING', 'VOLATILE', 'CALM'
    regime_confidence DECIMAL(5, 4),
    
    -- Creation timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Metadata
    metadata JSONB
);
```

**Indexing Strategy**:
- `CREATE INDEX idx_indicator_bot_symbol_time ON indicator_snapshots(bot_id, symbol, kline_timestamp DESC);`
- `CREATE INDEX idx_indicator_strategy_time ON indicator_snapshots(strategy_id, kline_timestamp DESC);`

### Risk & Governance Data

#### 8. `risk_metrics` table - Risk Data Storage
**Purpose**: Store per-trade and system-level risk metrics
**Type**: Medium-frequency, State + Historical
**Usage**: Live Trading, Paper Trading

```sql
CREATE TABLE risk_metrics (
    id BIGSERIAL PRIMARY KEY,
    risk_id UUID UNIQUE NOT NULL,
    bot_id VARCHAR(50) NOT NULL,
    strategy_id VARCHAR(50),
    
    -- Trade-specific metrics
    trade_id UUID REFERENCES trades(trade_id) ON DELETE SET NULL,
    position_id UUID REFERENCES positions(position_id) ON DELETE SET NULL,
    
    -- Risk measurements
    volatility_at_entry DECIMAL(8, 4),
    atr_at_entry DECIMAL(20, 8),
    position_risk DECIMAL(20, 8), -- Risk amount in quote currency
    risk_percentage DECIMAL(5, 4), -- Risk as % of portfolio
    
    -- Correlation data
    correlation_with_portfolio DECIMAL(5, 4),
    market_regime VARCHAR(20),
    
    -- System risk
    system_exposure DECIMAL(20, 8), -- Total system exposure
    system_correlation_risk DECIMAL(5, 4),
    
    -- Timestamps
    measurement_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    period_start TIMESTAMP WITH TIME ZONE,
    period_end TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    metadata JSONB
);
```

**Indexing Strategy**:
- `CREATE INDEX idx_risk_bot_timestamp ON risk_metrics(bot_id, measurement_timestamp DESC);`
- `CREATE INDEX idx_risk_position_id ON risk_metrics(position_id);`
- `CREATE INDEX idx_risk_trade_id ON risk_metrics(trade_id);`

#### 9. `risk_limits` table - Risk Configuration and Tracking
**Purpose**: Store risk limits and current usage for governance
**Type**: Low-frequency, State
**Usage**: Live Trading, Paper Trading

```sql
CREATE TABLE risk_limits (
    id BIGSERIAL PRIMARY KEY,
    risk_limit_id UUID UNIQUE NOT NULL,
    bot_id VARCHAR(50) NOT NULL,
    limit_type VARCHAR(50) NOT NULL CHECK (limit_type IN 
        ('DAILY_LOSS', 'MAX_POSITION_SIZE', 'MAX_DRAWDOWN', 'MAX_TRADES_PER_HOUR',
         'MAX_OPEN_POSITIONS', 'MAX_CORRELATION', 'MAX_VOLATILITY')),
    
    -- Limit configuration
    limit_value DECIMAL(20, 8) NOT NULL,
    current_usage DECIMAL(20, 8) DEFAULT 0,
    usage_percentage DECIMAL(5, 4) GENERATED ALWAYS AS (current_usage / limit_value) STORED,
    
    -- Time-based limits
    period_type VARCHAR(20) CHECK (period_type IN ('DAILY', 'WEEKLY', 'MONTHLY', 'LIFETIME')),
    period_start TIMESTAMP WITH TIME ZONE,
    period_end TIMESTAMP WITH TIME ZONE,
    
    -- State
    active BOOLEAN NOT NULL DEFAULT TRUE,
    breached BOOLEAN NOT NULL DEFAULT FALSE,
    breach_timestamp TIMESTAMP WITH TIME ZONE,
    
    -- Enforcement
    action_on_breach VARCHAR(20) CHECK (action_on_breach IN ('ALERT', 'PAUSE_BOT', 'CLOSE_POSITIONS')),
    
    -- Audit trail
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Metadata
    metadata JSONB
);
```

**Indexing Strategy**:
- `CREATE INDEX idx_risk_limits_bot_type ON risk_limits(bot_id, limit_type);`
- `CREATE INDEX idx_risk_limits_breached ON risk_limits(breached);`

#### 10. `risk_events` table - Risk Management Events
**Purpose**: Log all risk-related events and actions
**Type**: Medium-frequency, Event
**Usage**: Live Trading, Paper Trading

```sql
CREATE TABLE risk_events (
    id BIGSERIAL PRIMARY KEY,
    event_id UUID UNIQUE NOT NULL,
    bot_id VARCHAR(50) NOT NULL,
    
    -- Event details
    event_type VARCHAR(50) NOT NULL CHECK (event_type IN 
        'DAILY_LOSS_LIMIT_HIT', 'DRAWDOWN_LIMIT_EXCEEDED', 'MAX_POSITION_SIZE_EXCEEDED',
        'HIGH_VOLATILITY_WARNING', 'CORRELATION_ALERT', 'RISK_OVERRIDE', 'RISK_BLOCK_ENFORCED'),
        
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    description TEXT NOT NULL,
    
    -- Context
    trigger_value DECIMAL(20, 8),
    threshold_value DECIMAL(20, 8),
    affected_positions INTEGER DEFAULT 0,
    
    -- Response
    action_taken VARCHAR(50), -- What action was taken
    manual_override BOOLEAN DEFAULT FALSE,
    override_by VARCHAR(50),
    
    -- Timestamps
    event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_timestamp TIMESTAMP WITH TIME ZONE,
    resolved BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    metadata JSONB
);
```

**Indexing Strategy**:
- `CREATE INDEX idx_risk_events_bot_time ON risk_events(bot_id, event_timestamp DESC);`
- `CREATE INDEX idx_risk_events_type_severity ON risk_events(event_type, severity);`
- `CREATE INDEX idx_risk_events_resolved ON risk_events(resolved);`

### Execution & Exchange Interaction

#### 11. `order_lifecycle` table - Order State Tracking
**Purpose**: Track complete order lifecycle from creation to fill/cancel
**Type**: High-frequency, State + Event
**Usage**: Live Trading, Paper Trading

```sql
CREATE TABLE order_lifecycle (
    id BIGSERIAL PRIMARY KEY,
    order_id VARCHAR(100) NOT NULL, -- Exchange order ID
    client_order_id VARCHAR(100) UNIQUE, -- Internal order ID
    
    -- Bot and strategy context
    bot_id VARCHAR(50) NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    trade_id UUID REFERENCES trades(trade_id) ON DELETE SET NULL,
    
    -- Order details
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP', 'STOP_MARKET', 'TAKE_PROFIT', 'TAKE_PROFIT_MARKET')),
    
    -- Pricing and sizing
    price DECIMAL(20, 8), -- For limit orders
    stop_price DECIMAL(20, 8), -- For stop orders
    quantity DECIMAL(20, 8) NOT NULL,
    
    -- Execution state
    status VARCHAR(20) NOT NULL CHECK (status IN ('NEW', 'PARTIALLY_FILLED', 'FILLED', 'CANCELED', 'REJECTED', 'EXPIRED')),
    execution_type VARCHAR(20), -- How the order was filled
    time_in_force VARCHAR(10) DEFAULT 'GTC', -- GTC, IOC, FOK, etc.
    
    -- Execution details
    cummulative_qty DECIMAL(20, 8) DEFAULT 0,
    cummulative_quote_qty DECIMAL(20, 8) DEFAULT 0,
    avg_price DECIMAL(20, 8), -- Average execution price for partial fills
    stop_price_triggered BOOLEAN DEFAULT FALSE,
    
    -- Exchange timestamps
    created_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    executed_timestamp TIMESTAMP WITH TIME ZONE,
    
    -- Exchange-specific fields
    exchange_order_id VARCHAR(100),
    preventions JSONB, -- Order rate limiting, etc.
    
    -- Local processing
    local_creation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    local_update_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Metadata
    metadata JSONB
);
```

**Indexing Strategy**:
- `CREATE INDEX idx_orders_bot_status ON order_lifecycle(bot_id, status);`
- `CREATE INDEX idx_orders_exchange_id ON order_lifecycle(exchange_order_id);`
- `CREATE INDEX idx_orders_client_id ON order_lifecycle(client_order_id);`
- `CREATE INDEX idx_orders_symbol_timestamp ON order_lifecycle(symbol, local_creation_timestamp DESC);`

#### 12. `slippage_analysis` table - Execution Quality Analysis
**Purpose**: Track slippage and execution quality metrics
**Type**: Medium-frequency, Historical
**Usage**: All environments

```sql
CREATE TABLE slippage_analysis (
    id BIGSERIAL PRIMARY KEY,
    analysis_id UUID UNIQUE NOT NULL,
    trade_id UUID NOT NULL REFERENCES trades(trade_id) ON DELETE CASCADE,
    order_id VARCHAR(100) NOT NULL,
    
    -- Expected vs actual execution
    expected_price DECIMAL(20, 8) NOT NULL,
    actual_price DECIMAL(20, 8) NOT NULL,
    slippage_amount DECIMAL(20, 8) NOT NULL, -- actual - expected for buys, expected - actual for sells
    slippage_percentage DECIMAL(8, 4) NOT NULL,
    
    -- Market context
    market_conditions VARCHAR(20) NOT NULL CHECK (market_conditions IN ('CALM', 'MODERATE', 'VOLATILE', 'FLASH_CRASH')),
    bid_ask_spread DECIMAL(20, 8),
    liquidity_at_execution DECIMAL(20, 8), -- Volume available at price level
    
    -- Order characteristics
    order_size DECIMAL(20, 8) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    execution_time_ms INTEGER, -- Time from order creation to fill
    
    -- Analysis context
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10),
    volatility_at_execution DECIMAL(8, 4),
    
    -- Timestamps
    analysis_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    execution_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Metadata
    metadata JSONB
);
```

**Indexing Strategy**:
- `CREATE INDEX idx_slippage_trade_id ON slippage_analysis(trade_id);`
- `CREATE INDEX idx_slippage_symbol_time ON slippage_analysis(symbol, execution_timestamp DESC);`
- `CREATE INDEX idx_slippage_percentage ON slippage_analysis(slippage_percentage);`

### Backtesting & Simulation Data

#### 13. `backtest_runs` table - Backtest Execution Records
**Purpose**: Store information about backtest runs
**Type**: Low-frequency, State
**Usage**: Backtesting

```sql
CREATE TABLE backtest_runs (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    
    -- Backtest configuration
    parameters JSONB NOT NULL, -- Backtest-specific parameters
    initial_capital DECIMAL(20, 8) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    symbols TEXT[] NOT NULL,
    
    -- Execution context
    status VARCHAR(20) NOT NULL CHECK (status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')) DEFAULT 'PENDING',
    start_execution_time TIMESTAMP WITH TIME ZONE,
    end_execution_time TIMESTAMP WITH TIME ZONE,
    
    -- Performance results
    total_trades INTEGER DEFAULT 0,
    total_pnl DECIMAL(20, 8) DEFAULT 0,
    total_return_percentage DECIMAL(8, 4) DEFAULT 0,
    sharpe_ratio DECIMAL(8, 4),
    max_drawdown DECIMAL(8, 4),
    win_rate DECIMAL(5, 4),
    
    -- Resource usage
    execution_time_ms BIGINT,
    memory_used_mb INTEGER,
    data_processed_mb INTEGER,
    
    -- User context
    created_by VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Metadata
    metadata JSONB
);
```

**Indexing Strategy**:
- `CREATE INDEX idx_backtest_strategy_date ON backtest_runs(strategy_id, start_date DESC);`
- `CREATE INDEX idx_backtest_status ON backtest_runs(status);`
- `CREATE INDEX idx_backtest_user ON backtest_runs(created_by, created_at DESC);`

#### 14. `backtest_results` table - Detailed Backtest Results
**Purpose**: Store detailed results for each backtest run
**Type**: High-frequency, Historical
**Usage**: Backtesting

```sql
CREATE TABLE backtest_results (
    id BIGSERIAL PRIMARY KEY,
    result_id UUID UNIQUE NOT NULL,
    run_id UUID NOT NULL REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    
    -- Time context
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    
    -- Portfolio state
    portfolio_value DECIMAL(20, 8) NOT NULL,
    cash_balance DECIMAL(20, 8) NOT NULL,
    total_value DECIMAL(20, 8) NOT NULL,
    
    -- Position state
    position_size DECIMAL(20, 8),
    position_value DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8),
    
    -- Market data
    market_price DECIMAL(20, 8) NOT NULL,
    market_data JSONB, -- OHLCV and indicators at this point
    
    -- Performance tracking
    period_return DECIMAL(8, 4), -- Return for this period
    cumulative_return DECIMAL(8, 4), -- Cumulative return
    drawdown DECIMAL(8, 4), -- Current drawdown percentage
    drawdown_amount DECIMAL(20, 8), -- Current drawdown amount
    
    -- Strategy state
    strategy_state JSONB, -- Internal strategy state at this point
    active_signals JSONB, -- Current signals
    regime_state JSONB, -- Market regime at this point
    
    -- Metadata
    metadata JSONB
);
```

**Indexing Strategy**:
- `CREATE INDEX idx_backtest_results_run_time ON backtest_results(run_id, timestamp DESC);`
- `CREATE INDEX idx_backtest_results_symbol_time ON backtest_results(symbol, timestamp DESC);`

#### 15. `performance_metrics` table - Performance Analytics
**Purpose**: Store calculated performance metrics for strategies and bots
**Type**: Medium-frequency, Historical
**Usage**: All environments

```sql
CREATE TABLE performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_id UUID UNIQUE NOT NULL,
    bot_id VARCHAR(50),
    strategy_id VARCHAR(50) NOT NULL,
    
    -- Time period
    period_type VARCHAR(20) NOT NULL CHECK (period_type IN ('HOURLY', 'DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY', 'YEARLY', 'CUSTOM')),
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Trade metrics
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    total_pnl DECIMAL(20, 8) DEFAULT 0,
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
    
    -- Rate metrics
    win_rate DECIMAL(5, 4) DEFAULT 0,
    avg_win DECIMAL(20, 8) DEFAULT 0,
    avg_loss DECIMAL(20, 8) DEFAULT 0,
    largest_win DECIMAL(20, 8) DEFAULT 0,
    largest_loss DECIMAL(20, 8) DEFAULT 0,
    
    -- Risk-adjusted returns
    profit_factor DECIMAL(8, 4) DEFAULT 0,
    sharpe_ratio DECIMAL(8, 4),
    sortino_ratio DECIMAL(8, 4),
    calmar_ratio DECIMAL(8, 4),
    
    -- Risk metrics
    max_drawdown DECIMAL(8, 4) DEFAULT 0,
    max_drawdown_amount DECIMAL(20, 8) DEFAULT 0,
    volatility DECIMAL(8, 4),
    value_at_risk DECIMAL(8, 4),
    
    -- Volume and fees
    total_volume DECIMAL(20, 8) DEFAULT 0,
    total_fees DECIMAL(20, 8) DEFAULT 0,
    total_taxes DECIMAL(20, 8) DEFAULT 0,
    
    -- Efficiency metrics
    active_days INTEGER DEFAULT 0,
    avg_holding_time_minutes INTEGER DEFAULT 0,
    trades_per_day DECIMAL(8, 2) DEFAULT 0,
    
    -- Regime-specific metrics (for regime-aware strategies)
    regime_metrics JSONB,
    
    -- Calculation metadata
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    calculation_method VARCHAR(50) NOT NULL, -- How metrics were calculated
    data_completeness DECIMAL(5, 4) DEFAULT 1.0, -- Percentage of complete data used
    
    -- Metadata
    metadata JSONB
);
```

**Indexing Strategy**:
- `CREATE INDEX idx_perf_bot_period ON performance_metrics(bot_id, period_start DESC);`
- `CREATE INDEX idx_perf_strategy_period ON performance_metrics(strategy_id, period_start DESC);`
- `CREATE INDEX idx_perf_period_type ON performance_metrics(period_type, period_start DESC);`
- `CREATE INDEX idx_perf_calculated_at ON performance_metrics(calculated_at DESC);`

### Configuration & Versioning

#### 16. `strategy_versions` table - Strategy Evolution Tracking
**Purpose**: Track strategy versions and changes over time
**Type**: Low-frequency, Historical
**Usage**: All environments

```sql
CREATE TABLE strategy_versions (
    id BIGSERIAL PRIMARY KEY,
    version_id UUID UNIQUE NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    version_number VARCHAR(20) NOT NULL, -- Semantic versioning
    
    -- Version details
    parameters JSONB NOT NULL,
    code_hash VARCHAR(64), -- Hash of strategy implementation
    description TEXT,
    
    -- Change tracking
    change_type VARCHAR(20) NOT NULL CHECK (change_type IN ('MAJOR', 'MINOR', 'PATCH', 'HOTFIX')),
    changes JSONB, -- Detailed change description
    
    -- Performance comparison
    backtest_comparison JSONB, -- Performance vs previous version
    
    -- Lifecycle
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(50) NOT NULL,
    active_from TIMESTAMP WITH TIME ZONE, -- When this version became active
    active_to TIMESTAMP WITH TIME ZONE, -- When this version was retired
    
    -- State
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    is_deprecated BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Metadata
    metadata JSONB
);
```

**Indexing Strategy**:
- `CREATE INDEX idx_strategy_versions_strategy ON strategy_versions(strategy_id, created_at DESC);`
- `CREATE INDEX idx_strategy_versions_active ON strategy_versions(is_active);`
- `CREATE INDEX idx_strategy_versions_user ON strategy_versions(created_by);`

#### 17. `feature_flags` table - Dynamic Feature Control
**Purpose**: Store feature flags for risk controls and experimental features
**Type**: Low-frequency, State
**Usage**: All environments

```sql
CREATE TABLE feature_flags (
    id BIGSERIAL PRIMARY KEY,
    flag_id UUID UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    
    -- Flag configuration
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    environment VARCHAR(20) NOT NULL CHECK (environment IN ('LIVE', 'PAPER', 'BACKTEST', 'ALL')),
    
    -- Targeting
    target_bots TEXT[], -- Specific bots to apply to
    target_strategies TEXT[], -- Specific strategies to apply to
    rollout_percentage DECIMAL(5, 2) DEFAULT 100.00, -- Percentage of traffic
    
    -- Constraints
    constraints JSONB, -- Additional constraints for activation
    
    -- Audit trail
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(50) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_by VARCHAR(50),
    
    -- Metadata
    metadata JSONB
);
```

**Indexing Strategy**:
- `CREATE INDEX idx_feature_flags_name ON feature_flags(name);`
- `CREATE INDEX idx_feature_flags_env_enabled ON feature_flags(environment, enabled);`

## Data Flow (from Decision to Execution)

### 1. Strategy Decision Flow
```
Market Data Input → Indicator Calculation → Signal Generation → 
Strategy Decision → Risk Check → Order Creation → Exchange Execution → 
Trade Recording → Position Update → Performance Tracking
```

**Detailed Flow:**
1. **Market Data Reception**: Real-time market data feeds into the system
2. **Indicator Calculation**: Technical indicators are calculated and stored in `indicator_snapshots`
3. **Signal Generation**: Strategy generates buy/sell signals stored in `strategy_decisions`
4. **Risk Validation**: Before execution, risk limits in `risk_limits` are checked
5. **Order Creation**: Order is created and stored in `order_lifecycle` with status 'NEW'
6. **Exchange Execution**: Order is sent to exchange and lifecycle status updates
7. **Trade Recording**: Upon fill, trade details are stored in `trades` table
8. **Position Update**: Position state is updated in `positions` table
9. **Performance Tracking**: Performance metrics are calculated and stored in `performance_metrics`

### 2. Atomic Transaction Boundaries
Critical operations are grouped in database transactions to ensure consistency:

```sql
-- Trade execution transaction
BEGIN;
INSERT INTO trades (trade_id, bot_id, position_id, ...) VALUES (...);
UPDATE positions SET quantity = quantity + new_quantity, 
                    current_price = new_price,
                    unrealized_pnl = calculate_pnl(),
                    version = version + 1
      WHERE position_id = ? AND version = ?; -- Optimistic locking
INSERT INTO slippage_analysis (trade_id, expected_price, actual_price, ...) VALUES (...);
UPDATE bots SET performance_metrics = jsonb_set(performance_metrics, '{total_trades}', 
           (performance_metrics->'total_trades')::int + 1);
COMMIT;
```

### 3. Real-time State Management
- Critical state data is cached in Redis for high-frequency access
- PostgreSQL serves as the single source of truth
- Cache invalidation happens automatically when DB records update

## Storage for Live vs Paper vs Backtest

### Live Trading Storage
- **Primary Focus**: Execution accuracy and risk management
- **Tables Used**: All tables except `backtest_runs` and `backtest_results`
- **Characteristics**: 
  - High write frequency for trades, orders, positions
  - Real-time risk monitoring and limit enforcement
  - Audit requirements for compliance
  - Performance monitoring with real money at stake

### Paper Trading Storage
- **Primary Focus**: Strategy validation without financial risk
- **Tables Used**: All tables except `backtest_results` (uses `trades` and `positions`)
- **Characteristics**:
  - Similar schema to Live but with virtual execution
  - Risk metrics collected for comparison with Live
  - Performance tracking without real financial impact
  - Can be used to validate risk parameters before Live deployment

### Backtesting Storage
- **Primary Focus**: Historical performance analysis and optimization
- **Tables Used**: `backtest_runs`, `backtest_results`, `performance_metrics`
- **Characteristics**:
  - High-volume historical data processing
  - Detailed state tracking at each time step
  - Results stored for strategy comparison
  - No real execution - simulated environment

## Indexing, Partitioning & Retention Strategy

### Indexing Strategy

#### Primary Indexes
- **B-tree indexes** for high-selectivity lookups (IDs, timestamps)
- **GIN indexes** for JSONB fields and array columns
- **Partial indexes** for active records only

#### Critical Performance Indexes
```sql
-- Trades table indexes
CREATE INDEX idx_trades_bot_timestamp ON trades(bot_id, local_timestamp DESC);
CREATE INDEX idx_trades_symbol_timestamp ON trades(symbol, local_timestamp DESC);
CREATE INDEX idx_trades_strategy_timestamp ON trades(strategy_id, local_timestamp DESC);

-- Positions table indexes
CREATE INDEX idx_positions_bot_status ON positions(bot_id, status);
CREATE INDEX idx_positions_symbol_status ON positions(symbol, status);

-- Performance metrics indexes
CREATE INDEX idx_perf_bot_period ON performance_metrics(bot_id, period_start DESC);
CREATE INDEX idx_perf_strategy_period ON performance_metrics(strategy_id, period_start DESC);
```

### Partitioning Strategy

#### Time-based Partitioning
For high-frequency tables, implement time-based partitioning:

```sql
-- Partition trades table by month
CREATE TABLE trades PARTITION OF trades 
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Similar partitioning for performance_metrics, risk_events, etc.
```

#### Use Cases for Partitioning
- **Trades**: Partition by month for active access patterns
- **Performance Metrics**: Partition by period type (daily, weekly, monthly)
- **Strategy Decisions**: Partition by month due to high frequency
- **Backtest Results**: Partition by backtest run date

### Retention Strategy

#### Hot Data (Last 30 days)
- Full detail in main tables
- All fields and indexes active
- Highest performance requirements

#### Warm Data (30 days to 1 year)
- Same schema but with read-optimized indexes
- Archive to monthly partitions
- Aggregated metrics available for queries

#### Cold Data (Over 1 year)
- Moved to read-optimized partitions
- Summary tables for common queries
- Potentially moved to separate historical database

#### Data Lifecycle Policy
```sql
-- Example policy for trades
CREATE OR REPLACE FUNCTION manage_trade_partitions()
RETURNS VOID AS $$
BEGIN
    -- Create new monthly partitions
    -- Archive old partitions to read-optimized format
    -- Update statistics for query planner
END;
$$ LANGUAGE plpgsql;
```

### Performance Optimization

#### Write Optimization
- Use bulk operations for high-frequency inserts
- Implement proper connection pooling
- Use prepared statements for repeated operations

#### Read Optimization
- Materialized views for complex aggregations
- Proper indexing strategy based on query patterns
- Read replicas for analytics queries

## Migration Plan (from current logs to structured data)

### Phase 1: Database Setup and Schema Creation (Week 1)
1. **Set up PostgreSQL instance** with proper configuration for trading workloads
2. **Create all database schemas** using the above table definitions
3. **Implement connection pooling** with proper transaction handling
4. **Set up TimescaleDB** for time-series optimizations if needed
5. **Create initial indexes** based on expected query patterns

### Phase 2: Data Migration from Logs (Week 2)
1. **Extract existing log data** from current MongoDB/JSON logs
2. **Transform log data** into proper relational format
3. **Migrate trade history** from logs to `trades` table
4. **Migrate position data** from logs to `positions` table
5. **Migrate performance metrics** from logs to `performance_metrics` table

**Sample Migration Script:**
```sql
-- Migrate existing trade logs (hypothetical log data)
INSERT INTO trades (
    trade_id, bot_id, strategy_id, symbol, side, action, 
    price, quantity, cost, fees, status, 
    exchange_timestamp, local_timestamp, reason
)
SELECT 
    gen_random_uuid(), 
    log_data->>'bot_id',
    log_data->>'strategy_id', 
    log_data->>'symbol',
    log_data->>'side',
    CASE 
        WHEN log_data->>'type' = 'entry' THEN 'ENTRY'
        WHEN log_data->>'type' = 'exit' THEN 'FULL_EXIT'
        ELSE 'ADJUSTMENT'
    END,
    (log_data->>'price')::DECIMAL,
    (log_data->>'quantity')::DECIMAL,
    ((log_data->>'price')::DECIMAL * (log_data->>'quantity')::DECIMAL),
    COALESCE((log_data->>'fees')::DECIMAL, 0),
    'FILLED',
    (log_data->>'timestamp')::TIMESTAMP WITH TIME ZONE,
    NOW(),
    log_data->>'reason'
FROM existing_trade_logs;
```

### Phase 3: Application Integration (Week 3)
1. **Modify trading service** to write to PostgreSQL instead of RAM-only storage
2. **Update order management** to use `order_lifecycle` table
3. **Integrate position tracking** with `positions` table
4. **Implement atomic transactions** for trade/position consistency
5. **Update risk management** to use `risk_limits` and `risk_metrics` tables

### Phase 4: State Persistence (Week 4)
1. **Implement bot state persistence** to survive restarts
2. **Create startup recovery procedures** to restore active positions
3. **Implement configuration versioning** with `bot_configs` table
4. **Add strategy versioning** with `strategy_versions` table
5. **Implement backup and recovery** procedures

### Phase 5: Analytics and Monitoring (Week 5)
1. **Create performance calculation procedures** for `performance_metrics`
2. **Implement real-time metrics** with proper indexing
3. **Set up monitoring** for database performance
4. **Create dashboard queries** for operational visibility
5. **Implement audit trails** for compliance requirements

### Phase 6: Optimization and Scaling (Week 6)
1. **Implement partitioning strategy** for high-volume tables
2. **Optimize indexes** based on actual query patterns
3. **Set up read replicas** for analytics workloads
4. **Implement caching layer** with Redis for performance
5. **Finalize retention policies** for long-term storage

### Risk Mitigation During Migration

1. **Parallel Operation**: Run new PostgreSQL storage alongside existing logging temporarily
2. **Rollback Capability**: Maintain ability to revert to old system if needed
3. **Data Validation**: Implement checksums and validation to ensure data integrity
4. **Monitoring**: Implement comprehensive monitoring during migration
5. **Testing**: Thorough testing in staging environment before production deployment

This comprehensive storage architecture provides a robust, scalable, and auditable foundation for your algorithmic trading system that can handle the specific requirements of your Enhanced RSI Strategy V5, Dynamic Risk Manager, Market Regime Detector, and other components while ensuring data integrity, performance, and compliance.