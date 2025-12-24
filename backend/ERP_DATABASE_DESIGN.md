# ERP-Focused Database Design for Automated Trading System

## Complete Analysis of Current Architecture

### 1. Data Type Analysis
**State Data:**
- `positions` - Current position state
- `bots` - Runtime bot state  
- `risk_limits` - Active risk limits
- `feature_flags` - Feature toggle states

**Event Data:**
- `trades` - Trade execution events
- `strategy_decisions` - Decision events
- `order_lifecycle` - Order state changes
- `risk_events` - Risk-related events
- `indicator_snapshots` - Indicator value snapshots

**History/Analytics Data:**
- `indicator_snapshots` - Historical indicators
- `performance_metrics` - Performance history
- `backtest_results` - Historical simulation results
- `slippage_analysis` - Execution quality history

**Sensitivity Levels:**
- **Financial (Critical):** trades, positions, P&L data
- **Operational (High):** bot states, orders, risk events
- **Analytical (Medium):** metrics, indicators, decisions

**Read/Write Patterns:**
- **Write-heavy:** trades, decisions, indicators, order updates
- **Read-heavy:** performance metrics, historical analysis
- **Append-only:** risk events, backtest results

**Audit/Recovery Needs:**
- Full audit trail for compliance
- Ability to replay events for recovery
- Versioned configurations for rollback

## ERP-Focused Database Design

### ER Diagram (Logical Entities)

#### **Module 1: Trading Core (TRD)**
- `TRD_trades` - Execution records (Event)
- `TRD_positions` - Position state (State) 
- `TRD_orders` - Order lifecycle (State/Event hybrid)
- `TRD_exchanges` - Exchange configuration (State)

#### **Module 2: Risk Management (RISK)**
- `RISK_limits` - Risk configuration (State)
- `RISK_metrics` - Risk measurements (History)
- `RISK_events` - Risk incidents (Event)
- `RISK_controls` - Risk controls configuration (State)

#### **Module 3: Strategy & Decision Engine (STRAT)**
- `STRAT_decisions` - Strategy decisions (Event)
- `STRAT_indicators` - Indicator snapshots (History)
- `STRAT_strategies` - Strategy config (State)
- `STRAT_versions` - Strategy versions (History)

#### **Module 4: Analytics & Performance (ANLT)**
- `ANLT_performance` - Performance metrics (History)
- `ANLT_backtests` - Backtest runs (State)
- `ANLT_backtest_results` - Simulation results (History)

#### **Module 5: Governance & Configuration (GOV)**
- `GOV_bots` - Bot configurations (State)
- `GOV_bot_configs` - Bot config versions (History)
- `GOV_feature_flags` - Feature toggles (State)
- `GOV_users` - User management (State)

### Detailed Table Design with ERP Focus

#### **TRADING CORE MODULE**

```sql
-- TRD_orders: Central order management
CREATE TABLE TRD_orders (
    id BIGSERIAL PRIMARY KEY,
    order_id VARCHAR(100) UNIQUE NOT NULL, -- Exchange order ID
    client_order_id VARCHAR(100) UNIQUE, -- Internal ID
    bot_id VARCHAR(50) NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    trade_id UUID REFERENCES TRD_trades(id) ON DELETE SET NULL,
    
    -- Trading details
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP', 'TAKE_PROFIT')),
    price DECIMAL(20, 8) DEFAULT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    
    -- Execution state tracking
    status VARCHAR(20) NOT NULL CHECK (status IN ('NEW', 'PARTIALLY_FILLED', 'FILLED', 'CANCELED', 'REJECTED')),
    cummulative_qty DECIMAL(20, 8) DEFAULT 0,
    avg_fill_price DECIMAL(20, 8) DEFAULT NULL,
    
    -- Timestamps
    exchange_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    local_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Versioning for concurrent updates
    version INTEGER NOT NULL DEFAULT 1,
    
    -- Audit trail
    created_by VARCHAR(50) NOT NULL,
    updated_by VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TRD_trades: Execution records
CREATE TABLE TRD_trades (
    id BIGSERIAL PRIMARY KEY,
    trade_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    order_id VARCHAR(100) NOT NULL,
    bot_id VARCHAR(50) NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    position_id UUID, -- Reference to positions
    
    -- Trading details
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    action VARCHAR(20) NOT NULL CHECK (action IN ('ENTRY', 'PARTIAL_EXIT', 'FULL_EXIT')),
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    cost DECIMAL(20, 8) NOT NULL,
    fees DECIMAL(20, 8) NOT NULL,
    
    -- Performance impact
    pnl DECIMAL(20, 8) DEFAULT 0,
    pnl_pct DECIMAL(8, 4) DEFAULT 0,
    
    -- Execution context
    exchange_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    local_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    slippage DECIMAL(8, 4),
    
    -- Audit trail
    created_by VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Foreign key constraint
    CONSTRAINT fk_trades_position FOREIGN KEY (position_id) REFERENCES TRD_positions(position_id)
);

-- TRD_positions: Position state management
CREATE TABLE TRD_positions (
    id BIGSERIAL PRIMARY KEY,
    position_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    bot_id VARCHAR(50) NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    
    -- Position details
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    entry_price DECIMAL(20, 8) NOT NULL,
    avg_entry_price DECIMAL(20, 8),
    quantity DECIMAL(20, 8) NOT NULL,
    
    -- Current state
    current_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    
    -- Risk management
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    liquidation_price DECIMAL(20, 8),
    
    -- Lifecycle
    status VARCHAR(20) NOT NULL CHECK (status IN ('ACTIVE', 'CLOSING', 'CLOSED')) DEFAULT 'ACTIVE',
    entry_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    close_timestamp TIMESTAMP WITH TIME ZONE,
    
    -- Concurrency control
    version INTEGER NOT NULL DEFAULT 1,
    
    -- Audit trail
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### **RISK MANAGEMENT MODULE**

```sql
-- RISK_limits: Risk limit configuration
CREATE TABLE RISK_limits (
    id BIGSERIAL PRIMARY KEY,
    limit_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    bot_id VARCHAR(50) NOT NULL,
    limit_type VARCHAR(50) NOT NULL CHECK (limit_type IN 
        ('DAILY_LOSS', 'MAX_POSITION_SIZE', 'MAX_DRAWDOWN', 'MAX_TRADES_PER_HOUR',
         'MAX_OPEN_POSITIONS', 'MAX_CORRELATION')),
    
    -- Limit configuration
    limit_value DECIMAL(20, 8) NOT NULL,
    current_usage DECIMAL(20, 8) DEFAULT 0,
    usage_percentage DECIMAL(5, 4) GENERATED ALWAYS AS (current_usage / limit_value) STORED,
    
    -- Time-based limits
    period_type VARCHAR(20) CHECK (period_type IN ('DAILY', 'WEEKLY', 'MONTHLY')),
    period_start TIMESTAMP WITH TIME ZONE,
    period_end TIMESTAMP WITH TIME ZONE,
    
    -- State management
    active BOOLEAN NOT NULL DEFAULT TRUE,
    breached BOOLEAN NOT NULL DEFAULT FALSE,
    breach_timestamp TIMESTAMP WITH TIME ZONE,
    
    -- Enforcement
    action_on_breach VARCHAR(20) CHECK (action_on_breach IN ('ALERT', 'PAUSE_BOT', 'CLOSE_POSITIONS')),
    
    -- Audit trail
    created_by VARCHAR(50) NOT NULL,
    updated_by VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraint to prevent duplicate active limits
    CONSTRAINT unique_active_limit_per_type_per_bot 
        UNIQUE (bot_id, limit_type, active) 
        WHERE active = TRUE
);

-- RISK_events: Risk incidents
CREATE TABLE RISK_events (
    id BIGSERIAL PRIMARY KEY,
    event_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    bot_id VARCHAR(50) NOT NULL,
    
    -- Event details
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    description TEXT NOT NULL,
    
    -- Context
    trigger_value DECIMAL(20, 8),
    threshold_value DECIMAL(20, 8),
    affected_positions INTEGER DEFAULT 0,
    
    -- Response
    action_taken VARCHAR(50),
    manual_override BOOLEAN DEFAULT FALSE,
    override_by VARCHAR(50),
    
    -- Resolution
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by VARCHAR(50),
    
    -- Audit trail
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(50) NOT NULL
);

-- RISK_metrics: Risk measurements
CREATE TABLE RISK_metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    bot_id VARCHAR(50) NOT NULL,
    strategy_id VARCHAR(50),
    
    -- Trade-specific metrics
    trade_id UUID REFERENCES TRD_trades(id),
    position_id UUID REFERENCES TRD_positions(position_id),
    
    -- Risk measurements
    volatility_at_entry DECIMAL(8, 4),
    atr_at_entry DECIMAL(20, 8),
    position_risk DECIMAL(20, 8),
    correlation_with_portfolio DECIMAL(5, 4),
    max_drawdown DECIMAL(8, 4),
    
    -- Timestamps
    measurement_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    period_start TIMESTAMP WITH TIME ZONE,
    period_end TIMESTAMP WITH TIME ZONE,
    
    -- Audit trail
    created_by VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### **STRATEGY & DECISION ENGINE MODULE**

```sql
-- STRAT_decisions: Strategy decisions
CREATE TABLE STRAT_decisions (
    id BIGSERIAL PRIMARY KEY,
    decision_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    bot_id VARCHAR(50) NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    
    -- Decision details
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    signal_type VARCHAR(20) NOT NULL CHECK (signal_type IN ('BUY', 'SELL', 'HOLD', 'EXIT')),
    decision_price DECIMAL(20, 8) NOT NULL,
    confidence_score DECIMAL(5, 4),
    decision_reason TEXT,
    
    -- Decision context
    market_data JSONB,
    indicator_values JSONB,
    regime_state JSONB,
    trend_filter_result JSONB,
    
    -- Execution link
    executed_trade_id UUID REFERENCES TRD_trades(trade_id),
    
    -- Audit trail
    decision_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(50) NOT NULL
);

-- STRAT_indicators: Indicator snapshots
CREATE TABLE STRAT_indicators (
    id BIGSERIAL PRIMARY KEY,
    snapshot_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    bot_id VARCHAR(50) NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    
    -- Time context
    timeframe VARCHAR(10) NOT NULL,
    kline_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Technical indicators
    rsi_value DECIMAL(8, 4),
    moving_averages JSONB,
    bollinger_bands JSONB,
    macd_values JSONB,
    atr_value DECIMAL(20, 8),
    volatility DECIMAL(8, 4),
    
    -- Market regime
    regime_type VARCHAR(20),
    regime_confidence DECIMAL(5, 4),
    
    -- Audit trail
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(50) NOT NULL
);

-- STRAT_strategies: Strategy configuration
CREATE TABLE STRAT_strategies (
    id BIGSERIAL PRIMARY KEY,
    strategy_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    class_name VARCHAR(100) NOT NULL,
    
    -- Configuration
    parameters JSONB NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    version VARCHAR(20) NOT NULL,
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH')),
    
    -- Targeting
    target_symbols TEXT[],
    timeframes TEXT[],
    
    -- Audit trail
    created_by VARCHAR(50) NOT NULL,
    updated_by VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- STRAT_versions: Strategy versioning
CREATE TABLE STRAT_versions (
    id BIGSERIAL PRIMARY KEY,
    version_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    strategy_id VARCHAR(50) NOT NULL,
    version_number VARCHAR(20) NOT NULL,
    
    -- Version details
    parameters JSONB NOT NULL,
    code_hash VARCHAR(64),
    description TEXT,
    change_type VARCHAR(20) NOT NULL CHECK (change_type IN ('MAJOR', 'MINOR', 'PATCH')),
    
    -- Lifecycle
    active_from TIMESTAMP WITH TIME ZONE,
    active_to TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Audit trail
    created_by VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraint
    CONSTRAINT unique_active_strategy_version 
        UNIQUE (strategy_id, is_active) 
        WHERE is_active = TRUE
);
```

#### **ANALYTICS & PERFORMANCE MODULE**

```sql
-- ANLT_performance: Performance metrics
CREATE TABLE ANLT_performance (
    id BIGSERIAL PRIMARY KEY,
    metric_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    bot_id VARCHAR(50),
    strategy_id VARCHAR(50) NOT NULL,
    
    -- Time period
    period_type VARCHAR(20) NOT NULL CHECK (period_type IN ('DAILY', 'WEEKLY', 'MONTHLY')),
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Performance metrics
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    total_pnl DECIMAL(20, 8) DEFAULT 0,
    win_rate DECIMAL(5, 4) DEFAULT 0,
    sharpe_ratio DECIMAL(8, 4),
    max_drawdown DECIMAL(8, 4),
    profit_factor DECIMAL(8, 4),
    
    -- Audit trail
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    calculated_by VARCHAR(50) NOT NULL,
    calculation_method VARCHAR(50) NOT NULL
);

-- ANLT_backtests: Backtest runs
CREATE TABLE ANLT_backtests (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    
    -- Configuration
    parameters JSONB NOT NULL,
    initial_capital DECIMAL(20, 8) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    symbols TEXT[] NOT NULL,
    
    -- Execution
    status VARCHAR(20) NOT NULL CHECK (status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')),
    start_execution_time TIMESTAMP WITH TIME ZONE,
    end_execution_time TIMESTAMP WITH TIME ZONE,
    
    -- Results
    total_trades INTEGER DEFAULT 0,
    total_pnl DECIMAL(20, 8) DEFAULT 0,
    win_rate DECIMAL(5, 4) DEFAULT 0,
    sharpe_ratio DECIMAL(8, 4),
    
    -- Audit trail
    created_by VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ANLT_backtest_results: Detailed backtest results
CREATE TABLE ANLT_backtest_results (
    id BIGSERIAL PRIMARY KEY,
    result_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES ANLT_backtests(run_id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    
    -- Time & state
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    portfolio_value DECIMAL(20, 8) NOT NULL,
    
    -- Performance at this point
    cumulative_return DECIMAL(8, 4),
    drawdown DECIMAL(8, 4),
    
    -- Audit trail
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### **GOVERNANCE & CONFIGURATION MODULE**

```sql
-- GOV_bots: Bot configuration
CREATE TABLE GOV_bots (
    id BIGSERIAL PRIMARY KEY,
    bot_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    
    -- Configuration
    bot_type VARCHAR(20) NOT NULL CHECK (bot_type IN ('LIVE', 'PAPER', 'BACKTEST')),
    mode VARCHAR(20) NOT NULL CHECK (mode IN ('AUTOMATED', 'MANUAL', 'LIMITED')) DEFAULT 'AUTOMATED',
    exchange VARCHAR(50) NOT NULL,
    
    -- Runtime state
    status VARCHAR(20) NOT NULL CHECK (status IN ('RUNNING', 'PAUSED', 'STOPPED', 'ERROR')),
    start_time TIMESTAMP WITH TIME ZONE,
    last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Portfolio state
    portfolio_value DECIMAL(20, 8),
    initial_capital DECIMAL(20, 8),
    
    -- Configuration links
    active_config_id UUID, -- References GOV_bot_configs
    active_strategy_id VARCHAR(50), -- References STRAT_strategies
    
    -- Audit trail
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    version INTEGER NOT NULL DEFAULT 1
);

-- GOV_bot_configs: Bot configuration versions
CREATE TABLE GOV_bot_configs (
    id BIGSERIAL PRIMARY KEY,
    config_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    bot_id VARCHAR(50) NOT NULL REFERENCES GOV_bots(bot_id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    
    -- Configuration sections
    strategy_params JSONB,
    risk_params JSONB,
    trading_params JSONB,
    
    -- State tracking
    active BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(50) NOT NULL,
    applied_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraint to ensure only one active config per bot
    CONSTRAINT unique_active_bot_config 
        UNIQUE (bot_id, active) 
        WHERE active = TRUE
);

-- GOV_feature_flags: Feature control
CREATE TABLE GOV_feature_flags (
    id BIGSERIAL PRIMARY KEY,
    flag_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    
    -- Configuration
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    environment VARCHAR(20) NOT NULL CHECK (environment IN ('LIVE', 'PAPER', 'BACKTEST', 'ALL')),
    rollout_percentage DECIMAL(5, 2) DEFAULT 100.00,
    
    -- Targeting
    target_bots TEXT[],
    target_strategies TEXT[],
    
    -- Audit trail
    created_by VARCHAR(50) NOT NULL,
    updated_by VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Key Indexes for Performance

```sql
-- Trading Core Indexes
CREATE INDEX idx_trd_orders_bot_status ON TRD_orders(bot_id, status);
CREATE INDEX idx_trd_orders_symbol_time ON TRD_orders(symbol, local_timestamp DESC);
CREATE INDEX idx_trd_trades_bot_time ON TRD_trades(bot_id, local_timestamp DESC);
CREATE INDEX idx_trd_positions_bot_status ON TRD_positions(bot_id, status);

-- Risk Management Indexes
CREATE INDEX idx_risk_limits_bot_type ON RISK_limits(bot_id, limit_type);
CREATE INDEX idx_risk_events_bot_time ON RISK_events(bot_id, created_at DESC);
CREATE INDEX idx_risk_events_breached ON RISK_events(breached);

-- Strategy Engine Indexes
CREATE INDEX idx_strat_decisions_bot_time ON STRAT_decisions(bot_id, decision_timestamp DESC);
CREATE INDEX idx_strat_indicators_bot_symbol_time ON STRAT_indicators(bot_id, symbol, kline_timestamp DESC);

-- Performance Indexes
CREATE INDEX idx_anlt_performance_bot_period ON ANLT_performance(bot_id, period_start DESC);
CREATE INDEX idx_anlt_backtests_strategy_date ON ANLT_backtests(strategy_id, start_date DESC);

-- Governance Indexes
CREATE INDEX idx_gov_bots_status ON GOV_bots(status);
CREATE INDEX idx_gov_bots_type_status ON GOV_bots(bot_type, status);
```

### Critical Issues in Original Design and Corrections

#### **1. Issue: Missing Foreign Key Constraints**
**Original Problem:** Weak relationships between entities
**Correction:** Added proper foreign key constraints with cascading rules

#### **2. Issue: No Data Isolation for Live vs Paper vs Backtest**
**Original Problem:** Mixed data environments could cause conflicts
**Correction:** Implemented explicit `bot_type` field and environment-aware queries

#### **3. Issue: Insufficient Audit Trail**
**Original Problem:** Limited audit information
**Correction:** Added `created_by`, `updated_by`, and comprehensive audit fields

#### **4. Issue: No Concurrency Control**
**Original Problem:** Race conditions in high-frequency updates
**Correction:** Added `version` fields for optimistic locking

#### **5. Issue: No Data Integrity Constraints**
**Original Problem:** Risk of inconsistent data
**Correction:** Added check constraints, unique constraints, and referential integrity

### ERP-Driven Design Benefits

1. **Financial Safety:** Proper segregation of live trading data from other environments
2. **Auditability:** Complete audit trails with user attribution
3. **Risk Management:** Isolated risk tracking with proper validation
4. **Scalability:** Modular design allows independent scaling of modules
5. **Compliance:** Proper change tracking and versioning
6. **Recovery:** Clear state management with versioned configurations

This ERP-focused design ensures financial safety through proper data segregation, maintains complete audit trails for compliance, and provides the scalability needed for high-frequency trading operations while maintaining clean separation between different operational environments.