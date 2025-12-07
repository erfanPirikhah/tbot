export interface MarketAnalysis {
  symbol: string;
  price: number;
  market_state: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  regime: string;
  confidence: number;
  timestamp: string;
}

export interface BacktestRequest {
  symbol: string;
  timeframe: string;
  days: number;
  strategy_params?: Record<string, any>;
}

export interface BacktestResult {
  task_id: string;
  status: 'running' | 'completed' | 'failed';
  timestamp: string;
  metrics?: {
    total_trades: number;
    win_rate: number;
    total_pnl: number;
    sharpe_ratio?: number;
  };
  equity_curve?: Array<{ date: string; value: number }>;
  trades?: Array<Trade>;
  error?: string;
}

export interface Trade {
  timestamp: string;
  symbol: string;
  side: 'LONG' | 'SHORT';
  action: 'ENTRY' | 'EXIT';
  price: number;
  quantity?: number;
  pnl_percentage?: number;
  pnl_amount?: number;
  reason?: string;
}

export interface TradingStatus {
  is_running: boolean;
  current_position: string;
  portfolio_value: number;
  active_symbols: string[];
  uptime_seconds?: number;
}

export interface Position {
  symbol: string;
  side: 'LONG' | 'SHORT';
  entry_price: number;
  current_price: number;
  quantity: number;
  pnl_percentage: number;
  pnl_amount: number;
  stop_loss: number;
  take_profit: number;
}

export interface PerformanceMetrics {
  total_trades: number;
  win_rate: number;
  total_pnl: number;
  avg_win: number;
  avg_loss: number;
  profit_factor: number;
  wins: number;
  losses: number;
}

export interface DailyStats {
  date: string;
  trades: number;
  pnl: number;
  wins: number;
  losses: number;
}
