import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// System
export const getSystemHealth = () => api.get('/api/system/health');

// Market
export const getMarketAnalysis = (symbol: string, timeframe = '1h') =>
  api.get(`/api/market/analysis/${symbol}`, { params: { timeframe } });

export const getOHLCV = (symbol: string, timeframe = '1h', limit = 100) =>
  api.get(`/api/market/ohlcv/${symbol}`, { params: { timeframe, limit } });

export const getIndicators = (symbol: string, timeframe = '1h', limit = 100) =>
  api.get(`/api/market/indicators/${symbol}`, { params: { timeframe, limit } });

// Backtest
export const runBacktest = (data: {
  symbol: string;
  timeframe: string;
  days: number;
  strategy_params?: any;
}) => api.post('/api/backtest/run', data);

export const getBacktestResults = (taskId: string) =>
  api.get(`/api/backtest/${taskId}/results`);

export const getEquityCurve = (taskId: string) =>
  api.get(`/api/backtest/${taskId}/equity-curve`);

// Trading
export const startTrading = (data: {
  symbol: string;
  timeframe: string;
  strategy_params?: any;
}) => api.post('/api/trading/start', data);

export const stopTrading = () => api.post('/api/trading/stop');

export const getTradingStatus = () => api.get('/api/trading/status');

export const getPositions = () => api.get('/api/trading/positions');

export const getTradeHistory = (limit = 100) =>
  api.get('/api/trading/history', { params: { limit } });

// Strategies
export const getStrategies = () => api.get('/api/strategies/list');

export const getMLStatus = () => api.get('/api/strategies/ml-status');

export const configureStrategy = (params: any) =>
  api.post('/api/strategies/configure', params);

export const getStrategyParameters = () => api.get('/api/strategies/parameters');

export const retrainML = (data: { symbol: string; timeframe: string; days: number }) =>
  api.post('/api/strategies/ml/retrain', data);

// Reports
export const getPerformance = () => api.get('/api/reports/performance');

export const getTrades = (params?: {
  symbol?: string;
  start_date?: string;
  end_date?: string;
  limit?: number;
}) => api.get('/api/reports/trades', { params });

export const getDailyStats = (days = 30) =>
  api.get('/api/reports/daily-stats', { params: { days } });

// Config
export const getSymbols = () => api.get('/api/config/symbols');

export const getTimeframes = () => api.get('/api/config/timeframes');

export const getRiskConfig = () => api.get('/api/config/risk');

export const updateRiskConfig = (data: {
  risk_per_trade: number;
  max_position_size: number;
  max_daily_loss: number;
}) => api.put('/api/config/risk', data);

export default api;
