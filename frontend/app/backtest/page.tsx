'use client';

import { useState } from 'react';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import { runBacktest, getBacktestResults } from '@/lib/api';
import { BacktestResult } from '@/lib/types';

export default function BacktestPage() {
  const [symbol, setSymbol] = useState('BTCUSDT');
  const [timeframe, setTimeframe] = useState('1h');
  const [days, setDays] = useState(30);
  const [loading, setLoading] = useState(false);
  const [taskId, setTaskId] = useState('');
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [testMode, setTestMode] = useState(true); // Enable test mode by default

  const handleRunBacktest = async () => {
    setLoading(true);
    setResult(null);
    try {
      // Build strategy params with test mode if enabled
      const strategy_params = testMode ? {
        test_mode_enabled: true,
        bypass_contradiction_detection: true,
        relax_risk_filters: true,
        relax_entry_conditions: true,
        enable_all_signals: true,
        max_trades_per_100: 100,
        min_candles_between: 1,
        rsi_entry_buffer: 5,
        rsi_oversold: 35,
        rsi_overbought: 65,
        enable_short_trades: true,
        enable_trend_filter: false,
        enable_mtf: false,
        risk_per_trade: 0.02
      } : undefined;

      const res = await runBacktest({ symbol, timeframe, days, strategy_params });
      const newTaskId = res.data.task_id;
      setTaskId(newTaskId);

      // Poll for results
      const pollInterval = setInterval(async () => {
        const resultRes = await getBacktestResults(newTaskId);
        const data = resultRes.data;
        
        if (data.status === 'completed' || data.status === 'failed') {
          setResult(data);
          setLoading(false);
          clearInterval(pollInterval);
        }
      }, 2000);
    } catch (error) {
      console.error('Error running backtest:', error);
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold mb-2">Backtesting</h1>
        <p className="text-slate-400">Test your strategy on historical data</p>
      </div>

      {/* Configuration */}
      <Card title="Backtest Configuration">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm text-slate-400 mb-2">Symbol</label>
            <input
              type="text"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white"
            />
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-2">Timeframe</label>
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white"
            >
              <option value="1h">1 Hour</option>
              <option value="4h">4 Hours</option>
              <option value="1d">1 Day</option>
            </select>
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-2">Days Back</label>
            <input
              type="number"
              value={days}
              onChange={(e) => setDays(parseInt(e.target.value))}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white"
            />
          </div>
        </div>
        <div className="mt-4 space-y-3">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={testMode}
              onChange={(e) => setTestMode(e.target.checked)}
              className="w-4 h-4 rounded bg-slate-800 border-slate-700"
            />
            <span className="text-sm text-slate-300">
              Test Mode (More permissive parameters for testing)
            </span>
          </label>
          <Button onClick={handleRunBacktest} disabled={loading} variant="primary">
            {loading ? 'Running Backtest...' : 'Run Backtest'}
          </Button>
        </div>
      </Card>

      {/* Results */}
      {result && result.status === 'completed' && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <p className="text-sm text-slate-400">Total Trades</p>
              <p className="text-2xl font-bold mt-1">{result.metrics?.total_trades || 0}</p>
            </Card>
            <Card>
              <p className="text-sm text-slate-400">Win Rate</p>
              <p className="text-2xl font-bold text-green-500 mt-1">
                {result.metrics?.win_rate ? result.metrics.win_rate.toFixed(1) : '0'}%
              </p>
            </Card>
            <Card>
              <p className="text-sm text-slate-400">Total PnL</p>
              <p className={`text-2xl font-bold mt-1 ${(result.metrics?.total_pnl || 0) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                ${result.metrics?.total_pnl ? result.metrics.total_pnl.toFixed(2) : '0.00'}
              </p>
            </Card>
            <Card>
              <p className="text-sm text-slate-400">Sharpe Ratio</p>
              <p className="text-2xl font-bold mt-1">
                {result.metrics?.sharpe_ratio?.toFixed(2) || 'N/A'}
              </p>
            </Card>
          </div>

          <Card title="Trade History">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-slate-800">
                    <th className="text-left py-3 px-4 text-sm text-slate-400">Time</th>
                    <th className="text-left py-3 px-4 text-sm text-slate-400">Side</th>
                    <th className="text-left py-3 px-4 text-sm text-slate-400">Action</th>
                    <th className="text-left py-3 px-4 text-sm text-slate-400">Price</th>
                    <th className="text-left py-3 px-4 text-sm text-slate-400">PnL %</th>
                  </tr>
                </thead>
                <tbody>
                  {result.trades?.slice(0, 10).map((trade, idx) => (
                    <tr key={idx} className="border-b border-slate-800/50">
                      <td className="py-3 px-4 text-sm">{new Date(trade.timestamp).toLocaleString()}</td>
                      <td className="py-3 px-4 text-sm">{trade.side}</td>
                      <td className="py-3 px-4 text-sm">{trade.action}</td>
                      <td className="py-3 px-4 text-sm">${trade.price.toFixed(2)}</td>
                      <td className={`py-3 px-4 text-sm ${(trade.pnl_percentage || 0) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                        {trade.pnl_percentage?.toFixed(2) || '-'}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {result && result.status === 'failed' && (
        <Card>
          <p className="text-red-500">Backtest failed: {result.error}</p>
        </Card>
      )}
    </div>
  );
}
