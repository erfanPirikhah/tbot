'use client';

import { useState, useEffect } from 'react';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import { startTrading, stopTrading, getTradingStatus, getPositions, getTradeHistory } from '@/lib/api';
import { TradingStatus, Position, Trade } from '@/lib/types';
import { Play, Square } from 'lucide-react';

export default function TradingPage() {
  const [status, setStatus] = useState<TradingStatus | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [history, setHistory] = useState<Trade[]>([]);
  const [symbol, setSymbol] = useState('BTCUSDT');
  const [timeframe, setTimeframe] = useState('1h');
  const [loading, setLoading] = useState(false);

  const fetchData = async () => {
    try {
      const [statusRes, posRes, histRes] = await Promise.all([
        getTradingStatus(),
        getPositions(),
        getTradeHistory(20),
      ]);
      setStatus(statusRes.data);
      setPositions(posRes.data.positions || []);
      setHistory(histRes.data.trades || []);
    } catch (error) {
      console.error('Error fetching trading data:', error);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000); // Refresh every 5s
    return () => clearInterval(interval);
  }, []);

  const handleStart = async () => {
    setLoading(true);
    try {
      await startTrading({ symbol, timeframe });
      await fetchData();
    } catch (error) {
      console.error('Error starting trading:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    setLoading(true);
    try {
      await stopTrading();
      await fetchData();
    } catch (error) {
      console.error('Error stopping trading:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold mb-2">Live Trading</h1>
        <p className="text-slate-400">Control and monitor live trading operations</p>
      </div>

      {/* Controls */}
      <Card title="Trading Controls">
        <div className="flex flex-wrap gap-4 items-end">
          <div>
            <label className="block text-sm text-slate-400 mb-2">Symbol</label>
            <input
              type="text"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white"
              disabled={status?.is_running}
            />
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-2">Timeframe</label>
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              className="bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white"
              disabled={status?.is_running}
            >
              <option value="1h">1 Hour</option>
              <option value="4h">4 Hours</option>
            </select>
          </div>
          <div className="flex gap-2">
            <Button
              onClick={handleStart}
              disabled={loading || status?.is_running}
              variant="success"
            >
              <Play size={16} className="mr-2" />
              Start
            </Button>
            <Button
              onClick={handleStop}
              disabled={loading || !status?.is_running}
              variant="danger"
            >
              <Square size={16} className="mr-2" />
              Stop
            </Button>
          </div>
        </div>
      </Card>

      {/* Status */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <p className="text-sm text-slate-400">Status</p>
          <p className={`text-2xl font-bold mt-1 ${status?.is_running ? 'text-green-500' : 'text-red-500'}`}>
            {status?.is_running ? 'Running' : 'Stopped'}
          </p>
        </Card>
        <Card>
          <p className="text-sm text-slate-400">Portfolio Value</p>
          <p className="text-2xl font-bold mt-1">
            ${status?.portfolio_value.toFixed(2) || '0.00'}
          </p>
        </Card>
        <Card>
          <p className="text-sm text-slate-400">Position</p>
          <p className="text-2xl font-bold mt-1">{status?.current_position || 'OUT'}</p>
        </Card>
        <Card>
          <p className="text-sm text-slate-400">Open Positions</p>
          <p className="text-2xl font-bold mt-1">{positions.length}</p>
        </Card>
      </div>

      {/* Open Positions */}
      {positions.length > 0 && (
        <Card title="Open Positions">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-slate-800">
                  <th className="text-left py-3 px-4 text-sm text-slate-400">Symbol</th>
                  <th className="text-left py-3 px-4 text-sm text-slate-400">Side</th>
                  <th className="text-left py-3 px-4 text-sm text-slate-400">Entry</th>
                  <th className="text-left py-3 px-4 text-sm text-slate-400">Current</th>
                  <th className="text-left py-3 px-4 text-sm text-slate-400">PnL</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((pos, idx) => (
                  <tr key={idx} className="border-b border-slate-800/50">
                    <td className="py-3 px-4">{pos.symbol}</td>
                    <td className="py-3 px-4">{pos.side}</td>
                    <td className="py-3 px-4">${pos.entry_price.toFixed(2)}</td>
                    <td className="py-3 px-4">${pos.current_price.toFixed(2)}</td>
                    <td className={`py-3 px-4 ${pos.pnl_percentage >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                      {pos.pnl_percentage.toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Trade History */}
      <Card title="Recent Trades">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-800">
                <th className="text-left py-3 px-4 text-sm text-slate-400">Time</th>
                <th className="text-left py-3 px-4 text-sm text-slate-400">Symbol</th>
                <th className="text-left py-3 px-4 text-sm text-slate-400">Side</th>
                <th className="text-left py-3 px-4 text-sm text-slate-400">Action</th>
                <th className="text-left py-3 px-4 text-sm text-slate-400">Price</th>
              </tr>
            </thead>
            <tbody>
              {history.map((trade, idx) => (
                <tr key={idx} className="border-b border-slate-800/50">
                  <td className="py-3 px-4 text-sm">{new Date(trade.timestamp).toLocaleString()}</td>
                  <td className="py-3 px-4 text-sm">{trade.symbol}</td>
                  <td className="py-3 px-4 text-sm">{trade.side}</td>
                  <td className="py-3 px-4 text-sm">{trade.action}</td>
                  <td className="py-3 px-4 text-sm">${trade.price.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
