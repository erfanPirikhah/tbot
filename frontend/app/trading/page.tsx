'use client';

import { useState, useEffect } from 'react';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import {
    startTrading, stopTrading, getTradingStatus, getPositions,
    getTradeHistory, getSymbols
} from '@/lib/api';
import { TradingStatus, Position, Trade } from '@/lib/types';
import { Play, Square, RefreshCw, TrendingUp, AlertCircle } from 'lucide-react';

interface SymbolsData {
  symbols: {
    crypto: string[];
    forex: string[];
  };
}

export default function TradingPage() {
  const [status, setStatus] = useState<TradingStatus | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [history, setHistory] = useState<Trade[]>([]);
  const [symbol, setSymbol] = useState('EURUSD');
  const [timeframe, setTimeframe] = useState('1h');
  const [loading, setLoading] = useState(false);
  const [symbolsData, setSymbolsData] = useState<SymbolsData | null>(null);
  const [symbolCategory, setSymbolCategory] = useState<'forex' | 'crypto'>('forex');
  const [error, setError] = useState<string | null>(null);

  // Fetch symbols on load
  useEffect(() => {
    const loadSymbols = async () => {
      try {
        const res = await getSymbols();
        setSymbolsData(res.data);
      } catch (err) {
        console.error('Error loading symbols:', err);
      }
    };
    loadSymbols();
  }, []);

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
      setError(null);
    } catch (err) {
      console.error('Error fetching trading data:', err);
      setError('خطا در دریافت اطلاعات');
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 3000); // Refresh every 3s
    return () => clearInterval(interval);
  }, []);

  const handleStart = async () => {
    setLoading(true);
    setError(null);
    try {
      await startTrading({ symbol, timeframe });
      await fetchData();
    } catch (err: any) {
      console.error('Error starting trading:', err);
      setError(err?.response?.data?.detail || 'خطا در شروع معامله');
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    setLoading(true);
    try {
      await stopTrading();
      await fetchData();
    } catch (err) {
      console.error('Error stopping trading:', err);
    } finally {
      setLoading(false);
    }
  };

  // Format uptime
  const formatUptime = (seconds: number | undefined) => {
    if (!seconds) return '0s';
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    if (hours > 0) return `${hours}h ${mins}m`;
    if (mins > 0) return `${mins}m ${secs}s`;
    return `${secs}s`;
  };

  const currentSymbols = symbolsData?.symbols[symbolCategory] || [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2">🔴 Live Trading</h1>
          <p className="text-slate-400">کنترل و مانیتور معاملات زنده</p>
        </div>
        <Button onClick={fetchData} variant="secondary" className="flex items-center gap-2">
          <RefreshCw size={16} />
          Refresh
        </Button>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-4 flex items-center gap-3">
          <AlertCircle className="text-red-500" size={20} />
          <span className="text-red-400">{error}</span>
        </div>
      )}

      {/* Controls */}
      <Card title="⚙️ Trading Controls">
        <div className="flex flex-wrap gap-4 items-end">
          {/* Category Selector */}
          <div>
            <label className="block text-sm text-slate-400 mb-2">Market Type</label>
            <select
              value={symbolCategory}
              onChange={(e) => {
                setSymbolCategory(e.target.value as 'forex' | 'crypto');
                // Reset to first symbol in category
                const newSymbols = symbolsData?.symbols[e.target.value as 'forex' | 'crypto'] || [];
                if (newSymbols.length > 0) setSymbol(newSymbols[0]);
              }}
              className="bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-white min-w-[120px]"
              disabled={status?.is_running}
            >
              <option value="forex">🌍 Forex</option>
              <option value="crypto">₿ Crypto</option>
            </select>
          </div>

          {/* Symbol Selector */}
          <div>
            <label className="block text-sm text-slate-400 mb-2">Symbol</label>
            <select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-white min-w-[150px]"
              disabled={status?.is_running}
            >
              {currentSymbols.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>

          {/* Timeframe Selector */}
          <div>
            <label className="block text-sm text-slate-400 mb-2">Timeframe</label>
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              className="bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-white min-w-[120px]"
              disabled={status?.is_running}
            >
              <option value="1m">1 دقیقه</option>
              <option value="5m">5 دقیقه</option>
              <option value="15m">15 دقیقه</option>
              <option value="1h">1 ساعت</option>
              <option value="4h">4 ساعت</option>
            </select>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2">
            <Button
              onClick={handleStart}
              disabled={loading || status?.is_running}
              variant="success"
              className="px-6"
            >
              <Play size={16} className="mr-2" />
              شروع
            </Button>
            <Button
              onClick={handleStop}
              disabled={loading || !status?.is_running}
              variant="danger"
              className="px-6"
            >
              <Square size={16} className="mr-2" />
              توقف
            </Button>
          </div>
        </div>
      </Card>

      {/* Status Dashboard */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <Card>
          <p className="text-sm text-slate-400">وضعیت</p>
          <div className="flex items-center gap-2 mt-1">
            <div className={`w-3 h-3 rounded-full ${status?.is_running ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
            <p className={`text-xl font-bold ${status?.is_running ? 'text-green-500' : 'text-red-500'}`}>
              {status?.is_running ? 'فعال' : 'غیرفعال'}
            </p>
          </div>
        </Card>
        <Card>
          <p className="text-sm text-slate-400">ارزش پورتفو</p>
          <p className="text-xl font-bold mt-1 text-blue-400">
            ${status?.portfolio_value?.toFixed(2) || '10,000.00'}
          </p>
        </Card>
        <Card>
          <p className="text-sm text-slate-400">پوزیشن فعلی</p>
          <p className={`text-xl font-bold mt-1 ${
            status?.current_position === 'LONG' ? 'text-green-500' : 
            status?.current_position === 'SHORT' ? 'text-red-500' : 'text-slate-400'
          }`}>
            {status?.current_position === 'LONG' ? '📈 Long' : 
             status?.current_position === 'SHORT' ? '📉 Short' : '⏸️ Out'}
          </p>
        </Card>
        <Card>
          <p className="text-sm text-slate-400">سمبل فعال</p>
          <p className="text-xl font-bold mt-1">
            {status?.active_symbols?.[0] || '-'}
          </p>
        </Card>
        <Card>
          <p className="text-sm text-slate-400">زمان فعالیت</p>
          <p className="text-xl font-bold mt-1 text-purple-400">
            {formatUptime(status?.uptime_seconds)}
          </p>
        </Card>
      </div>

      {/* Open Positions */}
      {positions.length > 0 && (
        <Card title="📊 پوزیشن‌های باز">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-slate-800">
                  <th className="text-left py-3 px-4 text-sm text-slate-400">سمبل</th>
                  <th className="text-left py-3 px-4 text-sm text-slate-400">نوع</th>
                  <th className="text-left py-3 px-4 text-sm text-slate-400">ورود</th>
                  <th className="text-left py-3 px-4 text-sm text-slate-400">فعلی</th>
                  <th className="text-left py-3 px-4 text-sm text-slate-400">SL</th>
                  <th className="text-left py-3 px-4 text-sm text-slate-400">TP</th>
                  <th className="text-left py-3 px-4 text-sm text-slate-400">سود/ضرر</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((pos, idx) => (
                  <tr key={idx} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                    <td className="py-3 px-4 font-medium">{pos.symbol}</td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-1 rounded text-xs font-bold ${
                        pos.side === 'LONG' ? 'bg-green-500/20 text-green-500' : 'bg-red-500/20 text-red-500'
                      }`}>
                        {pos.side}
                      </span>
                    </td>
                    <td className="py-3 px-4">${pos.entry_price.toFixed(5)}</td>
                    <td className="py-3 px-4">${pos.current_price.toFixed(5)}</td>
                    <td className="py-3 px-4 text-red-400">${pos.stop_loss.toFixed(5)}</td>
                    <td className="py-3 px-4 text-green-400">${pos.take_profit.toFixed(5)}</td>
                    <td className={`py-3 px-4 font-bold ${pos.pnl_percentage >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                      {pos.pnl_percentage >= 0 ? '+' : ''}{pos.pnl_percentage.toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Trade History */}
      <Card title="📜 تاریخچه معاملات">
        {history.length === 0 ? (
          <div className="text-center py-8 text-slate-500">
            <TrendingUp size={48} className="mx-auto mb-4 opacity-50" />
            <p>هنوز معامله‌ای انجام نشده</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-slate-800">
                  <th className="text-left py-3 px-4 text-sm text-slate-400">زمان</th>
                  <th className="text-left py-3 px-4 text-sm text-slate-400">سمبل</th>
                  <th className="text-left py-3 px-4 text-sm text-slate-400">نوع</th>
                  <th className="text-left py-3 px-4 text-sm text-slate-400">اکشن</th>
                  <th className="text-left py-3 px-4 text-sm text-slate-400">قیمت</th>
                  <th className="text-left py-3 px-4 text-sm text-slate-400">دلیل</th>
                </tr>
              </thead>
              <tbody>
                {history.map((trade, idx) => (
                  <tr key={idx} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                    <td className="py-3 px-4 text-sm text-slate-300">
                      {new Date(trade.timestamp).toLocaleString('fa-IR')}
                    </td>
                    <td className="py-3 px-4 text-sm font-medium">{trade.symbol}</td>
                    <td className="py-3 px-4 text-sm">
                      <span className={`px-2 py-1 rounded text-xs ${
                        trade.side === 'LONG' ? 'bg-green-500/20 text-green-500' : 'bg-red-500/20 text-red-500'
                      }`}>
                        {trade.side}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-sm">
                      <span className={`px-2 py-1 rounded text-xs ${
                        trade.action === 'ENTRY' ? 'bg-blue-500/20 text-blue-400' : 'bg-yellow-500/20 text-yellow-400'
                      }`}>
                        {trade.action === 'ENTRY' ? 'ورود' : 'خروج'}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-sm">${trade.price.toFixed(5)}</td>
                    <td className="py-3 px-4 text-sm text-slate-400 max-w-[200px] truncate">
                      {trade.reason || '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
