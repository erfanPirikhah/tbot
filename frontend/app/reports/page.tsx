'use client';

import { useState, useEffect } from 'react';
import Card from '@/components/ui/Card';
import { getPerformance, getDailyStats } from '@/lib/api';
import { PerformanceMetrics, DailyStats } from '@/lib/types';

export default function ReportsPage() {
  const [performance, setPerformance] = useState<PerformanceMetrics | null>(null);
  const [dailyStats, setDailyStats] = useState<DailyStats[]>([]);
  const [days, setDays] = useState(30);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [perfRes, statsRes] = await Promise.all([
          getPerformance(),
          getDailyStats(days),
        ]);
        setPerformance(perfRes.data);
        setDailyStats(statsRes.data.stats || []);
      } catch (error) {
        console.error('Error fetching reports:', error);
      }
    };

    fetchData();
  }, [days]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold mb-2">Reports & Analytics</h1>
        <p className="text-slate-400">Performance metrics and trading statistics</p>
      </div>

      {/* Performance Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <p className="text-sm text-slate-400">Total Trades</p>
          <p className="text-2xl font-bold mt-1">{performance?.total_trades || 0}</p>
        </Card>
        <Card>
          <p className="text-sm text-slate-400">Win Rate</p>
          <p className="text-2xl font-bold text-green-500 mt-1">
            {performance?.win_rate.toFixed(1) || 0}%
          </p>
        </Card>
        <Card>
          <p className="text-sm text-slate-400">Total PnL</p>
          <p className={`text-2xl font-bold mt-1 ${(performance?.total_pnl || 0) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
            ${performance?.total_pnl.toFixed(2) || 0}
          </p>
        </Card>
        <Card>
          <p className="text-sm text-slate-400">Profit Factor</p>
          <p className="text-2xl font-bold mt-1">
            {performance?.profit_factor.toFixed(2) || 0}
          </p>
        </Card>
      </div>

      {/* Detailed Metrics */}
      <Card title="Detailed Performance">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          <div>
            <p className="text-sm text-slate-400">Wins</p>
            <p className="text-xl font-bold text-green-500 mt-1">{performance?.wins || 0}</p>
          </div>
          <div>
            <p className="text-sm text-slate-400">Losses</p>
            <p className="text-xl font-bold text-red-500 mt-1">{performance?.losses || 0}</p>
          </div>
          <div>
            <p className="text-sm text-slate-400">Avg Win</p>
            <p className="text-xl font-bold text-green-500 mt-1">
              ${performance?.avg_win.toFixed(2) || 0}
            </p>
          </div>
          <div>
            <p className="text-sm text-slate-400">Avg Loss</p>
            <p className="text-xl font-bold text-red-500 mt-1">
              ${performance?.avg_loss.toFixed(2) || 0}
            </p>
          </div>
        </div>
      </Card>

      {/* Daily Statistics */}
      <Card title="Daily Statistics">
        <div className="mb-4">
          <label className="block text-sm text-slate-400 mb-2">Days to Show</label>
          <select
            value={days}
            onChange={(e) => setDays(parseInt(e.target.value))}
            className="bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white"
          >
            <option value={7}>Last 7 Days</option>
            <option value={30}>Last 30 Days</option>
            <option value={90}>Last 90 Days</option>
          </select>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-800">
                <th className="text-left py-3 px-4 text-sm text-slate-400">Date</th>
                <th className="text-left py-3 px-4 text-sm text-slate-400">Trades</th>
                <th className="text-left py-3 px-4 text-sm text-slate-400">Wins</th>
                <th className="text-left py-3 px-4 text-sm text-slate-400">Losses</th>
                <th className="text-left py-3 px-4 text-sm text-slate-400">PnL</th>
              </tr>
            </thead>
            <tbody>
              {dailyStats.map((stat, idx) => (
                <tr key={idx} className="border-b border-slate-800/50">
                  <td className="py-3 px-4 text-sm">{stat.date}</td>
                  <td className="py-3 px-4 text-sm">{stat.trades}</td>
                  <td className="py-3 px-4 text-sm text-green-500">{stat.wins}</td>
                  <td className="py-3 px-4 text-sm text-red-500">{stat.losses}</td>
                  <td className={`py-3 px-4 text-sm ${stat.pnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                    ${stat.pnl.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
