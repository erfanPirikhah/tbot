'use client';

import { useEffect, useState } from 'react';
import Card from '@/components/ui/Card';
import { getTradingStatus, getPerformance, getSystemHealth } from '@/lib/api';
import { TradingStatus, PerformanceMetrics } from '@/lib/types';
import { TrendingUp, DollarSign, Activity, CheckCircle } from 'lucide-react';

export default function Dashboard() {
  const [status, setStatus] = useState<TradingStatus | null>(null);
  const [performance, setPerformance] = useState<PerformanceMetrics | null>(null);
  const [systemHealth, setSystemHealth] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statusRes, perfRes, healthRes] = await Promise.all([
          getTradingStatus(),
          getPerformance(),
          getSystemHealth(),
        ]);
        setStatus(statusRes.data);
        setPerformance(perfRes.data);
        setSystemHealth(healthRes.data);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold mb-2">Dashboard</h1>
        <p className="text-slate-400">Welcome to your AI-powered trading bot</p>
      </div>

      {/* System Status */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-400">System Status</p>
              <p className="text-2xl font-bold text-green-500 mt-1">
                {systemHealth?.status || 'Online'}
              </p>
            </div>
            <CheckCircle className="text-green-500" size={32} />
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-400">Portfolio Value</p>
              <p className="text-2xl font-bold mt-1">
                ${status?.portfolio_value.toFixed(2) || '0.00'}
              </p>
            </div>
            <DollarSign className="text-blue-500" size={32} />
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-400">Win Rate</p>
              <p className="text-2xl font-bold text-green-500 mt-1">
                {performance?.win_rate.toFixed(1) || '0'}%
              </p>
            </div>
            <TrendingUp className="text-green-500" size={32} />
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-400">Total Trades</p>
              <p className="text-2xl font-bold mt-1">
                {performance?.total_trades || 0}
              </p>
            </div>
            <Activity className="text-purple-500" size={32} />
          </div>
        </Card>
      </div>

      {/* Trading Status */}
      <Card title="Trading Status">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-sm text-slate-400">Status</p>
            <p className={`text-lg font-semibold mt-1 ${status?.is_running ? 'text-green-500' : 'text-red-500'}`}>
              {status?.is_running ? 'Running' : 'Stopped'}
            </p>
          </div>
          <div>
            <p className="text-sm text-slate-400">Position</p>
            <p className="text-lg font-semibold mt-1">{status?.current_position || 'OUT'}</p>
          </div>
          <div>
            <p className="text-sm text-slate-400">Active Symbols</p>
            <p className="text-lg font-semibold mt-1">{status?.active_symbols.length || 0}</p>
          </div>
          <div>
            <p className="text-sm text-slate-400">Uptime</p>
            <p className="text-lg font-semibold mt-1">
              {status?.uptime_seconds ? `${Math.floor(status.uptime_seconds / 60)}m` : '0m'}
            </p>
          </div>
        </div>
      </Card>

      {/* Performance Metrics */}
      <Card title="Performance Overview">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          <div>
            <p className="text-sm text-slate-400">Total PnL</p>
            <p className={`text-xl font-bold mt-1 ${(performance?.total_pnl || 0) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
              ${performance?.total_pnl.toFixed(2) || '0.00'}
            </p>
          </div>
          <div>
            <p className="text-sm text-slate-400">Avg Win</p>
            <p className="text-xl font-bold text-green-500 mt-1">
              ${performance?.avg_win.toFixed(2) || '0.00'}
            </p>
          </div>
          <div>
            <p className="text-sm text-slate-400">Avg Loss</p>
            <p className="text-xl font-bold text-red-500 mt-1">
              ${performance?.avg_loss.toFixed(2) || '0.00'}
            </p>
          </div>
          <div>
            <p className="text-sm text-slate-400">Profit Factor</p>
            <p className="text-xl font-bold mt-1">
              {performance?.profit_factor.toFixed(2) || '0.00'}
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
}
