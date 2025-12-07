'use client';

import { useState, useEffect } from 'react';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import { getMarketAnalysis, getSymbols } from '@/lib/api';
import { MarketAnalysis } from '@/lib/types';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

export default function MarketPage() {
  const [symbol, setSymbol] = useState('BTCUSDT');
  const [timeframe, setTimeframe] = useState('1h');
  const [analysis, setAnalysis] = useState<MarketAnalysis | null>(null);
  const [symbols, setSymbols] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getSymbols().then(res => setSymbols(res.data));
  }, []);

  const fetchAnalysis = async () => {
    setLoading(true);
    try {
      const res = await getMarketAnalysis(symbol, timeframe);
      setAnalysis(res.data);
    } catch (error) {
      console.error('Error fetching market analysis:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalysis();
    const interval = setInterval(fetchAnalysis, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [symbol, timeframe]);

  const getMarketStateIcon = (state: string) => {
    switch (state) {
      case 'BULLISH':
        return <TrendingUp className="text-green-500" size={32} />;
      case 'BEARISH':
        return <TrendingDown className="text-red-500" size={32} />;
      default:
        return <Minus className="text-yellow-500" size={32} />;
    }
  };

  const getMarketStateColor = (state: string) => {
    switch (state) {
      case 'BULLISH':
        return 'text-green-500';
      case 'BEARISH':
        return 'text-red-500';
      default:
        return 'text-yellow-500';
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold mb-2">Market Analysis</h1>
        <p className="text-slate-400">Real-time market data and ML-powered regime detection</p>
      </div>

      {/* Controls */}
      <Card>
        <div className="flex flex-wrap gap-4">
          <div>
            <label className="block text-sm text-slate-400 mb-2">Symbol</label>
            <select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white"
            >
              {symbols?.symbols?.crypto?.map((s: string) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-2">Timeframe</label>
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              className="bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white"
            >
              <option value="1m">1 Minute</option>
              <option value="5m">5 Minutes</option>
              <option value="15m">15 Minutes</option>
              <option value="1h">1 Hour</option>
              <option value="4h">4 Hours</option>
              <option value="1d">1 Day</option>
            </select>
          </div>
          <div className="flex items-end">
            <Button onClick={fetchAnalysis} disabled={loading}>
              {loading ? 'Loading...' : 'Refresh'}
            </Button>
          </div>
        </div>
      </Card>

      {/* Market State */}
      {analysis && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-400">Current Price</p>
                  <p className="text-3xl font-bold mt-1">
                    ${analysis.price.toFixed(2)}
                  </p>
                </div>
              </div>
            </Card>

            <Card>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-400">Market State</p>
                  <p className={`text-2xl font-bold mt-1 ${getMarketStateColor(analysis.market_state)}`}>
                    {analysis.market_state}
                  </p>
                </div>
                {getMarketStateIcon(analysis.market_state)}
              </div>
            </Card>

            <Card>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-400">ML Confidence</p>
                  <p className="text-2xl font-bold mt-1">
                    {(analysis.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
            </Card>
          </div>

          <Card title="Regime Analysis">
            <div className="space-y-4">
              <div>
                <p className="text-sm text-slate-400">Detected Regime</p>
                <p className="text-xl font-semibold mt-1">{analysis.regime}</p>
              </div>
              <div>
                <p className="text-sm text-slate-400">Last Updated</p>
                <p className="text-sm mt-1">{new Date(analysis.timestamp).toLocaleString()}</p>
              </div>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
