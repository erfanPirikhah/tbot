'use client';

import { useState, useEffect } from 'react';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import { getRiskConfig, updateRiskConfig, retrainML } from '@/lib/api';

export default function ConfigPage() {
  const [riskConfig, setRiskConfig] = useState({
    risk_per_trade: 0.015,
    max_position_size: 0.1,
    max_daily_loss: 0.05,
  });
  const [mlSymbol, setMlSymbol] = useState('BTCUSDT');
  const [mlTimeframe, setMlTimeframe] = useState('1h');
  const [mlDays, setMlDays] = useState(90);
  const [loading, setLoading] = useState(false);
  const [mlLoading, setMlLoading] = useState(false);
  const [message, setMessage] = useState('');

  useEffect(() => {
    getRiskConfig().then(res => setRiskConfig(res.data));
  }, []);

  const handleUpdateRisk = async () => {
    setLoading(true);
    setMessage('');
    try {
      await updateRiskConfig(riskConfig);
      setMessage('Risk configuration updated successfully!');
    } catch (error) {
      setMessage('Error updating risk configuration');
    } finally {
      setLoading(false);
    }
  };

  const handleRetrainML = async () => {
    setMlLoading(true);
    setMessage('');
    try {
      const res = await retrainML({ symbol: mlSymbol, timeframe: mlTimeframe, days: mlDays });
      setMessage(res.data.message || 'ML model retrained successfully!');
    } catch (error) {
      setMessage('Error retraining ML model');
    } finally {
      setMlLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold mb-2">Configuration</h1>
        <p className="text-slate-400">Manage risk settings and ML model</p>
      </div>

      {message && (
        <div className={`p-4 rounded-lg ${message.includes('Error') ? 'bg-red-900/50 text-red-200' : 'bg-green-900/50 text-green-200'}`}>
          {message}
        </div>
      )}

      {/* Risk Management */}
      <Card title="Risk Management">
        <div className="space-y-4">
          <div>
            <label className="block text-sm text-slate-400 mb-2">
              Risk Per Trade (0-5%)
            </label>
            <input
              type="number"
              step="0.001"
              min="0"
              max="0.05"
              value={riskConfig.risk_per_trade}
              onChange={(e) => setRiskConfig({ ...riskConfig, risk_per_trade: parseFloat(e.target.value) })}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white"
            />
            <p className="text-xs text-slate-500 mt-1">
              Current: {(riskConfig.risk_per_trade * 100).toFixed(2)}%
            </p>
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-2">
              Max Position Size (0-100%)
            </label>
            <input
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={riskConfig.max_position_size}
              onChange={(e) => setRiskConfig({ ...riskConfig, max_position_size: parseFloat(e.target.value) })}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white"
            />
            <p className="text-xs text-slate-500 mt-1">
              Current: {(riskConfig.max_position_size * 100).toFixed(0)}%
            </p>
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-2">
              Max Daily Loss (0-20%)
            </label>
            <input
              type="number"
              step="0.01"
              min="0"
              max="0.2"
              value={riskConfig.max_daily_loss}
              onChange={(e) => setRiskConfig({ ...riskConfig, max_daily_loss: parseFloat(e.target.value) })}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white"
            />
            <p className="text-xs text-slate-500 mt-1">
              Current: {(riskConfig.max_daily_loss * 100).toFixed(0)}%
            </p>
          </div>
          <Button onClick={handleUpdateRisk} disabled={loading}>
            {loading ? 'Updating...' : 'Update Risk Settings'}
          </Button>
        </div>
      </Card>

      {/* ML Model Retraining */}
      <Card title="ML Model Retraining">
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm text-slate-400 mb-2">Symbol</label>
              <input
                type="text"
                value={mlSymbol}
                onChange={(e) => setMlSymbol(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white"
              />
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-2">Timeframe</label>
              <select
                value={mlTimeframe}
                onChange={(e) => setMlTimeframe(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white"
              >
                <option value="1h">1 Hour</option>
                <option value="4h">4 Hours</option>
                <option value="1d">1 Day</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-2">Days of Data</label>
              <input
                type="number"
                value={mlDays}
                onChange={(e) => setMlDays(parseInt(e.target.value))}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white"
              />
            </div>
          </div>
          <Button onClick={handleRetrainML} disabled={mlLoading} variant="primary">
            {mlLoading ? 'Retraining...' : 'Retrain ML Model'}
          </Button>
          <p className="text-sm text-slate-400">
            This will fetch historical data and retrain the ML regime detection model. This may take several minutes.
          </p>
        </div>
      </Card>
    </div>
  );
}
