import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { CloudRain, Activity, Wifi, WifiOff, BarChart2, Map, Layers } from 'lucide-react';

import RainfallMap from '../components/RainfallMap';
import PredictionPanel from '../components/PredictionPanel';
import LocationSearch from '../components/LocationSearch';
import AlertToast from '../components/AlertToast';
import MetricsComparison from '../components/MetricsComparison';
import RainBackground from '../components/RainBackground';
import { predict, getHistorical, getModelMetrics, createAlertSocket } from '../services/api';

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN || 'pk.YOUR_TOKEN_HERE';
const TABS = [{ id: 'map', label: 'Forecast Map', icon: Map }, { id: 'metrics', label: 'Model Metrics', icon: BarChart2 }];

export default function Dashboard() {
  const [tab, setTab] = useState('map');
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [predResult, setPredResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [historical, setHistorical] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [wsConn, setWsConn] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const [stats, setStats] = useState({ req: 0, alerts: 0 });

  // WebSocket
  useEffect(() => {
    const ws = createAlertSocket(
      msg => { setAlerts(p => [...p, msg]); setStats(s => ({ ...s, alerts: s.alerts + 1 })); setWsConn(true); },
      () => setWsConn(false),
    );
    ws.onopen = () => setWsConn(true);
    return () => ws.close();
  }, []);

  // Metrics
  useEffect(() => { getModelMetrics().then(setMetrics).catch(console.error); }, []);

  const handleSelect = useCallback(async ({ latitude, longitude, locationName }) => {
    setSelectedLocation({ latitude, longitude });
    setLoading(true);
    setPredResult(null);
    setStats(s => ({ ...s, req: s.req + 1 }));
    try {
      const [pred, hist] = await Promise.all([
        predict({ latitude, longitude, locationName: locationName || '', forecastDays: 7 }),
        getHistorical({ location: locationName || 'location', lat: latitude, lon: longitude, days: 90 }),
      ]);
      setPredResult(pred);
      setHistorical(hist.records || []);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, []);

  // Rain intensity based on result
  const rainIntensity = predResult?.overall_risk === 'Extreme' ? 'heavy'
    : predResult?.overall_risk === 'High' ? 'moderate' : 'light';

  return (
    <div className="h-screen w-screen bg-deep grid-bg overflow-hidden flex flex-col relative">
      <RainBackground intensity={rainIntensity} />

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="shrink-0 flex items-center justify-between px-5 py-3 z-10 relative"
        style={{ borderBottom: '1px solid rgba(15,42,61,0.9)', background: 'rgba(5,13,24,0.95)' }}>
        
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center relative"
            style={{ background: 'rgba(0,201,167,0.08)', border: '1px solid rgba(0,201,167,0.25)' }}>
            <CloudRain size={15} className="text-[#00C9A7]" />
            <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-[#00B37E] animate-pulse" />
          </div>
          <div>
            <h1 className="text-sm font-bold font-display tracking-tight text-slate-100">
              Hydro<span className="text-teal-glow">AI</span>
            </h1>
            <p className="text-[9px] font-mono text-slate-600 tracking-widest uppercase">Rainfall Prediction System</p>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 p-1 rounded-lg" style={{ background: 'rgba(7,21,37,0.9)', border: '1px solid rgba(15,42,61,0.7)' }}>
          {TABS.map(t => {
            const Icon = t.icon;
            const active = tab === t.id;
            return (
              <button key={t.id} onClick={() => setTab(t.id)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all"
                style={{
                  background: active ? 'rgba(0,201,167,0.1)' : 'transparent',
                  color: active ? '#00C9A7' : '#475569',
                  border: active ? '1px solid rgba(0,201,167,0.2)' : '1px solid transparent',
                }}>
                <Icon size={11} />{t.label}
              </button>
            );
          })}
        </div>

        {/* Status */}
        <div className="flex items-center gap-4 text-[10px] font-mono text-slate-500">
          <span><span className="text-[#00C9A7]">{stats.req}</span> predictions</span>
          <span><span className="text-[#FF6B35]">{stats.alerts}</span> alerts</span>
          <div className="flex items-center gap-1" style={{ color: wsConn ? '#00B37E' : '#FF3B5C' }}>
            {wsConn ? <Wifi size={11} /> : <WifiOff size={11} />}
            {wsConn ? 'Live' : 'Offline'}
          </div>
        </div>
      </header>

      {/* ── Body ───────────────────────────────────────────────────────────── */}
      {tab === 'map' && (
        <div className="flex-1 flex overflow-hidden relative z-10">
          {/* Map */}
          <div className="flex-1 relative p-3">
            <RainfallMap
              onLocationSelect={handleSelect}
              selectedLocation={selectedLocation}
              historicalData={historical}
              predictionResult={predResult}
            />

            {/* Search */}
            <div className="absolute top-6 left-6 right-72 max-w-xs z-20">
              <LocationSearch onSelect={handleSelect} mapboxToken={MAPBOX_TOKEN} />
            </div>

            {/* Map legend */}
            <div className="absolute bottom-6 left-6 glass rounded-lg px-3 py-2 z-10">
              <p className="text-[9px] font-mono text-slate-500 uppercase tracking-widest mb-1.5">Rainfall Intensity</p>
              {[['#00C9A7', 'None–Light'], ['#00B37E', 'Moderate'], ['#FFD166', 'Heavy'], ['#FF6B35', 'Very Heavy'], ['#FF3B5C', 'Extreme']].map(([c, l]) => (
                <div key={l} className="flex items-center gap-2 mb-0.5">
                  <span className="w-2 h-2 rounded-sm" style={{ background: c }} />
                  <span className="text-[9px] font-mono text-slate-400">{l}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Sidebar */}
          <div className="w-72 shrink-0 border-l border-[#0F2A3D] p-4 overflow-y-auto z-10"
            style={{ background: 'rgba(7,21,37,0.7)' }}>
            <div className="flex items-center gap-2 mb-4">
              <Activity size={11} className="text-[#00C9A7]" />
              <span className="text-[9px] font-mono text-slate-500 uppercase tracking-widest">Forecast Output</span>
            </div>
            <PredictionPanel result={predResult} loading={loading} />

            {/* Historical summary */}
            {historical.length > 0 && !loading && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-4 pt-4 border-t border-[#0F2A3D]">
                <p className="text-[9px] font-mono text-slate-500 uppercase tracking-widest mb-2 flex items-center gap-1.5">
                  <Layers size={9} />
                  Last 90 Days — {historical.length} records
                </p>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    { l: 'Total', v: `${historical.reduce((s,r) => s + r.rainfall_mm, 0).toFixed(0)}mm` },
                    { l: 'Peak', v: `${Math.max(...historical.map(r => r.rainfall_mm)).toFixed(0)}mm` },
                    { l: 'Rainy Days', v: historical.filter(r => r.rainfall_mm >= 2.4).length },
                    { l: 'Daily Avg', v: `${(historical.reduce((s,r) => s+r.rainfall_mm,0)/historical.length).toFixed(1)}mm` },
                  ].map(({ l, v }) => (
                    <div key={l} className="rounded-lg p-2.5 text-center" style={{ background: 'rgba(15,42,61,0.5)' }}>
                      <p className="text-[9px] font-mono text-slate-500">{l}</p>
                      <p className="text-sm font-mono font-bold text-[#00C9A7]">{v}</p>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </div>
        </div>
      )}

      {tab === 'metrics' && (
        <div className="flex-1 overflow-y-auto p-6 relative z-10">
          <div className="max-w-5xl mx-auto space-y-6">
            <div>
              <h2 className="text-lg font-bold font-display text-slate-100 mb-1">
                Model <span className="text-teal-glow">Performance</span> Dashboard
              </h2>
              <p className="text-xs font-mono text-slate-500">
                Ensemble: BiLSTM-Attention + XGBoost + LightGBM — evaluated on Open-Meteo historical data
              </p>
            </div>

            {/* KPI cards */}
            {metrics && (() => {
              const best = metrics.models.find(m => m.model_name === metrics.best_model);
              if (!best) return null;
              return (
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  {[
                    { l: 'MAE (mm)', v: best.mae.toFixed(2), unit: 'mm', color: '#00C9A7' },
                    { l: 'RMSE (mm)', v: best.rmse.toFixed(2), unit: 'mm', color: '#00E5FF' },
                    { l: 'R² Score', v: `${(best.r2_score * 100).toFixed(1)}%`, color: '#00B37E' },
                    { l: 'F1 Score', v: `${(best.f1_score * 100).toFixed(1)}%`, color: '#7FDBCA' },
                  ].map(({ l, v, color }) => (
                    <div key={l} className="glass rounded-xl p-4 text-center">
                      <p className="text-[9px] font-mono text-slate-500 uppercase tracking-widest mb-2">{l}</p>
                      <p className="text-3xl font-bold font-mono" style={{ color }}>{v}</p>
                      <p className="text-[9px] font-mono text-slate-600 mt-1">Ensemble model</p>
                    </div>
                  ))}
                </div>
              );
            })()}

            {metrics && <MetricsComparison models={metrics.models} />}
          </div>
        </div>
      )}

      <AlertToast alerts={alerts} />
    </div>
  );
}
