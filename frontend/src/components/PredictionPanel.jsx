import { motion, AnimatePresence } from 'framer-motion';
import { MapPin, Wind, Thermometer, Droplets, Cloud, Clock, Cpu, Eye } from 'lucide-react';
import { format } from 'date-fns';
import RainfallGauge from './RainfallGauge';
import ForecastChart from './ForecastChart';

const CAT_COLORS = {
  'No Rain': '#475569', Light: '#00C9A7', Moderate: '#00B37E',
  Heavy: '#FFD166', 'Very Heavy': '#FF6B35', 'Extremely Heavy': '#FF3B5C',
};

function MeteoCard({ icon: Icon, label, value, color }) {
  return (
    <div className="flex items-center gap-2 p-2.5 rounded-lg" style={{ background: 'rgba(15,42,61,0.5)' }}>
      <Icon size={12} style={{ color: color || '#00C9A7' }} />
      <div>
        <p className="text-[9px] font-mono text-slate-500 uppercase tracking-wider">{label}</p>
        <p className="text-xs font-mono font-semibold" style={{ color: '#B0C4D8' }}>{value}</p>
      </div>
    </div>
  );
}

export default function PredictionPanel({ result, loading }) {
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-14 gap-4">
        <div className="relative w-10 h-10">
          <div className="absolute inset-0 rounded-full border-2 border-[#00C9A7] opacity-20 animate-ping" />
          <div className="absolute inset-1.5 rounded-full border-2 border-[#00E5FF] animate-spin border-t-transparent" />
          <Cpu size={14} className="absolute inset-0 m-auto text-[#00C9A7]" />
        </div>
        <div className="text-center">
          <p className="text-sm font-mono text-[#00C9A7]">Running ensemble models...</p>
          <p className="text-[10px] font-mono text-slate-500 mt-1">BiLSTM + XGBoost + LightGBM</p>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center py-14 gap-3 text-center">
        <div className="w-14 h-14 rounded-full border border-[#0F2A3D] flex items-center justify-center">
          <Droplets size={22} className="text-slate-600" />
        </div>
        <p className="text-sm font-mono text-slate-400">Select a location on the map</p>
        <p className="text-[10px] font-mono text-slate-600">or search by city name</p>
      </div>
    );
  }

  const today = result.forecast[0];
  const peakDay = result.forecast.reduce((a, b) => a.predicted_rainfall_mm > b.predicted_rainfall_mm ? a : b);
  const maxMm = peakDay.predicted_rainfall_mm;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={result.request_id}
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -16 }}
        transition={{ duration: 0.4 }}
        className="space-y-4">

        {/* Location */}
        <div className="flex items-start gap-2">
          <MapPin size={13} className="text-[#00C9A7] mt-0.5 shrink-0" />
          <div>
            <p className="text-sm font-semibold text-slate-200">{result.location}</p>
            <p className="text-[9px] font-mono text-slate-500">
              {result.latitude.toFixed(4)}°, {result.longitude.toFixed(4)}°
            </p>
          </div>
        </div>

        {/* Gauge */}
        <div className="flex justify-center py-1">
          <RainfallGauge maxRainfallMm={maxMm} riskLevel={result.overall_risk} season={result.season} />
        </div>

        {/* Today's forecast card */}
        {today && (
          <div className="rounded-xl p-3 space-y-2"
            style={{
              background: `${CAT_COLORS[today.rainfall_category] || '#475569'}10`,
              border: `1px solid ${CAT_COLORS[today.rainfall_category] || '#475569'}35`,
            }}>
            <div className="flex justify-between items-center">
              <div>
                <p className="text-[9px] font-mono text-slate-500 uppercase tracking-widest">Today</p>
                <p className="text-xl font-bold font-mono" style={{ color: CAT_COLORS[today.rainfall_category] }}>
                  {today.predicted_rainfall_mm.toFixed(1)} mm
                </p>
                <p className="text-xs font-mono" style={{ color: CAT_COLORS[today.rainfall_category] }}>
                  {today.rainfall_category}
                </p>
              </div>
              <div className="text-right">
                <p className="text-[9px] font-mono text-slate-500 mb-1">Rain Probability</p>
                <p className="text-2xl font-bold font-mono text-[#00E5FF]">
                  {(today.probability_of_rain * 100).toFixed(0)}%
                </p>
              </div>
            </div>

            {/* Meteo quick stats */}
            <div className="grid grid-cols-2 gap-1.5 pt-1">
              {today.temperature_max_c != null && (
                <MeteoCard icon={Thermometer} label="Max Temp" value={`${today.temperature_max_c.toFixed(1)}°C`} color="#FFD166" />
              )}
              {today.humidity_percent != null && (
                <MeteoCard icon={Droplets} label="Humidity" value={`${today.humidity_percent?.toFixed(0)}%`} color="#00C9A7" />
              )}
              {today.wind_speed_kmh != null && (
                <MeteoCard icon={Wind} label="Wind" value={`${today.wind_speed_kmh?.toFixed(0)} km/h`} color="#7FDBCA" />
              )}
              {today.pressure_hpa != null && (
                <MeteoCard icon={Cloud} label="Pressure" value={`${today.pressure_hpa?.toFixed(0)} hPa`} color="#94A3B8" />
              )}
            </div>
          </div>
        )}

        {/* 7-day chart */}
        <ForecastChart forecast={result.forecast} />

        {/* Forecast list */}
        <div className="space-y-1 max-h-36 overflow-y-auto pr-0.5">
          {result.forecast.map((day, i) => {
            const color = CAT_COLORS[day.rainfall_category] || '#475569';
            return (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.05 }}
                className="flex items-center justify-between py-1.5 px-2 rounded-lg"
                style={{ background: 'rgba(15,42,61,0.35)' }}>
                <span className="text-[10px] font-mono text-slate-400 w-16">
                  {format(new Date(day.date), 'EEE dd')}
                </span>
                <div className="flex-1 mx-2">
                  <div className="h-1 rounded-full bg-[#0F2A3D] overflow-hidden">
                    <div className="h-full rounded-full transition-all duration-700"
                      style={{
                        width: `${Math.min(100, (day.predicted_rainfall_mm / 200) * 100)}%`,
                        background: color,
                      }} />
                  </div>
                </div>
                <span className="text-[10px] font-mono font-bold w-14 text-right" style={{ color }}>
                  {day.predicted_rainfall_mm.toFixed(1)} mm
                </span>
                <span className="text-[9px] font-mono w-8 text-right text-slate-500">
                  {(day.probability_of_rain * 100).toFixed(0)}%
                </span>
              </motion.div>
            );
          })}
        </div>

        {/* Footer */}
        <div className="flex justify-between text-[9px] font-mono text-slate-600 pt-1 border-t border-[#0F2A3D]">
          <span>v{result.model_version} • BiLSTM+XGB+LGBM</span>
          <span className="flex items-center gap-1">
            <Clock size={8} />
            {result.processing_time_ms.toFixed(0)}ms
          </span>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
