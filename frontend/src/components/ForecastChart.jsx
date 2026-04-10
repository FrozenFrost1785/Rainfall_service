import { useState } from 'react';
import { motion } from 'framer-motion';
import { format } from 'date-fns';
import { Droplets } from 'lucide-react';

const CATEGORY_CONFIG = {
  'No Rain':         { color: '#334155', glow: 'transparent',           label: 'None' },
  'Light':           { color: '#00C9A7', glow: 'rgba(0,201,167,0.4)',   label: 'Light' },
  'Moderate':        { color: '#00B37E', glow: 'rgba(0,179,126,0.4)',   label: 'Moderate' },
  'Heavy':           { color: '#FFD166', glow: 'rgba(255,209,102,0.4)', label: 'Heavy' },
  'Very Heavy':      { color: '#FF6B35', glow: 'rgba(255,107,53,0.4)',  label: 'V.Heavy' },
  'Extremely Heavy': { color: '#FF3B5C', glow: 'rgba(255,59,92,0.5)',   label: 'Extreme' },
};

function getConfig(category) {
  return CATEGORY_CONFIG[category] || CATEGORY_CONFIG['No Rain'];
}

export default function ForecastChart({ forecast = [] }) {
  const [hovered, setHovered] = useState(null);
  if (!forecast.length) return null;

  const maxMm = Math.max(...forecast.map(f => f.predicted_rainfall_mm), 10);

  return (
    <div className="w-full space-y-3">
      <div className="flex items-center gap-2">
        <Droplets size={13} style={{ color: '#00C9A7' }} />
        <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">
          7-Day Rainfall Forecast
        </span>
      </div>

      {/* Bar chart */}
      <div className="flex items-end justify-between gap-1.5 h-32">
        {forecast.map((day, i) => {
          const cfg = getConfig(day.rainfall_category);
          const pct = maxMm > 0 ? (day.predicted_rainfall_mm / maxMm) * 100 : 0;
          const isHovered = hovered === i;
          const dt = new Date(day.date);

          return (
            <motion.div
              key={i}
              className="flex-1 flex flex-col items-center gap-1 cursor-pointer"
              onHoverStart={() => setHovered(i)}
              onHoverEnd={() => setHovered(null)}>

              {/* Tooltip */}
              {isHovered && (
                <motion.div
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="absolute mb-2 z-20 glass rounded-lg px-2.5 py-2 text-center shadow-lg pointer-events-none"
                  style={{
                    border: `1px solid ${cfg.color}50`,
                    bottom: '100%',
                    minWidth: '80px',
                  }}>
                  <p className="text-xs font-mono font-bold" style={{ color: cfg.color }}>
                    {day.predicted_rainfall_mm.toFixed(1)} mm
                  </p>
                  <p className="text-[9px] font-mono text-slate-500">{cfg.label}</p>
                  <p className="text-[9px] font-mono text-slate-500">
                    {(day.probability_of_rain * 100).toFixed(0)}% rain
                  </p>
                </motion.div>
              )}

              {/* Bar */}
              <div className="relative w-full flex-1 flex items-end">
                <motion.div
                  className="w-full rounded-t-md relative overflow-hidden"
                  initial={{ height: 0 }}
                  animate={{ height: `${Math.max(4, pct)}%` }}
                  transition={{ delay: i * 0.07, duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
                  style={{
                    background: `linear-gradient(to top, ${cfg.color}, ${cfg.color}80)`,
                    boxShadow: isHovered ? `0 0 14px ${cfg.glow}` : 'none',
                    minHeight: '4px',
                  }}>
                  {/* Shimmer */}
                  <div className="absolute inset-0 bg-gradient-to-t from-transparent to-white opacity-10 rounded-t-md" />
                </motion.div>
              </div>

              {/* Date label */}
              <span className="text-[9px] font-mono" style={{ color: isHovered ? cfg.color : '#475569' }}>
                {format(dt, 'EEE')}
              </span>
              <span className="text-[8px] font-mono text-slate-600">
                {format(dt, 'dd')}
              </span>
            </motion.div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-x-3 gap-y-1">
        {Object.entries(CATEGORY_CONFIG).filter(([k]) => k !== 'No Rain').map(([name, cfg]) => (
          <div key={name} className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-sm" style={{ background: cfg.color }} />
            <span className="text-[9px] font-mono text-slate-500">{cfg.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
