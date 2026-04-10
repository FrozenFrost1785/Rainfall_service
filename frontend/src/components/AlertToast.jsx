import { useState, useEffect } from 'react';
import { CloudRain, X, MapPin } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const RISK_STYLES = {
  Extreme:  { border: '#FF3B5C', bg: 'rgba(255,59,92,0.08)',  color: '#FF3B5C' },
  High:     { border: '#FF6B35', bg: 'rgba(255,107,53,0.08)', color: '#FF6B35' },
  Moderate: { border: '#FFD166', bg: 'rgba(255,209,102,0.08)',color: '#FFD166' },
  Low:      { border: '#00B37E', bg: 'rgba(0,179,126,0.08)',  color: '#00B37E' },
};

export default function AlertToast({ alerts }) {
  const [visible, setVisible] = useState([]);

  useEffect(() => {
    if (!alerts.length) return;
    const latest = alerts[alerts.length - 1];
    const id = Date.now();
    setVisible(prev => [...prev.slice(-2), { ...latest, id }]);
    const t = setTimeout(() => setVisible(prev => prev.filter(a => a.id !== id)), 9000);
    return () => clearTimeout(t);
  }, [alerts]);

  return (
    <div className="fixed top-6 right-6 z-50 space-y-2 pointer-events-none">
      <AnimatePresence>
        {visible.map(alert => {
          const s = RISK_STYLES[alert.risk_level] || RISK_STYLES.Low;
          return (
            <motion.div
              key={alert.id}
              initial={{ x: 360, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: 360, opacity: 0 }}
              transition={{ type: 'spring', stiffness: 280, damping: 28 }}
              className="pointer-events-auto w-76 rounded-xl p-4"
              style={{ background: s.bg, border: `1px solid ${s.border}`, boxShadow: `0 0 25px ${s.border}25` }}>
              <div className="flex items-start gap-2.5">
                <div className="relative mt-0.5">
                  <CloudRain size={16} style={{ color: s.color }} />
                  {['Extreme','High'].includes(alert.risk_level) && (
                    <span className="absolute inset-0 rounded-full animate-ping" style={{ background: s.color, opacity: 0.25 }} />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-[10px] font-mono font-bold tracking-widest uppercase" style={{ color: s.color }}>
                      {alert.risk_level} Rain Alert
                    </span>
                    <button onClick={() => setVisible(p => p.filter(a => a.id !== alert.id))} className="text-slate-500 hover:text-slate-300">
                      <X size={12} />
                    </button>
                  </div>
                  <div className="flex items-center gap-1 text-[10px] text-slate-400 mb-1.5">
                    <MapPin size={9} /><span className="truncate">{alert.location}</span>
                  </div>
                  <p className="text-xs text-slate-200">
                    <span className="font-bold">{alert.predicted_rainfall_mm?.toFixed(1)}mm</span> predicted — {alert.category}
                  </p>
                </div>
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
