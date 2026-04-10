import { useEffect, useRef } from 'react';

const RISK_COLORS = {
  Low:      { stroke: '#00B37E', fill: '#00B37E20' },
  Moderate: { stroke: '#FFD166', fill: '#FFD16620' },
  High:     { stroke: '#FF6B35', fill: '#FF6B3520' },
  Extreme:  { stroke: '#FF3B5C', fill: '#FF3B5C20' },
};

const SIZE = 160;
const STROKE = 10;
const R = (SIZE - STROKE) / 2;
const CIRC = 2 * Math.PI * R;
const ARC = (CIRC * 270) / 360;

export default function RainfallGauge({ maxRainfallMm = 0, riskLevel = 'Low', season = '' }) {
  const arcRef = useRef(null);
  const colors = RISK_COLORS[riskLevel] || RISK_COLORS.Low;

  // Normalize 0-300mm to 0-1
  const pct = Math.min(1, maxRainfallMm / 300);
  const offset = ARC - ARC * pct;

  useEffect(() => {
    if (!arcRef.current) return;
    arcRef.current.style.strokeDashoffset = ARC;
    void arcRef.current.offsetWidth;
    arcRef.current.style.transition = 'stroke-dashoffset 1.6s cubic-bezier(0.4,0,0.2,1), stroke 0.5s';
    arcRef.current.style.strokeDashoffset = offset;
  }, [offset]);

  return (
    <div className="flex flex-col items-center gap-2">
      <svg width={SIZE} height={SIZE + 8} viewBox={`0 0 ${SIZE} ${SIZE + 8}`}>
        <defs>
          <filter id="rain-glow">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <linearGradient id="arc-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={colors.stroke} stopOpacity="0.6" />
            <stop offset="100%" stopColor={colors.stroke} />
          </linearGradient>
        </defs>

        {/* Track */}
        <circle cx={SIZE/2} cy={SIZE/2} r={R}
          fill="none" stroke="rgba(15,42,61,0.9)" strokeWidth={STROKE}
          strokeLinecap="round"
          strokeDasharray={`${ARC} ${CIRC}`} strokeDashoffset={0}
          transform={`rotate(135 ${SIZE/2} ${SIZE/2})`} />

        {/* Value arc */}
        <circle ref={arcRef} cx={SIZE/2} cy={SIZE/2} r={R}
          fill="none" stroke={colors.stroke} strokeWidth={STROKE}
          strokeLinecap="round"
          strokeDasharray={`${ARC} ${CIRC}`} strokeDashoffset={offset}
          transform={`rotate(135 ${SIZE/2} ${SIZE/2})`}
          filter="url(#rain-glow)" />

        {/* Center */}
        <text x={SIZE/2} y={SIZE/2 - 8}
          textAnchor="middle" dominantBaseline="middle"
          fill={colors.stroke} fontSize="28" fontWeight="700"
          fontFamily="'Fira Code', monospace" filter="url(#rain-glow)">
          {maxRainfallMm.toFixed(0)}
        </text>
        <text x={SIZE/2} y={SIZE/2 + 14}
          textAnchor="middle" fill="rgba(176,196,216,0.5)"
          fontSize="9" fontFamily="'Fira Code', monospace" letterSpacing="1.5">
          mm/day (peak)
        </text>
      </svg>

      {/* Risk badge */}
      <div className="flex flex-col items-center gap-1">
        <span className="px-3 py-0.5 rounded-full text-[10px] font-mono font-bold tracking-widest uppercase"
          style={{
            border: `1px solid ${colors.stroke}`,
            color: colors.stroke,
            background: colors.fill,
            boxShadow: `0 0 10px ${colors.stroke}30`,
          }}>
          {riskLevel} Risk
        </span>
        {season && (
          <span className="text-[9px] font-mono text-slate-500 tracking-widest uppercase">{season} Season</span>
        )}
      </div>
    </div>
  );
}
