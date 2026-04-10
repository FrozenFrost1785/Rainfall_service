import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LineChart, Line, Legend } from 'recharts';

const BEST_COLOR = '#00C9A7';
const OTHER_COLOR = '#0F2A3D';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass rounded-lg p-3 text-xs font-mono border border-[#0F2A3D]">
      <p className="text-[#00C9A7] font-bold mb-1.5">{label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color }}>{p.name}: {p.value}</p>
      ))}
    </div>
  );
};

export default function MetricsComparison({ models = [] }) {
  const f1Data = models.map(m => ({
    name: m.model_name.replace('Ensemble (Ours)', 'Ensemble★').replace('LSTM-Attention', 'LSTM-Attn'),
    'F1 Score': +(m.f1_score * 100).toFixed(1),
    'R² Score': +(m.r2_score * 100).toFixed(1),
    'Accuracy': +(m.accuracy_class * 100).toFixed(1),
    fullName: m.model_name,
  }));

  const errorData = models.map(m => ({
    name: m.model_name.replace('Ensemble (Ours)', 'Ensemble★').replace('LSTM-Attention', 'LSTM-Attn'),
    MAE: +m.mae.toFixed(2),
    RMSE: +m.rmse.toFixed(2),
    fullName: m.model_name,
  }));

  return (
    <div className="space-y-6">
      {/* F1/R²/Accuracy chart */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-slate-200 font-display mb-4">
          Classification & Regression Performance
        </h3>
        <div className="h-52">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={f1Data} barGap={2}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(15,42,61,0.8)" vertical={false} />
              <XAxis dataKey="name" tick={{ fill: '#475569', fontSize: 9, fontFamily: 'Fira Code' }} axisLine={{ stroke: '#0F2A3D' }} tickLine={false} />
              <YAxis domain={[50, 100]} tick={{ fill: '#475569', fontSize: 9, fontFamily: 'Fira Code' }} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 10, fontFamily: 'Fira Code', color: '#64748B' }} />
              <Bar dataKey="F1 Score" radius={[3, 3, 0, 0]} maxBarSize={22}>
                {f1Data.map((e, i) => (
                  <Cell key={i} fill={e.fullName === 'Ensemble (Ours)' ? BEST_COLOR : '#0F2A3D'} stroke={e.fullName === 'Ensemble (Ours)' ? '#00C9A7' : '#1E3A52'} strokeWidth={1} />
                ))}
              </Bar>
              <Bar dataKey="R² Score" radius={[3, 3, 0, 0]} maxBarSize={22} fill="#00E5FF" opacity={0.7} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* MAE/RMSE line chart */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-slate-200 font-display mb-4">
          Regression Error (Lower is Better)
        </h3>
        <div className="h-44">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={errorData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(15,42,61,0.8)" vertical={false} />
              <XAxis dataKey="name" tick={{ fill: '#475569', fontSize: 9, fontFamily: 'Fira Code' }} axisLine={{ stroke: '#0F2A3D' }} tickLine={false} />
              <YAxis tick={{ fill: '#475569', fontSize: 9, fontFamily: 'Fira Code' }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 10, fontFamily: 'Fira Code', color: '#64748B' }} />
              <Line type="monotone" dataKey="MAE" stroke="#00C9A7" strokeWidth={2} dot={{ fill: '#00C9A7', r: 3 }} />
              <Line type="monotone" dataKey="RMSE" stroke="#FF6B35" strokeWidth={2} dot={{ fill: '#FF6B35', r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Table */}
      <div className="glass rounded-xl overflow-hidden">
        <table className="w-full text-xs font-mono">
          <thead>
            <tr style={{ borderBottom: '1px solid #0F2A3D', background: 'rgba(15,42,61,0.5)' }}>
              {['Model', 'MAE', 'RMSE', 'R²', 'Accuracy', 'F1'].map(h => (
                <th key={h} className="text-left px-3 py-2.5 text-[9px] tracking-widest text-slate-500 uppercase">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((m, i) => {
              const best = m.model_name === 'Ensemble (Ours)';
              return (
                <tr key={i} style={{
                  borderBottom: '1px solid rgba(15,42,61,0.5)',
                  background: best ? 'rgba(0,201,167,0.04)' : 'transparent',
                }}>
                  <td className="px-3 py-2.5 font-semibold" style={{ color: best ? '#00C9A7' : '#64748B' }}>
                    {best && '★ '}{m.model_name}
                  </td>
                  <td className="px-3 py-2.5" style={{ color: best ? '#B0C4D8' : '#64748B' }}>{m.mae.toFixed(2)}</td>
                  <td className="px-3 py-2.5" style={{ color: best ? '#B0C4D8' : '#64748B' }}>{m.rmse.toFixed(2)}</td>
                  <td className="px-3 py-2.5" style={{ color: best ? '#00C9A7' : '#64748B' }}>{(m.r2_score * 100).toFixed(1)}%</td>
                  <td className="px-3 py-2.5" style={{ color: best ? '#B0C4D8' : '#64748B' }}>{(m.accuracy_class * 100).toFixed(1)}%</td>
                  <td className="px-3 py-2.5" style={{ color: best ? '#B0C4D8' : '#64748B' }}>{(m.f1_score * 100).toFixed(1)}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
