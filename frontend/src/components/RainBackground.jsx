import { useMemo } from 'react';

export default function RainBackground({ intensity = 'light' }) {
  const drops = useMemo(() => {
    const count = intensity === 'heavy' ? 80 : intensity === 'moderate' ? 50 : 25;
    return Array.from({ length: count }, (_, i) => ({
      id: i,
      left: Math.random() * 100,
      height: 8 + Math.random() * 18,
      duration: 0.8 + Math.random() * 1.2,
      delay: Math.random() * 2,
      opacity: 0.2 + Math.random() * 0.3,
    }));
  }, [intensity]);

  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
      {drops.map(d => (
        <div
          key={d.id}
          className="rain-drop absolute"
          style={{
            left: `${d.left}%`,
            height: `${d.height}px`,
            animationDuration: `${d.duration}s`,
            animationDelay: `${d.delay}s`,
            opacity: d.opacity,
          }}
        />
      ))}
    </div>
  );
}
