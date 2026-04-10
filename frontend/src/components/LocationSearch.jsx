import { useState, useCallback } from 'react';
import { Search, Loader2, X } from 'lucide-react';

export default function LocationSearch({ onSelect, mapboxToken }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const search = useCallback(async (q) => {
    if (!q.trim() || q.length < 3) { setResults([]); return; }
    setLoading(true);
    try {
      const url = `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(q)}.json?access_token=${mapboxToken}&types=place,region,country&limit=5`;
      const r = await fetch(url);
      const d = await r.json();
      setResults(d.features || []);
    } catch { setResults([]); } finally { setLoading(false); }
  }, [mapboxToken]);

  const handleInput = e => {
    const v = e.target.value;
    setQuery(v);
    clearTimeout(window._searchTimer);
    window._searchTimer = setTimeout(() => search(v), 400);
  };

  const handleSelect = f => {
    const [lon, lat] = f.center;
    onSelect({ latitude: lat, longitude: lon, locationName: f.place_name });
    setQuery(f.place_name);
    setResults([]);
  };

  return (
    <div className="relative">
      <div className="flex items-center gap-2 px-3 py-2.5 rounded-xl glass"
        style={{ border: '1px solid rgba(0,201,167,0.22)' }}>
        {loading ? <Loader2 size={13} className="text-[#00C9A7] animate-spin" /> : <Search size={13} className="text-[#00C9A7]" />}
        <input type="text" value={query} onChange={handleInput}
          placeholder="Search city or region..."
          className="flex-1 bg-transparent text-sm text-slate-200 placeholder-slate-600 outline-none font-mono" />
        {query && (
          <button onClick={() => { setQuery(''); setResults([]); }}>
            <X size={11} className="text-slate-500 hover:text-slate-300" />
          </button>
        )}
      </div>
      {results.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-1 z-50 glass rounded-xl overflow-hidden"
          style={{ border: '1px solid rgba(15,42,61,0.9)' }}>
          {results.map(f => (
            <button key={f.id} onClick={() => handleSelect(f)}
              className="w-full text-left px-3 py-2.5 text-xs font-mono text-slate-300 hover:text-[#00C9A7] hover:bg-[rgba(0,201,167,0.05)] transition-colors border-b border-[rgba(15,42,61,0.6)] last:border-0">
              <span className="text-[#00C9A7] mr-1.5">→</span>{f.place_name}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
