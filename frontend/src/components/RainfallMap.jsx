import { useRef, useEffect } from 'react';
import mapboxgl from 'mapbox-gl';

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN || 'pk.YOUR_TOKEN_HERE';

const RISK_COLORS = {
  Low:      '#00B37E',
  Moderate: '#FFD166',
  High:     '#FF6B35',
  Extreme:  '#FF3B5C',
};

export default function RainfallMap({ onLocationSelect, selectedLocation, historicalData = [], predictionResult }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const markerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new mapboxgl.Map({
      container: containerRef.current,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [78.96, 20.59],   // India center
      zoom: 4,
      projection: 'mercator',
    });

    map.on('style.load', () => {
      map.setFog({
        color: 'rgb(5,13,24)',
        'high-color': 'rgb(7,21,37)',
        'horizon-blend': 0.04,
        'space-color': '#050D18',
        'star-intensity': 0.5,
      });

      // Historical rainfall heatmap source
      map.addSource('rainfall-heat', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });

      map.addLayer({
        id: 'rainfall-heatmap',
        type: 'heatmap',
        source: 'rainfall-heat',
        paint: {
          'heatmap-weight': ['interpolate', ['linear'], ['get', 'rainfall'], 0, 0, 200, 1],
          'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 0, 1, 9, 3],
          'heatmap-color': [
            'interpolate', ['linear'], ['heatmap-density'],
            0,   'rgba(0,201,167,0)',
            0.2, 'rgba(0,201,167,0.4)',
            0.4, 'rgba(0,179,126,0.6)',
            0.6, 'rgba(255,209,102,0.7)',
            0.8, 'rgba(255,107,53,0.8)',
            1,   'rgba(255,59,92,1)',
          ],
          'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 0, 20, 9, 40],
          'heatmap-opacity': 0.75,
        },
      });
    });

    map.on('click', e => {
      const { lng, lat } = e.lngLat;
      onLocationSelect({ latitude: lat, longitude: lng });
    });

    mapRef.current = map;
    return () => { map.remove(); mapRef.current = null; };
  }, []);

  // Update heatmap data
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;
    const src = map.getSource('rainfall-heat');
    if (!src) return;
    const features = historicalData.map(d => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [d.longitude || 78.96, d.latitude || 20.59] },
      properties: { rainfall: d.rainfall_mm || 0 },
    }));
    src.setData({ type: 'FeatureCollection', features });
  }, [historicalData]);

  // Update marker
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !selectedLocation) return;
    if (markerRef.current) markerRef.current.remove();

    const risk = predictionResult?.overall_risk || 'Low';
    const color = RISK_COLORS[risk];

    const el = document.createElement('div');
    el.style.cssText = 'width:24px;height:24px;position:relative;';
    el.innerHTML = `
      <div style="position:absolute;inset:0;border-radius:50%;background:${color};opacity:0.2;animation:ripple 2s ease-out infinite;"></div>
      <div style="position:absolute;inset:0;border-radius:50%;background:${color};opacity:0.1;animation:ripple 2s ease-out 0.6s infinite;"></div>
      <div style="position:absolute;inset:5px;border-radius:50%;background:${color};box-shadow:0 0 14px ${color}80;"></div>
    `;

    markerRef.current = new mapboxgl.Marker(el)
      .setLngLat([selectedLocation.longitude, selectedLocation.latitude])
      .addTo(map);

    map.flyTo({ center: [selectedLocation.longitude, selectedLocation.latitude], zoom: 7, duration: 1400 });
  }, [selectedLocation, predictionResult]);

  return <div ref={containerRef} className="w-full h-full rounded-xl overflow-hidden" />;
}
