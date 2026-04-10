import axios from 'axios';

const BASE = import.meta.env.VITE_API_URL || '/api/v1';
const api = axios.create({ baseURL: BASE, timeout: 30000 });

export async function predict({ latitude, longitude, locationName, forecastDays = 7 }) {
  const { data } = await api.post('/predict', {
    latitude, longitude,
    location_name: locationName,
    forecast_days: forecastDays,
  });
  return data;
}

export async function getHistorical({ location, lat, lon, days = 30 }) {
  const { data } = await api.get(`/historical/${encodeURIComponent(location)}`, {
    params: { lat, lon, days },
  });
  return data;
}

export async function getModelMetrics() {
  const { data } = await api.get('/model-metrics');
  return data;
}

const WS_URL = (import.meta.env.VITE_WS_URL || 'ws://localhost:8001') + '/alerts';

export function createAlertSocket(onMessage, onError) {
  const ws = new WebSocket(WS_URL);
  ws.onmessage = e => { try { onMessage(JSON.parse(e.data)); } catch {} };
  ws.onerror = onError;
  ws.onclose = () => setTimeout(() => createAlertSocket(onMessage, onError), 3000);
  return ws;
}
