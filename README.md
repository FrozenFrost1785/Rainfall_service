# HydroAI — Rainfall Prediction Module
### AI-Powered Global Disaster Forecasting & Resource Allocation System
**BTech Final Year Project — Department of Computer Engineering, PCCOE Pune**

---

## System Architecture

```
              Open-Meteo API (Free)           NOAA (optional)
                      │                            │
              ┌───────▼────────────────────────────▼─────┐
              │         Weather Data Fetcher              │
              │  Historical (archive) + Forecast (7-day) │
              └──────────────────────┬───────────────────┘
                                     │
              ┌──────────────────────▼───────────────────┐
              │          Feature Engineering              │
              │  45 features: Raw(15) + Lag(4) +          │
              │  Rolling stats(9) + Calendar(5) +         │
              │  Derived Indices(5) + Sequence(20-dim)    │
              └──────────┬──────────────┬────────────────┘
                         │              │
          ┌──────────────▼──┐  ┌────────▼──────────────────┐
          │ BiLSTM-Attention │  │ XGBoost + LightGBM        │
          │  3 layers, 128h  │  │  800 trees, early stop    │
          │  + MultiHead     │  │  + feature importance     │
          │  Attention Pool  │  └────────────┬──────────────┘
          └──────────┬──────┘               │
                     └──────────┬───────────┘
                                │
              ┌─────────────────▼────────────────────────┐
              │   Ridge Regression Meta-Learner (Stack)   │
              │   Weights: LSTM×0.45 + XGB×0.30 +        │
              │            LGBM×0.25 (learned)            │
              └─────────────────┬────────────────────────┘
                                │
              ┌─────────────────▼────────────────────────┐
              │         Final Prediction                   │
              │   • Rainfall mm/day (regression)          │
              │   • Category (6-class classification)     │
              │   • Probability of rain                   │
              │   • Risk level: Low/Moderate/High/Extreme │
              └────────────────────────────────────────────┘
```

## Rainfall Classification (IMD Standard)

| Category | Range (mm/day) | Color |
|----------|---------------|-------|
| No Rain | 0 – 2.4 | Gray |
| Light | 2.4 – 15.6 | Teal |
| Moderate | 15.6 – 64.5 | Emerald |
| Heavy | 64.5 – 115.6 | Yellow |
| Very Heavy | 115.6 – 204.4 | Orange |
| Extremely Heavy | > 204.4 | Red |

---

## Feature Engineering (45 Features)

### Group 1: Raw Meteorological (15)
- Temperature max/min/mean
- Relative humidity
- Dew point
- Wind speed + direction (sin/cos encoded)
- Pressure anomaly (normalized)
- Cloud cover %
- Solar radiation (shortwave sum)
- CAPE (Convective Available Potential Energy)
- Precipitable water vapour
- Evapotranspiration (FAO-56)
- Surface pressure

### Group 2: Lag Features (4)
- Rainfall 1-day lag, 3-day lag, 7-day lag, 14-day lag

### Group 3: Rolling Statistics (9)
- 7-day, 14-day, 30-day rolling mean, std, and max rainfall

### Group 4: Temporal / Calendar (5)
- Season one-hot encoding (Winter/Pre-Monsoon/Monsoon/Post-Monsoon)
- Day of year (normalized 0–1)

### Group 5: Derived Indices (5)
- Consecutive rainy days
- Consecutive dry days
- Vapour Pressure Deficit (VPD)
- Wind moisture flux (speed × humidity)
- 5-day cumulative rainfall (monsoon onset proxy)

### Sequence (LSTM): 20-feature vector per time step
- Raw meteorological (15) + 5 rolling extras embedded into 30-day window

---

## Model Performance Targets

| Model | MAE (mm) | RMSE (mm) | R² | Accuracy | F1 |
|-------|----------|-----------|-----|----------|-----|
| Linear Regression | 18.4 | 34.2 | 0.58 | 0.65 | 0.62 |
| Random Forest | 9.2 | 18.7 | 0.78 | 0.80 | 0.79 |
| XGBoost | 7.1 | 15.3 | 0.84 | 0.84 | 0.83 |
| LightGBM | 6.8 | 14.9 | 0.85 | 0.85 | 0.84 |
| LSTM-Attention | 6.2 | 13.8 | 0.87 | 0.87 | 0.86 |
| **Ensemble (Ours)** | **5.4** | **11.9** | **0.91** | **0.91** | **0.90** |

---

## Folder Structure

```
rainfall_module/
├── docker-compose.yml
├── README.md
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── rainfall_service/
│       ├── main.py                        # FastAPI app
│       ├── config.py                      # Settings
│       │
│       ├── api/routes.py                  # /predict /historical /model-metrics
│       │
│       ├── models/
│       │   ├── lstm_model.py              # BiLSTM + Multi-Head Attention
│       │   ├── db_models.py               # SQLAlchemy ORM
│       │   ├── schemas.py                 # Pydantic schemas
│       │   └── saved/                     # Trained weights
│       │
│       ├── services/
│       │   ├── weather_fetcher.py         # Open-Meteo API (historical + forecast)
│       │   ├── feature_engineering.py     # 45-feature extractor
│       │   ├── model_loader.py            # Singleton model loader
│       │   └── predictor.py              # Ensemble inference
│       │
│       ├── training/
│       │   ├── ingest_data.py             # Open-Meteo historical → PostgreSQL
│       │   ├── train.py                   # Full pipeline (LSTM+XGB+LGBM+Stack)
│       │   └── evaluate.py               # Metrics + plots
│       │
│       ├── utils/
│       │   ├── database.py, cache.py, logger.py
│       │
│       └── websocket/manager.py           # Real-time alerts
│
└── frontend/
    └── src/
        ├── components/
        │   ├── RainfallMap.jsx            # Mapbox GL + heatmap
        │   ├── PredictionPanel.jsx        # Full forecast display
        │   ├── RainfallGauge.jsx          # SVG arc gauge
        │   ├── ForecastChart.jsx          # Animated 7-day bar chart
        │   ├── MetricsComparison.jsx      # Recharts comparison
        │   ├── LocationSearch.jsx         # Geocoding search
        │   ├── AlertToast.jsx             # Rain warnings
        │   └── RainBackground.jsx         # Animated rain drops
        ├── pages/Dashboard.jsx
        └── services/api.js
```

---

## Setup Instructions

### 1. Environment Setup

```bash
# Backend
cp backend/.env.example backend/.env

# Frontend
echo "VITE_MAPBOX_TOKEN=pk.your_real_token_here" > frontend/.env
echo "VITE_API_URL=http://localhost:8001/api/v1" >> frontend/.env
echo "VITE_WS_URL=ws://localhost:8001" >> frontend/.env
```

### 2. Docker Compose

```bash
docker-compose up --build
# Frontend: http://localhost:3001
# API Docs: http://localhost:8001/docs
```

### 3. Training

```bash
cd backend
pip install -r requirements.txt

# Step 1: Ingest historical data (2000-2023) for Indian cities
python -m rainfall_service.training.ingest_data \
    --start-year 2000 --end-year 2023

# Step 2: Train all models (runs in ~30 min with synthetic data)
python -m rainfall_service.training.train \
    --model-dir ./rainfall_service/models/saved \
    --epochs 50

# Step 3: Evaluate + generate plots
python -m rainfall_service.training.evaluate \
    --model-dir ./rainfall_service/models/saved \
    --output-dir ./reports
```

### 4. Local Dev

```bash
# Backend
uvicorn rainfall_service.main:app --reload --port 8001

# Frontend
cd frontend && npm install && npm run dev
```

---

## API Reference

### POST /api/v1/predict

```json
{
  "latitude": 18.5204,
  "longitude": 73.8567,
  "location_name": "Pune, Maharashtra",
  "forecast_days": 7
}
```

**Response:**
```json
{
  "request_id": "uuid",
  "location": "Pune, Maharashtra",
  "forecast": [
    {
      "date": "2026-02-21",
      "predicted_rainfall_mm": 42.5,
      "rainfall_category": "Moderate",
      "probability_of_rain": 0.83,
      "confidence": 0.78,
      "temperature_max_c": 31.2,
      "humidity_percent": 82.0,
      "wind_speed_kmh": 18.5,
      "pressure_hpa": 1008.3
    }
    // ... 6 more days
  ],
  "overall_risk": "Moderate",
  "season": "Post-Monsoon",
  "model_version": "1.0.0",
  "processing_time_ms": 892.4
}
```

---

## Data Sources

| Source | Data | API |
|--------|------|-----|
| Open-Meteo Archive | Historical daily weather (2000–2023) | `archive-api.open-meteo.com` |
| Open-Meteo Forecast | 7–16 day forecast | `api.open-meteo.com` |
| NOAA GHCN (optional) | Ground truth rainfall validation | `ncdc.noaa.gov/cdo-web/api` |
| PostgreSQL | Local historical store | Internal |

**Open-Meteo is completely free, no API key required.**

---

## UI Design

- **Theme**: Deep ocean / midnight blue (#050D18) — contrasts rain with dark water depth
- **Typography**: Sora (display) + Fira Code (monospace)
- **Accents**: Teal (#00C9A7) / Emerald (#00B37E) for safe/normal; Orange/Red for alerts
- **Animations**: CSS rain drops, SVG arc gauge fill, Framer Motion forecast bars
- **Map**: Mapbox GL heatmap — teal → yellow → red intensity gradient
- **Alerts**: Spring-animation toast cards with risk color coding
#   R a i n f a l l _ s e r v i c e  
 #   R a i n f a l l _ s e r v i c e  
 #   R a i n f a l l _ s e r v i c e  
 #   R a i n f a l l _ s e r v i c e  
 #   R a i n f a l l _ s e r v i c e  
 