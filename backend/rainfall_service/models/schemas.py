"""Pydantic v2 schemas for Rainfall Prediction API."""
from __future__ import annotations
from datetime import datetime, date
from typing import Optional, List
from pydantic import BaseModel, Field


# ── Request ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, example=18.5204)
    longitude: float = Field(..., ge=-180, le=180, example=73.8567)
    location_name: Optional[str] = Field(None, example="Pune, Maharashtra")
    forecast_days: int = Field(7, ge=1, le=16, description="Number of days to forecast")


# ── Response ──────────────────────────────────────────────────────────────────

class DailyForecast(BaseModel):
    date: date
    predicted_rainfall_mm: float
    rainfall_category: str          # No Rain / Light / Moderate / Heavy / Very Heavy / Extremely Heavy
    probability_of_rain: float
    confidence: float
    temperature_max_c: Optional[float]
    temperature_min_c: Optional[float]
    humidity_percent: Optional[float]
    wind_speed_kmh: Optional[float]
    pressure_hpa: Optional[float]


class PredictionResponse(BaseModel):
    request_id: str
    location: str
    latitude: float
    longitude: float
    forecast: List[DailyForecast]
    overall_risk: str               # Low / Moderate / High / Extreme
    season: str
    model_version: str
    processing_time_ms: float
    timestamp: datetime


class HistoricalRainfall(BaseModel):
    date: date
    rainfall_mm: float
    category: str
    location: str


class HistoricalResponse(BaseModel):
    location: str
    latitude: float
    longitude: float
    days: int
    records: List[HistoricalRainfall]
    avg_rainfall_mm: float
    max_rainfall_mm: float
    rainy_days: int
    total_rainfall_mm: float


class ModelMetric(BaseModel):
    model_name: str
    mae: float
    rmse: float
    r2_score: float
    accuracy_class: float
    f1_score: float
    evaluation_date: str


class ModelMetricsResponse(BaseModel):
    models: List[ModelMetric]
    best_model: str
    confusion_matrix: List[List[int]]
    class_names: List[str]
    feature_importance: dict


class AlertPayload(BaseModel):
    alert_type: str
    location: str
    latitude: float
    longitude: float
    risk_level: str
    predicted_rainfall_mm: float
    category: str
    message: str
    timestamp: datetime
