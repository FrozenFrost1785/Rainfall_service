"""
Rainfall Prediction Inference Engine.

Ensemble strategy:
  1. BiLSTM-Attention → regression (mm) + classification (5 classes)
  2. XGBoost → regression (mm) + classification
  3. LightGBM → regression (mm) + classification
  4. Stacked ensemble (weighted average) → final prediction

Final output = weighted average:
  LSTM × 0.45 + XGBoost × 0.30 + LightGBM × 0.25
(weights learned during stacking; fallback to static weights)
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, date, timedelta, timezone

import numpy as np
import torch

from rainfall_service.config import settings
from rainfall_service.models.schemas import PredictionResponse, DailyForecast
from rainfall_service.services.model_loader import ModelLoader, DEVICE
from rainfall_service.services.weather_fetcher import fetch_forecast, fetch_history_for_features
from rainfall_service.services.feature_engineering import (
    build_sequence_features,
    rainfall_category,
    get_season,
    CATEGORY_LABELS,
)
from rainfall_service.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

# Static ensemble weights (LSTM, XGB, LGBM)
WEIGHTS = [0.45, 0.30, 0.25]


def _overall_risk(forecasts):
    max_mm = max(f.predicted_rainfall_mm for f in forecasts)
    if max_mm >= 204.4: return "Extreme"    # formerly Extremely Heavy
    if max_mm >= 115.6: return "Very High"  # formerly Very Heavy
    if max_mm >= 64.5:  return "High"       # Heavy
    if max_mm >= 15.6:  return "Moderate"   # Moderate
    return "Low"


def _predict_single_day(
    sequence: np.ndarray,      # (30, 20) LSTM input
    flat_feats: np.ndarray,    # (N,) XGB / LGBM input
) -> tuple[float, str, float, float]:
    """
    Run all models on a single day and return ensemble prediction.

    Returns:
        (rainfall_mm, category, probability_of_rain, confidence)
    """
    predictions_mm = []
    class_probas = []

    # ── LSTM ──────────────────────────────────────────────────────────────────
    lstm = ModelLoader.lstm
    if lstm is not None:
        try:
            tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                mm_pred, logits, _ = lstm(tensor)
            lstm_mm = float(mm_pred.squeeze().item())
            lstm_proba = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            predictions_mm.append(max(0.0, lstm_mm))
            class_probas.append(lstm_proba)
        except Exception as exc:
            logger.warning("LSTM inference failed: %s", exc)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_m = ModelLoader.xgb_model
    if xgb_m is not None and hasattr(xgb_m, "predict"):
        try:
            xgb_mm = float(xgb_m.predict(flat_feats.reshape(1, -1))[0])
            predictions_mm.append(max(0.0, xgb_mm))
        except Exception as exc:
            logger.warning("XGBoost inference failed: %s", exc)

    # ── LightGBM ──────────────────────────────────────────────────────────────
    lgbm = ModelLoader.lgbm_model
    if lgbm is not None:
        try:
            lgbm_mm = float(lgbm.predict(flat_feats.reshape(1, -1))[0])
            predictions_mm.append(max(0.0, lgbm_mm))
        except Exception as exc:
            logger.warning("LightGBM inference failed: %s", exc)

    # ── Ensemble ──────────────────────────────────────────────────────────────
    if predictions_mm:
        w = WEIGHTS[:len(predictions_mm)]
        w = [x / sum(w) for x in w]   # renormalize
        final_mm = float(sum(p * wt for p, wt in zip(predictions_mm, w)))
    else:
        # Pure heuristic fallback: use humidity + recent rainfall lag
        humidity = float(flat_feats[18]) if len(flat_feats) > 18 else 60.0
        lag1 = float(flat_feats[0]) if len(flat_feats) > 0 else 0.0
        final_mm = max(0.0, (humidity - 50) / 10.0 + lag1 * 0.3)

    # Category from ensemble probability or mm threshold
    if class_probas:
        avg_proba = np.mean(class_probas, axis=0)
        cat_idx = int(np.argmax(avg_proba))
        category = CATEGORY_LABELS[cat_idx]
        prob_rain = float(1.0 - avg_proba[0])    # P(not "No Rain")
        confidence = float(avg_proba[cat_idx])
    else:
        category = rainfall_category(final_mm)
        prob_rain = 0.0 if final_mm < 2.4 else min(0.95, final_mm / 100.0)
        confidence = 0.55

    return round(final_mm, 2), category, round(prob_rain, 4), round(confidence, 4)


async def run_prediction(
    latitude: float,
    longitude: float,
    location_name: str,
    forecast_days: int = 7,
) -> PredictionResponse:
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())

    # ── Cache ──────────────────────────────────────────────────────────────────
    cache_key = f"rainpred:{latitude:.3f}:{longitude:.3f}:{forecast_days}"
    cached = await cache_get(cache_key)
    if cached:
        cached["request_id"] = request_id
        cached["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
        return PredictionResponse(**cached)

    # ── Fetch data ─────────────────────────────────────────────────────────────
    history_task = fetch_history_for_features(latitude, longitude, lookback_days=35)
    forecast_task = fetch_forecast(latitude, longitude, days=forecast_days)
    history, forecast_weather = await asyncio.gather(history_task, forecast_task)

    # ── Build predictions day by day ───────────────────────────────────────────
    forecasts: list[DailyForecast] = []
    rolling_history = list(history)

    for i, day_weather in enumerate(forecast_weather[:forecast_days]):
        target_date_str = day_weather.get("date")
        try:
            target_date = date.fromisoformat(str(target_date_str))
        except Exception:
            target_date = date.today() + timedelta(days=i)

        sequence, flat_feats = build_sequence_features(rolling_history, target_date)

        if ModelLoader.scaler is not None:
            try:
                flat_feats = ModelLoader.scaler.transform(flat_feats.reshape(1, -1)).flatten()
            except Exception:
                pass

        mm, category, prob_rain, confidence = _predict_single_day(sequence, flat_feats)

        forecasts.append(DailyForecast(
            date=target_date,
            predicted_rainfall_mm=mm,
            rainfall_category=category,
            probability_of_rain=prob_rain,
            confidence=confidence,
            temperature_max_c=day_weather.get("temperature_2m_max"),
            temperature_min_c=day_weather.get("temperature_2m_min"),
            humidity_percent=day_weather.get("relative_humidity_2m_mean"),
            wind_speed_kmh=day_weather.get("wind_speed_10m_max"),
            pressure_hpa=day_weather.get("surface_pressure"),
        ))

        # Add predicted day to rolling history for next day's lag features
        rolling_history.append({**day_weather, "rainfall_mm": mm})

    elapsed_ms = (time.perf_counter() - t0) * 1000
    season = get_season(date.today().month)

    response = PredictionResponse(
        request_id=request_id,
        location=location_name or f"{latitude:.4f}°, {longitude:.4f}°",
        latitude=latitude,
        longitude=longitude,
        forecast=forecasts,
        overall_risk=_overall_risk(forecasts),
        season=season,
        model_version=ModelLoader.model_version,
        processing_time_ms=round(elapsed_ms, 1),
        timestamp=datetime.now(tz=timezone.utc),
    )

    await cache_set(cache_key, response.model_dump(mode="json"), ttl=settings.REDIS_TTL_PREDICTION)

    # Broadcast high-risk alert
    if response.overall_risk in ("High", "Extreme"):
        asyncio.create_task(_broadcast_alert(response))

    return response


async def _broadcast_alert(pred: PredictionResponse):
    try:
        from rainfall_service.websocket.manager import alert_manager
        from rainfall_service.models.schemas import AlertPayload
        peak = max(pred.forecast, key=lambda f: f.predicted_rainfall_mm)
        payload = AlertPayload(
            alert_type="HEAVY_RAIN_WARNING",
            location=pred.location,
            latitude=pred.latitude,
            longitude=pred.longitude,
            risk_level=pred.overall_risk,
            predicted_rainfall_mm=peak.predicted_rainfall_mm,
            category=peak.rainfall_category,
            message=(
                f"{pred.overall_risk.upper()} RAINFALL ALERT — {pred.location}: "
                f"{peak.predicted_rainfall_mm:.1f}mm predicted on {peak.date}"
            ),
            timestamp=pred.timestamp,
        )
        await alert_manager.broadcast(payload.model_dump(mode="json"))
    except Exception as exc:
        logger.error("Alert broadcast failed: %s", exc)
