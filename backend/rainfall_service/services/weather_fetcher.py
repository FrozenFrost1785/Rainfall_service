"""
Open-Meteo Weather Data Fetcher.

Uses Open-Meteo free API (no key required) to fetch:
  1. Historical daily data (archive endpoint) for training/features
  2. Forecast data (forecast endpoint) for prediction

Variables fetched align exactly with our 15-feature vector.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta, datetime
from typing import Optional

import aiohttp
from matplotlib import dates
from matplotlib.pylab import rec

from rainfall_service.config import settings
from rainfall_service.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

# Open-Meteo daily variables (subset for our feature set)
DAILY_VARS = [
    "precipitation_sum",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "cloud_cover_mean",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "precipitation_hours",
]

# Additional hourly → daily aggregated
HOURLY_VARS_DAILY = [
    "cape",
    "dew_point_2m",
]


async def fetch_historical(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
) -> list[dict]:
    """
    Fetch historical daily weather data from Open-Meteo Archive API.
    Returns list of daily dicts with all meteorological variables.
    """
    cache_key = f"hist:{latitude:.3f}:{longitude:.3f}:{start_date}:{end_date}"
    cached = await cache_get(cache_key)
    if cached:
        return cached["records"]

    url = settings.OPENMETEO_HISTORICAL_URL
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ",".join(DAILY_VARS),
        "timezone": "auto",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                resp.raise_for_status()
                data = await resp.json()
    except Exception as exc:
        logger.error("Open-Meteo historical fetch failed: %s", exc)
        return []

    records = _parse_daily_response(data)
    await cache_set(cache_key, {"records": [_serialize_record(r) for r in records]}, ttl=3600)
    return records


async def fetch_forecast(
    latitude: float,
    longitude: float,
    days: int = 7,
) -> list[dict]:
    """
    Fetch weather forecast from Open-Meteo forecast API.
    Returns list of daily forecast dicts for the next `days` days.
    """
    cache_key = f"forecast:{latitude:.3f}:{longitude:.3f}:{days}"
    cached = await cache_get(cache_key)
    if cached:
        return cached["records"]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "forecast_days": days,
        "daily": ",".join(DAILY_VARS),
        "timezone": "auto",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(settings.OPENMETEO_URL, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                resp.raise_for_status()
                data = await resp.json()
    except Exception as exc:
        logger.error("Open-Meteo forecast fetch failed: %s", exc)
        return _dummy_forecast(days)

    records = _parse_daily_response(data)
    await cache_set(cache_key, {"records": [_serialize_record(r) for r in records]}, ttl=settings.REDIS_TTL_WEATHER)
    return records


async def fetch_history_for_features(
    latitude: float,
    longitude: float,
    lookback_days: int = 35,
) -> list[dict]:
    """Fetch recent history (lookback_days) to build LSTM feature sequence."""
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=lookback_days)
    return await fetch_historical(latitude, longitude, start, end)


# ── Private helpers ───────────────────────────────────────────────────────────

def _parse_daily_response(data: dict) -> list[dict]:
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    records = []
    for i, d in enumerate(dates):
        rec = {"date": d}
        for key in DAILY_VARS:
            vals = daily.get(key, [])
            rec[key] = vals[i] if i < len(vals) else None
        # Map API names to our internal names
        rec["rainfall_mm"] = rec.pop("precipitation_sum", 0.0) or 0.0
        rec["day_of_year"] = datetime.strptime(dates[i], "%Y-%m-%d").timetuple().tm_yday
        rec["month"] = datetime.strptime(dates[i], "%Y-%m-%d").month
        records.append(rec)
    return records


def _serialize_record(r: dict) -> dict:
    return {k: (str(v) if isinstance(v, date) else v) for k, v in r.items()}


def _dummy_forecast(days: int) -> list[dict]:
    """Fallback if API fails — includes ALL fields needed by feature_engineering."""
    today = date.today()
    result = []
    for i in range(days):
        d = today + timedelta(days=i)
        doy = d.timetuple().tm_yday
        result.append({
            "date": d.isoformat(),
            "rainfall_mm": 0.0,
            "temperature_2m_max": 30.0,
            "temperature_2m_min": 22.0,
            "relative_humidity_2m_mean": 70.0,
            "wind_speed_10m_max": 15.0,
            "wind_direction_10m_dominant": 180.0,
            "cloud_cover_mean": 40.0,
            "shortwave_radiation_sum": 18.0,
            "et0_fao_evapotranspiration": 3.5,
            "precipitation_hours": 0.0,
            "day_of_year": doy,
            "month": d.month,
        })
    return result