"""
Ingest historical rainfall + meteorological data from Open-Meteo Archive API.

Downloads data for configurable locations and date ranges,
stores in PostgreSQL for training and historical queries.

Usage:
    python -m rainfall_service.training.ingest_data \
        --start-year 2000 --end-year 2023 \
        --locations-file locations.json
        
locations.json format:
    [
      {"name": "Pune", "lat": 18.52, "lon": 73.85},
      {"name": "Mumbai", "lat": 19.08, "lon": 72.88}
    ]
"""
from __future__ import annotations

import argparse, asyncio, json, logging, sys, os
from datetime import date, timedelta
from pathlib import Path

import aiohttp

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# Default Indian cities for monsoon research
DEFAULT_LOCATIONS = [
    {"name": "Pune",        "lat": 18.52, "lon": 73.85},
    {"name": "Mumbai",      "lat": 19.08, "lon": 72.88},
    {"name": "Delhi",       "lat": 28.61, "lon": 77.20},
    {"name": "Chennai",     "lat": 13.08, "lon": 80.27},
    {"name": "Kolkata",     "lat": 22.57, "lon": 88.36},
    {"name": "Bangalore",   "lat": 12.97, "lon": 77.59},
    {"name": "Hyderabad",   "lat": 17.38, "lon": 78.49},
    {"name": "Jaipur",      "lat": 26.91, "lon": 75.79},
    {"name": "Guwahati",    "lat": 26.19, "lon": 91.74},
    {"name": "Shillong",    "lat": 25.57, "lon": 91.88},
    {"name": "Agartala",    "lat": 23.83, "lon": 91.28},
    {"name": "Bhubaneswar", "lat": 20.30, "lon": 85.82},
    {"name": "Kochi",       "lat": 9.93,  "lon": 76.27},
    {"name": "Nagpur",      "lat": 21.15, "lon": 79.09},
    {"name": "Ahmedabad",   "lat": 23.03, "lon": 72.59},
    {"name": "Bhopal",      "lat": 23.26, "lon": 77.41},
    {"name": "Lucknow",     "lat": 26.85, "lon": 80.95},
    {"name": "Patna",       "lat": 25.59, "lon": 85.14},
]

DAILY_VARS = [
    "precipitation_sum", "temperature_2m_max", "temperature_2m_min",
    "relative_humidity_2m_mean", "wind_speed_10m_max", "wind_direction_10m_dominant",
    "cloud_cover_mean", "shortwave_radiation_sum",
    "et0_fao_evapotranspiration", "precipitation_hours",
]


async def fetch_location_year(
    session: aiohttp.ClientSession,
    name: str, lat: float, lon: float,
    year: int,
) -> list[dict]:
    start = date(year, 1, 1)
    end = min(date(year, 12, 31), date.today() - timedelta(days=1))
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARS), "timezone": "auto",
    }
    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
        resp.raise_for_status()
        data = await resp.json()

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    records = []
    for i, d in enumerate(dates):
        rain = (daily.get("precipitation_sum") or [])[i]
        if rain is None: rain = 0.0
        from rainfall_service.services.feature_engineering import rainfall_category
        records.append({
            "location_name": name,
            "latitude": lat,
            "longitude": lon,
            "record_date": date.fromisoformat(d),
            "rainfall_mm": float(rain),
            "temperature_max_c": _get(daily, "temperature_2m_max", i),
            "temperature_min_c": _get(daily, "temperature_2m_min", i),
            "humidity_percent": _get(daily, "relative_humidity_2m_mean", i),
            "wind_speed_kmh": _get(daily, "wind_speed_10m_max", i),
            "pressure_hpa": None,
            "cloud_cover_pct": _get(daily, "cloud_cover_mean", i),
            "rainfall_category": rainfall_category(float(rain)),
            "source": "Open-Meteo",
        })
    return records


def _get(d: dict, key: str, idx: int):
    arr = d.get(key) or []
    return arr[idx] if idx < len(arr) else None


async def ingest(start_year: int, end_year: int, locations: list[dict]):
    from rainfall_service.config import settings
    from rainfall_service.utils.database import Base, init_db
    from rainfall_service.models.db_models import RainfallRecord
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    total = 0
    async with aiohttp.ClientSession() as http:
        for loc in locations:
            for year in range(start_year, end_year + 1):
                logger.info("Fetching %s — %d", loc["name"], year)
                try:
                    rows = await fetch_location_year(http, loc["name"], loc["lat"], loc["lon"], year)
                except Exception as exc:
                    logger.error("Failed %s %d: %s", loc["name"], year, exc)
                    await asyncio.sleep(30)  # wait longer after a failure
                    continue

                if rows:
                    async with session_factory() as db:
                        stmt = (
                            pg_insert(RainfallRecord.__table__)
                            .values(rows)
                            .on_conflict_do_nothing()
                        )
                        result = await db.execute(stmt)
                        await db.commit()
                        total += result.rowcount or 0
                        logger.info("  → Inserted %d rows", result.rowcount or 0)

                await asyncio.sleep(6)   # between years

            logger.info("Pausing 60s after %s...", loc["name"])
            await asyncio.sleep(60)  # between cities

    logger.info("✅ Ingestion complete. Total rows: %d", total)
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--locations-file", type=str, default=None)
    args = parser.parse_args()

    locations = DEFAULT_LOCATIONS
    if args.locations_file and os.path.exists(args.locations_file):
        with open(args.locations_file) as f:
            locations = json.load(f)

    asyncio.run(ingest(args.start_year, args.end_year, locations))
