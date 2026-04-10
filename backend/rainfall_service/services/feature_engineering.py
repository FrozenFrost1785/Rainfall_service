"""
Feature Engineering for Rainfall Prediction.

Generates 45 features across 5 categories:
  1. Raw meteorological (15 features)
  2. Lag features — 1d, 3d, 7d, 14d (16 features)
  3. Rolling statistics — mean, std, max over windows (9 features)
  4. Calendar / seasonal features (5 features)
  5. Derived atmospheric indices (5 features)
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
from typing import Optional

# IMD Rainfall Classification (mm/day)
RAINFALL_CATEGORIES = [
    (0.0, 2.4,   "No Rain"),
    (2.4, 15.6,  "Light"),
    (15.6, 64.5, "Moderate"),
    (64.5, 115.6,"Heavy"),
    (115.6, 999.9,"Very Heavy"),
]
CATEGORY_LABELS = [c[2] for c in RAINFALL_CATEGORIES]
N_FEATURES_RAW = 10
SEQ_LEN = 30          # 30-day lookback window for LSTM


def rainfall_category(mm: float) -> str:
    for lo, hi, name in RAINFALL_CATEGORIES:
        if lo <= mm < hi:
            return name
    return "Extremely Heavy"


def category_to_index(cat: str) -> int:
    for i, (_, _, name) in enumerate(RAINFALL_CATEGORIES):
        if name == cat:
            return i
    return 0


def get_season(month: int) -> str:
    if month in [6, 7, 8, 9]:    return "Monsoon"
    if month in [3, 4, 5]:       return "Pre-Monsoon"
    if month in [10, 11]:        return "Post-Monsoon"
    return "Winter"


def season_index(month: int) -> int:
    return {"Winter": 0, "Pre-Monsoon": 1, "Monsoon": 2, "Post-Monsoon": 3}[get_season(month)]


def build_raw_feature_vector(day_data: dict) -> np.ndarray:
    """
    Build a 15-feature vector from a single day's meteorological observations.

    Features:
        0  temperature_max_c
        1  temperature_min_c
        2  temperature_mean_c
        3  humidity_percent
        4  dew_point_c
        5  wind_speed_kmh
        6  wind_direction_deg (sin-encoded)
        7  wind_direction_deg (cos-encoded)
        8  pressure_hpa (normalized)
        9  cloud_cover_pct
        10 solar_radiation_wm2
        11 cape_j_kg (Convective Available Potential Energy)
        12 precipitable_water_mm
        13 evapotranspiration_mm
        14 surface_pressure_hpa
    """
    def g(k, default=0.0):
            v = day_data.get(k)
            return float(v) if v is not None else default

    t_max = g("temperature_2m_max")
    t_min = g("temperature_2m_min")
    wind_dir = g("wind_direction_10m_dominant")

     # Seasonal features
    day_of_year = g("day_of_year", 180)
    month = g("month", 6)
    sin_doy = math.sin(2 * math.pi * day_of_year / 365)
    cos_doy = math.cos(2 * math.pi * day_of_year / 365)
    monsoon = 1.0 if 6 <= int(month) <= 9 else 0.0
    pre_monsoon = 1.0 if 3 <= int(month) <= 5 else 0.0
    post_monsoon = 1.0 if 10 <= int(month) <= 11 else 0.0

    return np.array([
        t_max,
        t_min,
        (t_max + t_min) / 2.0,
        g("relative_humidity_2m_mean"),
        g("wind_speed_10m_max"),
        math.sin(math.radians(wind_dir)),
        math.cos(math.radians(wind_dir)),
        g("cloud_cover_mean", 50.0),
        g("shortwave_radiation_sum", 15.0),
        g("et0_fao_evapotranspiration", 3.0),
        sin_doy,        # seasonal sine
        cos_doy,        # seasonal cosine
        monsoon,        # monsoon flag
        pre_monsoon,    # pre-monsoon flag
        post_monsoon,   # post-monsoon flag
    ], dtype=np.float32)


def build_sequence_features(
    history: list[dict],
    target_date: Optional[date] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build:
      - sequence: (SEQ_LEN, N_FEATURES_FULL) for LSTM
      - flat_feats: (N_FLAT,) for XGBoost / LightGBM

    history: list of dicts (recent to oldest, length >= SEQ_LEN)
    """
    # Clip/pad to SEQ_LEN
    h = history[-SEQ_LEN:] if len(history) >= SEQ_LEN else history
    if len(h) < SEQ_LEN:
        pad = [{}] * (SEQ_LEN - len(h))
        h = pad + h

    raw_matrix = np.stack([build_raw_feature_vector(d) for d in h])  # (30, 15)
    rainfall_hist = np.array([float(d.get("rainfall_mm", 0.0) or 0.0) for d in h])

    # ── Lag features ──────────────────────────────────────────────────────────
    lag_1  = rainfall_hist[-1]   if len(rainfall_hist) >= 1  else 0.0
    lag_3  = rainfall_hist[-3]   if len(rainfall_hist) >= 3  else 0.0
    lag_7  = rainfall_hist[-7]   if len(rainfall_hist) >= 7  else 0.0
    lag_14 = rainfall_hist[-14]  if len(rainfall_hist) >= 14 else 0.0

    # ── Rolling stats ─────────────────────────────────────────────────────────
    roll7  = rainfall_hist[-7:]
    roll14 = rainfall_hist[-14:]
    roll30 = rainfall_hist

    flat_feats = np.array([
        # Lag features (4)
        lag_1, lag_3, lag_7, lag_14,
        # Rolling mean (3)
        float(roll7.mean()), float(roll14.mean()), float(roll30.mean()),
        # Rolling std (3)
        float(roll7.std()), float(roll14.std()), float(roll30.std()),
        # Rolling max (3)
        float(roll7.max()), float(roll14.max()), float(roll30.max()),
        # Consecutive rainy days
        float(_consecutive_rainy_days(rainfall_hist)),
        # Consecutive dry days
        float(_consecutive_dry_days(rainfall_hist)),
        # Current day meteorological features (15)
        *raw_matrix[-1],
        # Season (1) — one-hot-encoded (4)
        *_month_to_season_onehot(
            target_date.month if target_date else _guess_month(h)
        ),
        # Day of year normalized
        float(target_date.timetuple().tm_yday) / 365.0 if target_date else 0.0,
        # Monsoon onset proxy: sum rainfall last 5 days
        float(rainfall_hist[-5:].sum()),
        # Vapor pressure deficit proxy
        float(_vpd(raw_matrix[-1])),
        # Wind moisture flux proxy
        float(raw_matrix[-1, 5] * raw_matrix[-1, 3] / 100.0),   # speed * humidity
    ], dtype=np.float32)

    # Append lag + rolling cols to sequence for LSTM
    n_extra = 5
    extras = np.zeros((SEQ_LEN, n_extra), dtype=np.float32)
    for i in range(SEQ_LEN):
        w = rainfall_hist[:i+1]
        extras[i, 0] = w[-1] if len(w) >= 1 else 0.0
        extras[i, 1] = w[-3] if len(w) >= 3 else 0.0
        extras[i, 2] = float(w.mean())
        extras[i, 3] = float(w[-7:].sum()) if len(w) >= 7 else float(w.sum())
        extras[i, 4] = float(_consecutive_rainy_days(w))

    sequence = np.concatenate([raw_matrix, extras], axis=1)  # (30, 20)
    return sequence.astype(np.float32), flat_feats.astype(np.float32)


# ── Private helpers ───────────────────────────────────────────────────────────

def _consecutive_rainy_days(arr: np.ndarray, threshold: float = 2.4) -> int:
    count = 0
    for v in reversed(arr):
        if v >= threshold:
            count += 1
        else:
            break
    return count


def _consecutive_dry_days(arr: np.ndarray, threshold: float = 2.4) -> int:
    count = 0
    for v in reversed(arr):
        if v < threshold:
            count += 1
        else:
            break
    return count


def _month_to_season_onehot(month: int) -> list[float]:
    seasons = ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]
    s = get_season(month)
    return [1.0 if s == x else 0.0 for x in seasons]


def _guess_month(history: list[dict]) -> int:
    import datetime
    return datetime.date.today().month


def _vpd(raw: np.ndarray) -> float:
    """Vapour Pressure Deficit (approximate)."""
    t_mean = raw[2]   # temperature_mean_c
    rh = raw[3]       # humidity_percent
    es = 6.112 * math.exp(17.67 * t_mean / (t_mean + 243.5))
    ea = es * rh / 100.0
    return max(0.0, es - ea)
