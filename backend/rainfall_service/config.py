"""Application configuration using pydantic-settings."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    APP_ENV: str = "development"
    SECRET_KEY: str = "change-me"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/rainfall_db"
    SYNC_DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/rainfall_db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_TTL_PREDICTION: int = 300     # 5 min
    REDIS_TTL_WEATHER: int = 900        # 15 min

    # Open-Meteo (free, no key required)
    OPENMETEO_URL: str = "https://api.open-meteo.com/v1/forecast"
    OPENMETEO_HISTORICAL_URL: str = "https://archive-api.open-meteo.com/v1/archive"

    # NOAA
    NOAA_BASE_URL: str = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
    NOAA_TOKEN: str = ""    # optional

    # Model paths
    MODEL_DIR: str = "rainfall_service/models/saved/"
    ENSEMBLE_MODEL: str = "ensemble_model.pkl"
    LSTM_MODEL: str = "lstm_model.pt"
    XGBOOST_MODEL: str = "xgb_model.json"
    LGBM_MODEL: str = "lgbm_model.pkl"
    SCALER_FILE: str = "feature_scaler.pkl"
    LABEL_ENCODER: str = "label_encoder.pkl"

    # Thresholds (mm/day)
    HEAVY_RAIN_THRESHOLD: float = 64.5     # IMD classification
    VERY_HEAVY_THRESHOLD: float = 115.6
    EXTREMELY_HEAVY_THRESHOLD: float = 204.4

    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
