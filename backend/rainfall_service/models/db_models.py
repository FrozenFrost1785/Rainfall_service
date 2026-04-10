"""ORM models for rainfall records and prediction logs."""
import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Date, JSON, Text
from rainfall_service.utils.database import Base


class RainfallRecord(Base):
    """Historical daily rainfall observation."""
    __tablename__ = "rainfall_records"

    id = Column(Integer, primary_key=True, index=True)
    location_name = Column(String(256), index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    record_date = Column(Date, nullable=False, index=True)
    rainfall_mm = Column(Float, nullable=False)
    temperature_max_c = Column(Float)
    temperature_min_c = Column(Float)
    humidity_percent = Column(Float)
    wind_speed_kmh = Column(Float)
    pressure_hpa = Column(Float)
    cloud_cover_pct = Column(Float)
    dew_point_c = Column(Float)
    rainfall_category = Column(String(32))
    source = Column(String(64), default="Open-Meteo")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class PredictionLog(Base):
    """Audit log for every prediction request."""
    __tablename__ = "rainfall_prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(64), unique=True, index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    location_name = Column(String(256))
    forecast_days = Column(Integer)
    overall_risk = Column(String(32))
    model_version = Column(String(32))
    processing_time_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
