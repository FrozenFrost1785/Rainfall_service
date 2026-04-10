"""Rainfall Prediction Service — FastAPI application."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from rainfall_service.api.routes import router as api_router
from rainfall_service.websocket.manager import ws_router
from rainfall_service.utils.database import init_db
from rainfall_service.utils.cache import init_redis
from rainfall_service.utils.logger import setup_logging
from rainfall_service.services.model_loader import ModelLoader

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🌧️  Starting Rainfall Prediction Service...")
    await init_db()
    await init_redis()
    await ModelLoader.load_all()
    logger.info("✅ All services initialized.")
    yield
    logger.info("🛑 Shutting down.")


app = FastAPI(
    title="Rainfall Prediction API",
    description="AI-Powered Rainfall Forecasting — BiLSTM-Attention + XGBoost + LightGBM Ensemble",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")
app.include_router(ws_router)


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "rainfall-prediction"}
