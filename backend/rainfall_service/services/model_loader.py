"""Model Loader — loads LSTM + XGBoost + LightGBM + Ensemble on startup."""
from __future__ import annotations

import logging
import os
import joblib
import torch
import xgboost as xgb

from rainfall_service.config import settings
from rainfall_service.models.lstm_model import RainfallLSTM

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelLoader:
    lstm: RainfallLSTM | None = None
    xgb_model: xgb.XGBRegressor | None = None
    xgb_classifier: xgb.XGBClassifier | None = None
    lgbm_model = None
    scaler = None
    label_encoder = None
    ensemble = None
    model_version: str = "1.0.0"
    _loaded: bool = False

    @classmethod
    async def load_all(cls):
        model_dir = settings.MODEL_DIR
        os.makedirs(model_dir, exist_ok=True)

        # ── LSTM ──────────────────────────────────────────────────────────────
        cls.lstm = RainfallLSTM(n_features=20).to(DEVICE)
        lstm_path = os.path.join(model_dir, settings.LSTM_MODEL)
        if os.path.exists(lstm_path):
            cls.lstm.load_state_dict(torch.load(lstm_path, map_location=DEVICE))
            logger.info("LSTM model loaded from %s", lstm_path)
        else:
            logger.warning("LSTM weights not found — using random init.")
        cls.lstm.eval()

        # ── XGBoost ───────────────────────────────────────────────────────────
        cls.xgb_model = xgb.XGBRegressor()
        xgb_path = os.path.join(model_dir, settings.XGBOOST_MODEL)
        if os.path.exists(xgb_path):
            cls.xgb_model.load_model(xgb_path)
            logger.info("XGBoost model loaded.")

        # ── LightGBM ──────────────────────────────────────────────────────────
        lgbm_path = os.path.join(model_dir, settings.LGBM_MODEL)
        if os.path.exists(lgbm_path):
            cls.lgbm_model = joblib.load(lgbm_path)
            logger.info("LightGBM model loaded.")

        # ── Scaler / Encoder ──────────────────────────────────────────────────
        scaler_path = os.path.join(model_dir, settings.SCALER_FILE)
        if os.path.exists(scaler_path):
            cls.scaler = joblib.load(scaler_path)

        enc_path = os.path.join(model_dir, settings.LABEL_ENCODER)
        if os.path.exists(enc_path):
            cls.label_encoder = joblib.load(enc_path)

        # ── Ensemble weights ──────────────────────────────────────────────────
        ens_path = os.path.join(model_dir, "ensemble_weights.pkl")
        if os.path.exists(ens_path):
            cls.ensemble = joblib.load(ens_path)

        cls._loaded = True

    @classmethod
    def is_ready(cls) -> bool:
        return cls._loaded

    @classmethod
    def get_device(cls) -> torch.device:
        return DEVICE
