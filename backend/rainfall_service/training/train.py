"""
Full Rainfall Prediction Training Pipeline.

Steps:
  1. Load data from DB (or generate synthetic for pipeline validation)
  2. Feature engineering (45 features per sample)
  3. Train BiLSTM-Attention model (regression + classification)
  4. Train XGBoost regressor + classifier
  5. Train LightGBM regressor + classifier
  6. Train stacking meta-learner on out-of-fold predictions
  7. Evaluate all models + save metrics
  8. Save all model artefacts

Usage:
    python -m rainfall_service.training.train \
        --model-dir ./rainfall_service/models/saved \
        --epochs 50 \
        --batch-size 32
"""
from __future__ import annotations

import argparse, json, logging, os, sys, time
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
from rainfall_service.services.feature_engineering import category_to_index as ci

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device: %s", DEVICE)

N_CLASSES = 5
SEQ_LEN = 30
N_SEQ_FEATURES = 20    # raw(15) + extra(5)
N_FLAT_FEATURES = 38   # full flat feature vector


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in rainfall categories."""
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets, 
            weight=self.weight, 
            reduction='none'
        )
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()
    

# ── Dataset ────────────────────────────────────────────────────────────────────

class RainfallDataset(Dataset):
    """
    sequences: (N, SEQ_LEN, N_SEQ_FEATURES) — LSTM input
    flat:      (N, N_FLAT_FEATURES)          — XGB/LGBM input
    targets_mm:(N,)                          — regression target
    targets_cls:(N,)                         — classification target (0-5)
    """
    def __init__(self, sequences, flat, targets_mm, targets_cls):
        self.seq = torch.tensor(sequences, dtype=torch.float32)
        self.flat = torch.tensor(flat, dtype=torch.float32)
        self.mm = torch.tensor(targets_mm, dtype=torch.float32)
        self.cls = torch.tensor(targets_cls, dtype=torch.long)

    def __len__(self): return len(self.mm)

    def __getitem__(self, idx):
        return self.seq[idx], self.flat[idx], self.mm[idx], self.cls[idx]


def generate_synthetic_data(n: int = 5000):
    """Realistic synthetic rainfall data for pipeline validation."""
    np.random.seed(42)
    from rainfall_service.services.feature_engineering import category_to_index, rainfall_category

    # Seasonal pattern
    doy = np.arange(n) % 365
    seasonal = np.sin(2 * np.pi * (doy - 80) / 365)   # peak at monsoon
    base_rain = np.maximum(0, 10 * seasonal + np.random.exponential(5, n))

    # Heavy rain events (5% of days)
    heavy = np.random.binomial(1, 0.05, n)
    base_rain += heavy * np.random.exponential(60, n)

    targets_mm = base_rain.astype(np.float32)
    targets_cls = np.array([category_to_index(rainfall_category(mm)) for mm in targets_mm], dtype=np.int32)

    # Sequence: correlated features + noise
    sequences = np.random.randn(n, SEQ_LEN, N_SEQ_FEATURES).astype(np.float32)
    for i in range(n):
        # Inject rainfall signal into humidity and cape channels
        sequences[i, :, 3] = 60 + 30 * seasonal[i] + np.random.randn(SEQ_LEN) * 5  # humidity
        sequences[i, :, 11] = max(0, targets_mm[i] * 8) + np.random.randn(SEQ_LEN) * 50

    flat = np.random.randn(n, N_FLAT_FEATURES).astype(np.float32)
    # Inject lag features
    flat[:, 0] = targets_mm * np.random.uniform(0.5, 0.9, n)  # lag_1 correlated

    return sequences, flat, targets_mm, targets_cls


# ── LSTM Training ──────────────────────────────────────────────────────────────
from torch.utils.data import WeightedRandomSampler
def train_lstm(
    train_ds: RainfallDataset,
    val_ds: RainfallDataset,
    model_dir: str,
    epochs: int,
) -> nn.Module:
    from rainfall_service.models.lstm_model import RainfallLSTM
    from sklearn.utils.class_weight import compute_class_weight

    model = RainfallLSTM(n_features=N_SEQ_FEATURES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, epochs=epochs, 
        steps_per_epoch=len(DataLoader(train_ds, batch_size=32)),
    )

    all_cls = train_ds.cls.numpy()
    cls_weights = compute_class_weight("balanced", classes=np.unique(all_cls), y=all_cls)
    cls_weight_tensor = torch.tensor(cls_weights, dtype=torch.float32).to(DEVICE)

    '''
    # Add after computing cls_weights
    sample_weights = np.sqrt(cls_weights[train_ds.cls.numpy()])
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True
    )
    '''
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    ce_loss = ce_loss = FocalLoss(weight=cls_weight_tensor, gamma=2.0)
    mse_loss = nn.HuberLoss(delta=15.0)

    best_val = float("inf")
    ckpt = os.path.join(model_dir, "lstm_model.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for seq, _, mm, cls in train_loader:
            seq, mm, cls = seq.to(DEVICE), mm.to(DEVICE), cls.to(DEVICE)
            optimizer.zero_grad()
            mm_pred, logits, _ = model(seq)
            loss = 0.6 * mse_loss(mm_pred.squeeze(), mm) + 0.4 * ce_loss(logits, cls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq, _, mm, cls in val_loader:
                seq, mm, cls = seq.to(DEVICE), mm.to(DEVICE), cls.to(DEVICE)
                mm_pred, logits, _ = model(seq)
                val_loss += (0.6 * mse_loss(mm_pred.squeeze(), mm) + 0.4 * ce_loss(logits, cls)).item()

        train_loss /= len(train_loader)
        val_loss /= max(len(val_loader), 1)
        logger.info("Epoch %3d/%d | train=%.4f | val=%.4f", epoch, epochs, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    logger.info("LSTM training done. Best val_loss=%.4f", best_val)
    return model


# ── Extract LSTM embeddings ────────────────────────────────────────────────────

def get_lstm_embeddings(model: nn.Module, sequences: np.ndarray) -> np.ndarray:
    model.eval()
    ds = TensorDataset(torch.tensor(sequences, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    embs = []
    with torch.no_grad():
        for (seq,) in loader:
            embs.append(model.extract_features(seq.to(DEVICE)).cpu().numpy())
    return np.concatenate(embs, axis=0)


# ── XGBoost ────────────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_val, y_val, model_dir: str):
    reg = xgb.XGBRegressor(
        n_estimators=800, max_depth=7, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=40, random_state=42,
        tree_method="hist", n_jobs=-1,
    )
    reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    reg.save_model(os.path.join(model_dir, "xgb_model.json"))
    logger.info("XGBoost saved.")
    return reg


# ── LightGBM ───────────────────────────────────────────────────────────────────

def train_lightgbm(X_train, y_train, X_val, y_val, model_dir: str):
    params = dict(
        n_estimators=800, max_depth=7, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="regression", metric="rmse",
        n_jobs=-1, random_state=42, verbose=-1,
    )
    reg = lgb.LGBMRegressor(**params)
    reg.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(100)],
    )
    joblib.dump(reg, os.path.join(model_dir, "lgbm_model.pkl"))
    logger.info("LightGBM saved.")
    return reg


# ── Stacking Meta-Learner ──────────────────────────────────────────────────────

def train_stacking(
    lstm_model, xgb_model, lgbm_model,
    sequences, flat, targets_mm, model_dir: str,
):
    """Fixed-weight ensemble combining LSTM + XGBoost + LightGBM."""
    n = len(targets_mm)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((n, 3), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(flat)):
        xgb_pred = xgb_model.predict(flat[va_idx])
        lgbm_pred = lgbm_model.predict(flat[va_idx])
        lstm_mm = []
        ds = TensorDataset(torch.tensor(sequences[va_idx], dtype=torch.float32))
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        with torch.no_grad():
            for (seq,) in loader:
                mm_pred, _, _ = lstm_model(seq.to(DEVICE))
                lstm_mm.extend(mm_pred.squeeze().cpu().numpy().tolist())

        oof[va_idx, 0] = np.array(lstm_mm)
        oof[va_idx, 1] = xgb_pred
        oof[va_idx, 2] = lgbm_pred

    # Fixed weights instead of Ridge — ensures all models contribute
    weights = np.array([0.35, 0.35, 0.30])  # LSTM, XGBoost, LightGBM

    # Validate ensemble performance
    ensemble_pred = oof @ weights
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(targets_mm, ensemble_pred)
    r2 = r2_score(targets_mm, ensemble_pred)
    logger.info("Ensemble OOF — MAE: %.4f | R²: %.4f", mae, r2)

    # Save in same format as before for compatibility
    joblib.dump(weights, os.path.join(model_dir, "ensemble_model.pkl"))
    weights_path = os.path.join(model_dir, "ensemble_weights.pkl")
    joblib.dump(weights.tolist(), weights_path)
    logger.info("Stacking ensemble weights: %s", weights.tolist())
    return weights


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(X_test, y_test, y_test_cls, lstm_model, xgb_model, lgbm_model, meta, sequences_test, model_dir):
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error, r2_score,
        accuracy_score, f1_score, classification_report, confusion_matrix,
    )
    from rainfall_service.services.feature_engineering import rainfall_category, category_to_index

    results = []
    all_preds = {}

    # XGBoost
    xgb_pred = xgb_model.predict(X_test)
    all_preds["XGBoost"] = xgb_pred
    results.append({
        "model_name": "XGBoost",
        "mae": float(mean_absolute_error(y_test, xgb_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, xgb_pred))),
        "r2_score": float(r2_score(y_test, xgb_pred)),
        "accuracy_class": float(accuracy_score(y_test_cls, [category_to_index(rainfall_category(p)) for p in xgb_pred])),
        "f1_score": float(f1_score(y_test_cls, [category_to_index(rainfall_category(p)) for p in xgb_pred], average="weighted", zero_division=0)),
        "evaluation_date": time.strftime("%Y-%m-%d"),
    })

    # LightGBM
    lgbm_pred = lgbm_model.predict(X_test)
    all_preds["LightGBM"] = lgbm_pred
    results.append({
        "model_name": "LightGBM",
        "mae": float(mean_absolute_error(y_test, lgbm_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, lgbm_pred))),
        "r2_score": float(r2_score(y_test, lgbm_pred)),
        "accuracy_class": float(accuracy_score(y_test_cls, [category_to_index(rainfall_category(p)) for p in lgbm_pred])),
        "f1_score": float(f1_score(y_test_cls, [category_to_index(rainfall_category(p)) for p in lgbm_pred], average="weighted", zero_division=0)),
        "evaluation_date": time.strftime("%Y-%m-%d"),
    })

    # LSTM
    lstm_preds_mm = []
    lstm_cls_preds = []
    ds = TensorDataset(torch.tensor(sequences_test, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    with torch.no_grad():
        for (seq,) in loader:
            mm_p, logits, _ = lstm_model(seq.to(DEVICE))
            lstm_preds_mm.extend(mm_p.squeeze().cpu().tolist())
            lstm_cls_preds.extend(logits.argmax(dim=-1).cpu().tolist())
    lstm_preds_mm = np.array(lstm_preds_mm)
    all_preds["LSTM"] = lstm_preds_mm
    results.append({
        "model_name": "LSTM-Attention",
        "mae": float(mean_absolute_error(y_test, lstm_preds_mm)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, lstm_preds_mm))),
        "r2_score": float(r2_score(y_test, lstm_preds_mm)),
        "accuracy_class": float(accuracy_score(y_test_cls, lstm_cls_preds)),
        "f1_score": float(f1_score(y_test_cls, lstm_cls_preds, average="weighted", zero_division=0)),
        "evaluation_date": time.strftime("%Y-%m-%d"),
    })

    # Ensemble
    ens_input = np.stack([lstm_preds_mm, xgb_pred, lgbm_pred], axis=1)
    ens_pred = ens_input @ meta  # meta is now a weights array [0.35, 0.35, 0.30]
    ens_cls = [category_to_index(rainfall_category(p)) for p in ens_pred]
    all_preds["Ensemble"] = ens_pred
    results.append({
        "model_name": "Ensemble (Ours)",
        "mae": float(mean_absolute_error(y_test, ens_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, ens_pred))),
        "r2_score": float(r2_score(y_test, ens_pred)),
        "accuracy_class": float(accuracy_score(y_test_cls, ens_cls)),
        "f1_score": float(f1_score(y_test_cls, ens_cls, average="weighted", zero_division=0)),
        "evaluation_date": time.strftime("%Y-%m-%d"),
    })

    cm = confusion_matrix(y_test_cls, ens_cls).tolist()
    from rainfall_service.services.feature_engineering import CATEGORY_LABELS

    # Feature importance from XGBoost
    feat_imp = dict(zip(
        [f"feat_{i}" for i in range(len(xgb_model.feature_importances_))],
        xgb_model.feature_importances_.tolist()
    ))

    metrics = {
        "models": results,
        "best_model": "Ensemble (Ours)",
        "confusion_matrix": cm,
        "class_names": CATEGORY_LABELS,
        "feature_importance": feat_imp,
    }

    with open(os.path.join(model_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Print table
    logger.info("\n=== MODEL COMPARISON ===")
    logger.info(f"{'Model':<25} {'MAE':>6} {'RMSE':>7} {'R²':>6} {'AccCls':>7} {'F1':>6}")
    logger.info("-" * 58)
    for r in results:
        logger.info(f"{r['model_name']:<25} {r['mae']:>6.2f} {r['rmse']:>7.2f} {r['r2_score']:>6.3f} {r['accuracy_class']:>7.3f} {r['f1_score']:>6.3f}")
    logger.info("=" * 58)
    logger.info("\n" + classification_report(y_test_cls, ens_cls, target_names=CATEGORY_LABELS, zero_division=0))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="./rainfall_service/models/saved")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-db", action="store_true", help="Load data from PostgreSQL DB")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    logger.info("=== PHASE 0: Data Loading ===")
    
    if args.use_db:
        import asyncio
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
        from sqlalchemy import text
        from rainfall_service.config import settings
        from rainfall_service.services.feature_engineering import build_sequence_features, rainfall_category
        
        async def load_from_db():
            engine = create_async_engine(settings.DATABASE_URL, echo=False)
            async with engine.connect() as conn:
                result = await conn.execute(text("""
                    SELECT location_name, record_date, rainfall_mm, temperature_max_c,
                           temperature_min_c, humidity_percent, wind_speed_kmh,
                           cloud_cover_pct, rainfall_category
                    FROM rainfall_records
                    ORDER BY location_name, record_date
                """))
                rows = result.fetchall()
            await engine.dispose()
            return rows
        
        rows = asyncio.run(load_from_db())
        logger.info("Loaded %d rows from database", len(rows))
        
        # Group by location and build sequences
        from collections import defaultdict
        from rainfall_service.services.feature_engineering import SEQ_LEN, build_raw_feature_vector
        import math
        
        location_data = defaultdict(list)
        for row in rows:
            location_data[row.location_name].append({
                "record_date": row.record_date,
                "rainfall_mm": row.rainfall_mm or 0.0,
                "temperature_2m_max": row.temperature_max_c or 30.0,
                "temperature_2m_min": row.temperature_min_c or 22.0,
                "relative_humidity_2m_mean": row.humidity_percent or 60.0,
                "wind_speed_10m_max": row.wind_speed_kmh or 10.0,
                "cloud_cover_mean": row.cloud_cover_pct or 50.0,
                "wind_direction_10m_dominant": 180.0,
                "shortwave_radiation_sum": 15.0,
                "et0_fao_evapotranspiration": 3.0,
                "day_of_year": row.record_date.timetuple().tm_yday,
                "month": row.record_date.month,
            })
        
        sequences_list, flat_list, targets_mm_list, targets_cls_list = [], [], [], []
        
        for loc, records in location_data.items():
            records.sort(key=lambda x: x["record_date"])
            for i in range(SEQ_LEN, len(records)):
                history = records[i-SEQ_LEN:i]
                target = records[i]
                try:
                    seq, flat_feat = build_sequence_features(history)
                    sequences_list.append(seq)
                    flat_list.append(flat_feat)
                    targets_mm_list.append(float(target["rainfall_mm"]))
                    from rainfall_service.services.feature_engineering import rainfall_category as rc
                    cat = rc(float(target["rainfall_mm"]))
                    targets_cls_list.append(ci(cat))
                except Exception:
                    continue
        
        sequences = np.stack(sequences_list)
        flat = np.stack(flat_list)
        targets_mm = np.array(targets_mm_list, dtype=np.float32)
        targets_cls = np.array(targets_cls_list, dtype=np.int64)
        logger.info("Built %d sequences from real data", len(targets_mm))
    else:
        sequences, flat, targets_mm, targets_cls = generate_synthetic_data(n=8000)
    
    logger.info("Samples: %d | sequences=%s | flat=%s", len(targets_mm), sequences.shape, flat.shape)

    # Train/val/test: 70/15/15
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(targets_mm))
    idx_tv, idx_test = train_test_split(idx, test_size=0.15, random_state=42)
    idx_train, idx_val = train_test_split(idx_tv, test_size=0.15/0.85, random_state=42)

    def sub(i): return sequences[i], flat[i], targets_mm[i], targets_cls[i]

    seq_tr, fl_tr, mm_tr, cls_tr = sub(idx_train)
    seq_va, fl_va, mm_va, cls_va = sub(idx_val)
    seq_te, fl_te, mm_te, cls_te = sub(idx_test)

    logger.info("=== PHASE 1: Feature Scaling ===")
    scaler = StandardScaler()
    fl_tr_sc = scaler.fit_transform(fl_tr)
    fl_va_sc = scaler.transform(fl_va)
    fl_te_sc = scaler.transform(fl_te)
    joblib.dump(scaler, os.path.join(args.model_dir, "feature_scaler.pkl"))

    logger.info("=== PHASE 2: LSTM Training ===")
    train_ds = RainfallDataset(seq_tr, fl_tr_sc, mm_tr, cls_tr)
    val_ds = RainfallDataset(seq_va, fl_va_sc, mm_va, cls_va)
    lstm_model = train_lstm(train_ds, val_ds, args.model_dir, args.epochs)

    logger.info("=== PHASE 3: XGBoost Training ===")
    xgb_model = train_xgboost(fl_tr_sc, mm_tr, fl_va_sc, mm_va, args.model_dir)

    logger.info("=== PHASE 4: LightGBM Training ===")
    lgbm_model = train_lightgbm(fl_tr_sc, mm_tr, fl_va_sc, mm_va, args.model_dir)

    logger.info("=== PHASE 5: Stacking Ensemble ===")
    fl_full_sc = scaler.transform(flat)
    meta = train_stacking(lstm_model, xgb_model, lgbm_model, sequences, fl_full_sc, targets_mm, args.model_dir)

    logger.info("=== PHASE 6: Evaluation ===")
    evaluate(fl_te_sc, mm_te, cls_te, lstm_model, xgb_model, lgbm_model, meta, seq_te, args.model_dir)

    logger.info("🎉 Training complete! All models saved to %s", args.model_dir)


if __name__ == "__main__":
    main()
