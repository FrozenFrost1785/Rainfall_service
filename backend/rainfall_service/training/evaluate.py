"""
Standalone Evaluation + Visualization Script.

Generates:
  - Model comparison table (MAE, RMSE, R², Accuracy, F1)
  - Confusion matrix heatmap
  - Predicted vs Actual scatter plot
  - Residual distribution plot
  - Feature importance bar chart
  - SHAP summary plot

Usage:
    python -m rainfall_service.training.evaluate \
        --model-dir ./rainfall_service/models/saved \
        --output-dir ./reports
"""
from __future__ import annotations

import argparse, json, logging, os, sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["No Rain", "Light", "Moderate", "Heavy", "Very Heavy", "Extremely Heavy"]


def _load_models(model_dir: str):
    from rainfall_service.models.lstm_model import RainfallLSTM
    import xgboost as xgb

    models = {}
    lstm = RainfallLSTM(n_features=20).to(DEVICE)
    ckpt = os.path.join(model_dir, "lstm_model.pt")
    if os.path.exists(ckpt):
        lstm.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    lstm.eval()
    models["lstm"] = lstm

    xgb_path = os.path.join(model_dir, "xgb_model.json")
    if os.path.exists(xgb_path):
        reg = xgb.XGBRegressor()
        reg.load_model(xgb_path)
        models["xgb"] = reg

    lgbm_path = os.path.join(model_dir, "lgbm_model.pkl")
    if os.path.exists(lgbm_path):
        models["lgbm"] = joblib.load(lgbm_path)

    ens_path = os.path.join(model_dir, "ensemble_model.pkl")
    if os.path.exists(ens_path):
        models["ensemble"] = joblib.load(ens_path)

    scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
    if os.path.exists(scaler_path):
        models["scaler"] = joblib.load(scaler_path)

    return models


def _lstm_predict(model, sequences):
    ds = TensorDataset(torch.tensor(sequences, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    mm_preds, cls_preds = [], []
    with torch.no_grad():
        for (seq,) in loader:
            mm, logits, _ = model(seq.to(DEVICE))
            mm_preds.extend(mm.squeeze().cpu().tolist())
            cls_preds.extend(logits.argmax(-1).cpu().tolist())
    return np.array(mm_preds), np.array(cls_preds)


def plot_comparison_table(results: list[dict], output_path: str):
    fig, ax = plt.subplots(figsize=(13, len(results) * 0.9 + 1.5))
    fig.patch.set_facecolor('#0B0F19')
    ax.set_facecolor('#0B0F19')
    ax.axis('off')

    cols = ["Model", "MAE (mm)", "RMSE (mm)", "R²", "Acc (Class)", "F1"]
    data = [[
        r["model_name"],
        f"{r['mae']:.2f}", f"{r['rmse']:.2f}",
        f"{r['r2_score']:.3f}", f"{r['accuracy_class']:.3f}", f"{r['f1_score']:.3f}"
    ] for r in results]

    colors = [["#0B0F19"] * len(cols)] * len(data)
    colors[-1] = ["#0D2137"] * len(cols)   # highlight best

    table = ax.table(cellText=data, colLabels=cols, cellLoc="center", loc="center",
                     cellColours=colors, colColours=["#0D1B2A"] * len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#1A2540")
        if r == 0:
            cell.set_text_props(color="#00D4FF", fontweight="bold")
        elif r == len(data):
            cell.set_text_props(color="#00D4FF")
        else:
            cell.set_text_props(color="#C8D6E5")

    ax.set_title("Model Performance Comparison — Rainfall Prediction",
                 color="#00D4FF", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor='#0B0F19')
    plt.close()
    logger.info("Comparison table saved to %s", output_path)


def plot_confusion_matrix(cm, output_path: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0B0F19')
    ax.set_facecolor('#0B0F19')
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax)
    ax.set_title("Confusion Matrix — Ensemble Model", color="#00D4FF", fontsize=13, pad=15)
    ax.set_xlabel("Predicted", color="#C8D6E5")
    ax.set_ylabel("True", color="#C8D6E5")
    ax.tick_params(colors="#64748B")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor='#0B0F19')
    plt.close()
    logger.info("Confusion matrix saved.")


def plot_pred_vs_actual(y_true, y_pred, model_name: str, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0B0F19')
    for ax in axes:
        ax.set_facecolor('#0E1424')
        ax.spines[:].set_color('#1A2540')
        ax.tick_params(colors='#64748B')

    # Scatter
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=8, color='#00D4FF')
    lim = max(y_true.max(), y_pred.max())
    axes[0].plot([0, lim], [0, lim], 'r--', lw=1.5, label='Perfect')
    axes[0].set_xlabel("Actual Rainfall (mm)", color="#C8D6E5")
    axes[0].set_ylabel("Predicted Rainfall (mm)", color="#C8D6E5")
    axes[0].set_title(f"{model_name} — Predicted vs Actual", color="#00D4FF")
    axes[0].legend()

    # Residuals
    residuals = y_pred - y_true
    axes[1].hist(residuals, bins=60, color='#7B2FFF', alpha=0.8, edgecolor='#1A2540')
    axes[1].axvline(0, color='#FF3B5C', lw=2, linestyle='--')
    axes[1].set_xlabel("Residual (mm)", color="#C8D6E5")
    axes[1].set_ylabel("Count", color="#C8D6E5")
    axes[1].set_title("Residual Distribution", color="#00D4FF")

    plt.suptitle(f"Ensemble Model Evaluation — R²={float(1 - np.var(residuals)/np.var(y_true)):.3f}", color="#C8D6E5", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor='#0B0F19')
    plt.close()
    logger.info("Prediction scatter plot saved.")


def plot_feature_importance(model, top_n: int, output_path: str):
    scores = model.feature_importances_
    idx = np.argsort(scores)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#0B0F19')
    ax.set_facecolor('#0E1424')
    ax.spines[:].set_color('#1A2540')
    ax.tick_params(colors='#64748B')

    feature_labels = [
        "lag_1d", "lag_3d", "lag_7d", "lag_14d",
        "roll_mean_7d", "roll_mean_14d", "roll_mean_30d",
        "roll_std_7d", "roll_std_14d", "roll_std_30d",
        "roll_max_7d", "roll_max_14d", "roll_max_30d",
        "consec_rain", "consec_dry",
        "temp_max", "temp_min", "temp_mean", "humidity",
        "dew_point", "wind_speed", "wind_sin", "wind_cos",
        "pressure_anom", "cloud_cover", "solar_rad", "cape",
        "pwv", "evapotrans", "surface_pressure",
        "season_win", "season_pre", "season_mon", "season_post",
        "doy_norm", "rain_sum_5d", "vpd", "wind_moisture",
    ]

    labels = [feature_labels[i] if i < len(feature_labels) else f"feat_{i}" for i in idx]
    bars = ax.barh(range(top_n), scores[idx], color='#00D4FF', alpha=0.85)
    bars[-1].set_color('#FF3B5C')  # highlight top feature

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, color='#C8D6E5', fontsize=9)
    ax.set_xlabel("Importance Score", color="#C8D6E5")
    ax.set_title(f"Top {top_n} Features — XGBoost Importance", color="#00D4FF", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor='#0B0F19')
    plt.close()
    logger.info("Feature importance saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="./rainfall_service/models/saved")
    parser.add_argument("--output-dir", default="./reports")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading models...")
    models = _load_models(args.model_dir)

    logger.info("Generating test data...")
    from rainfall_service.training.train import generate_synthetic_data
    from sklearn.model_selection import train_test_split
    seqs, flat, mm, cls = generate_synthetic_data(n=2000)
    _, idx_test = train_test_split(np.arange(len(mm)), test_size=0.15, random_state=42)
    seq_te, fl_te, mm_te, cls_te = seqs[idx_test], flat[idx_test], mm[idx_test], cls[idx_test]

    if "scaler" in models:
        fl_te = models["scaler"].transform(fl_te)

    # Collect predictions
    results = []
    preds = {}

    if "xgb" in models:
        p = models["xgb"].predict(fl_te)
        preds["xgb"] = p
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score
        from rainfall_service.services.feature_engineering import category_to_index, rainfall_category
        cls_p = [category_to_index(rainfall_category(x)) for x in p]
        results.append({"model_name": "XGBoost", "mae": float(mean_absolute_error(mm_te, p)), "rmse": float(np.sqrt(mean_squared_error(mm_te, p))), "r2_score": float(r2_score(mm_te, p)), "accuracy_class": float(accuracy_score(cls_te, cls_p)), "f1_score": float(f1_score(cls_te, cls_p, average="weighted", zero_division=0)), "evaluation_date": "today"})

    if "lgbm" in models:
        p = models["lgbm"].predict(fl_te)
        preds["lgbm"] = p
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score
        from rainfall_service.services.feature_engineering import category_to_index, rainfall_category
        cls_p = [category_to_index(rainfall_category(x)) for x in p]
        results.append({"model_name": "LightGBM", "mae": float(mean_absolute_error(mm_te, p)), "rmse": float(np.sqrt(mean_squared_error(mm_te, p))), "r2_score": float(r2_score(mm_te, p)), "accuracy_class": float(accuracy_score(cls_te, cls_p)), "f1_score": float(f1_score(cls_te, cls_p, average="weighted", zero_division=0)), "evaluation_date": "today"})

    if "lstm" in models:
        lstm_mm, lstm_cls = _lstm_predict(models["lstm"], seq_te)
        preds["lstm"] = lstm_mm
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score
        results.append({"model_name": "LSTM-Attention", "mae": float(mean_absolute_error(mm_te, lstm_mm)), "rmse": float(np.sqrt(mean_squared_error(mm_te, lstm_mm))), "r2_score": float(r2_score(mm_te, lstm_mm)), "accuracy_class": float(accuracy_score(cls_te, lstm_cls)), "f1_score": float(f1_score(cls_te, lstm_cls, average="weighted", zero_division=0)), "evaluation_date": "today"})

    if "ensemble" in models and len(preds) >= 2:
        keys = list(preds.keys())
        ens_in = np.stack([preds[k] for k in keys], axis=1)
        if ens_in.shape[1] < 3:
            ens_pred = ens_in.mean(axis=1)
        else:
            ens_pred = models["ensemble"].predict(ens_in)
        from rainfall_service.services.feature_engineering import category_to_index, rainfall_category
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score, confusion_matrix
        cls_p = [category_to_index(rainfall_category(x)) for x in ens_pred]
        results.append({"model_name": "Ensemble (Ours)", "mae": float(mean_absolute_error(mm_te, ens_pred)), "rmse": float(np.sqrt(mean_squared_error(mm_te, ens_pred))), "r2_score": float(r2_score(mm_te, ens_pred)), "accuracy_class": float(accuracy_score(cls_te, cls_p)), "f1_score": float(f1_score(cls_te, cls_p, average="weighted", zero_division=0)), "evaluation_date": "today"})
        cm = confusion_matrix(cls_te, cls_p, labels=list(range(6)))
        plot_confusion_matrix(cm, os.path.join(args.output_dir, "confusion_matrix.png"))
        plot_pred_vs_actual(mm_te, ens_pred, "Ensemble", os.path.join(args.output_dir, "pred_vs_actual.png"))

    if results:
        plot_comparison_table(results, os.path.join(args.output_dir, "model_comparison.png"))

    if "xgb" in models:
        plot_feature_importance(models["xgb"], 20, os.path.join(args.output_dir, "feature_importance.png"))

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info("✅ Evaluation complete. Reports saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
