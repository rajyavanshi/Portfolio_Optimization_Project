"""
Stage 3.4 — Alpha Signal Generation
------------------------------------
Generates next-week expected returns ("alphas")
using the best trained model (by IC) and the latest factor data.

Author: Suraj Prakash
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ==============================================================
# CONFIGURATION
# ==============================================================
PROJECT_ROOT = Path(r"D:\Portfolio Optimzation project")
FACTOR_FILE = PROJECT_ROOT / "results" / "factors" / "price_factors.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
SELECTED_FEATURES_FILE = ARTIFACTS_DIR / "selected_features.json"
METRICS_FILE = ARTIFACTS_DIR / "metrics_summary.csv"
OUTPUT_FILE = ARTIFACTS_DIR / "alpha_predictions.csv"

# ==============================================================
# LOAD BEST MODEL BASED ON IC
# ==============================================================
print(" Loading best model based on IC...")

metrics = pd.read_csv(METRICS_FILE)
best_row = metrics.groupby("model", as_index=False)["ic"].mean().sort_values("ic", ascending=False).iloc[0]
best_model_name = best_row["model"]
print(f" Best model selected: {best_model_name}")

# Find best model file (from latest fold)
model_files = sorted(MODELS_DIR.glob(f"{best_model_name}_fold*.pkl"))
if not model_files:
    raise FileNotFoundError(f" No trained model found for {best_model_name}")
model_path = model_files[-1]
model = joblib.load(model_path)
print(f" Loaded model from {model_path.name}")

# ==============================================================
# LOAD SELECTED FEATURES
# ==============================================================
with open(SELECTED_FEATURES_FILE, "r") as f:
    sel = json.load(f)
features = sel["selected_features"]
print(f" Selected features: {features}")

# ==============================================================
# LOAD SCALER (same fold as model)
# ==============================================================
fold_number = ''.join(filter(str.isdigit, model_path.stem))
scaler_file = MODELS_DIR / f"scaler_fold{fold_number}.pkl"

if scaler_file.exists():
    scaler = joblib.load(scaler_file)
    print(f" Loaded scaler from {scaler_file.name}")
else:
    print(" No scaler found — using raw (unscaled) features.")
    scaler = None

# ==============================================================
# LOAD LATEST FACTOR DATA
# ==============================================================
print(f" Loading factor data from {FACTOR_FILE.name} ...")
df = pd.read_csv(FACTOR_FILE)
df.columns = [c.strip().lower() for c in df.columns]

required_cols = ["date", "ticker"] + features
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f" Missing column in factor data: {col}")

# Get the most recent date snapshot
df["date"] = pd.to_datetime(df["date"])
latest_date = df["date"].max()
latest_df = df[df["date"] == latest_date].reset_index(drop=True)
print(f" Using latest date: {latest_date.date()} | tickers: {latest_df['ticker'].nunique()}")

# Extract feature matrix
X = latest_df[features].copy()
if scaler:
    X_scaled = pd.DataFrame(scaler.transform(X), columns=features)
else:
    X_scaled = X.copy()

# ==============================================================
# PREDICTIONS
# ==============================================================
print(" Generating alpha predictions ...")
preds = model.predict(X_scaled)

alpha_df = latest_df[["date", "ticker"]].copy()
alpha_df["predicted_return"] = preds
alpha_df.sort_values("predicted_return", ascending=False, inplace=True)
alpha_df.to_csv(OUTPUT_FILE, index=False)

print(f" Alpha predictions saved to: {OUTPUT_FILE}")
print("\n Top predicted returns:")
print(alpha_df.head(10).to_string(index=False))
