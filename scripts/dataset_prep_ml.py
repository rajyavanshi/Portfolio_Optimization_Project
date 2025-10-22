"""
scripts/dataset_prep_ml.py
Step 3.1: Build training_data.csv linking factors (t) -> forward return (t+H).
Usage:
    python scripts/dataset_prep_ml.py
Outputs:
    - data/training_data.csv
    - artifacts/scaler.pkl
    - artifacts/feature_list.json
"""

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import joblib

# --------------------------
# CONFIG (edit if needed)
# --------------------------
PROJECT_ROOT = Path(r"D:\Portfolio Optimzation project")  
FACTORS_FILE = PROJECT_ROOT /"results"/ "factors" / "price_factors.csv"   # expected: Date, Ticker, Factor1,...
MERGED_PRICE_FILE = PROJECT_ROOT / "processed_data" / "merged_prices.csv"  # optional: Date, Ticker, Close
OUTPUT_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
HORIZON_DAYS = 5   # forward horizon (5 trading days = ~1 week)
TRAIN_TEST_SPLIT_RATIO = 0.8
WINSOR_PCT = (0.01, 0.99)  # lower, upper percentile for winsorization
MAX_ROW_NA_RATIO = 0.2     # drop rows with >20% features missing
MAX_COL_NA_RATIO = 0.3     # drop features with >30% missing
RANDOM_SEED = 42

# create dirs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------
# Helper functions
# --------------------------
def compute_forward_returns(prices_df, horizon=5):
    """
    prices_df: columns ['Date','Ticker','Close'] (Date sorted ascending)
    returns forward by horizon trading rows per ticker.
    """
    prices_df = prices_df.sort_values(["ticker", "date"])
    prices_df["forward_close"] = prices_df.groupby("ticker")["close"].shift(-horizon)
    prices_df["forward_return"] = prices_df["forward_close"] / prices_df["close"] - 1.0
    return prices_df.drop(columns=["forward_close"])

def winsorize_series(s, lower=0.01, upper=0.99):
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lower=lo, upper=hi)

# --------------------------
# Load data
# --------------------------
print("Loading factor data from:", FACTORS_FILE)
factors = pd.read_csv(FACTORS_FILE, parse_dates=["date"])

# Try to find price data
if MERGED_PRICE_FILE.exists():
    print("Loading merged price file:", MERGED_PRICE_FILE)
    prices = pd.read_csv(MERGED_PRICE_FILE, parse_dates=["date"])
    # Expect columns Date, Ticker, Close
else:
    raise FileNotFoundError(f"Price file not found at {MERGED_PRICE_FILE}. Please provide merged prices.")

# --------------------------
# Compute forward returns (target)
# --------------------------
print(f"Computing forward returns with horizon = {HORIZON_DAYS} days")
prices_with_fwd = compute_forward_returns(prices, horizon=HORIZON_DAYS)
targets = prices_with_fwd[["date", "ticker", "forward_return"]].rename(columns={"forward_return": "target_return"})

# --------------------------
# Merge factors with target
# --------------------------
print("Merging factors with forward returns (aligning by Date & Ticker)")
df = pd.merge(factors, targets, on=["date", "ticker"], how="left")

# Drop rows where target is missing (we cannot train on missing future returns)
before = len(df)
df = df.dropna(subset=["target_return"])
print(f"Dropped {before - len(df)} rows with missing target. Remaining rows: {len(df)}")

# --------------------------
# Feature selection / cleaning
# --------------------------
feature_cols = [c for c in df.columns if c not in ("date", "ticker", "target_return")]
print(f"Initial feature count: {len(feature_cols)}")

# 1) Drop columns with too many NaNs
col_na_frac = df[feature_cols].isna().mean()
drop_cols = col_na_frac[col_na_frac > MAX_COL_NA_RATIO].index.tolist()
if drop_cols:
    print(f"Dropping {len(drop_cols)} columns with > {MAX_COL_NA_RATIO*100:.0f}% missing: {drop_cols[:10]}{'...' if len(drop_cols)>10 else ''}")
df = df.drop(columns=drop_cols)
feature_cols = [c for c in feature_cols if c not in drop_cols]

# 2) Drop rows with too many NaNs across features
row_na_frac = df[feature_cols].isna().mean(axis=1)
rows_before = len(df)
df = df.loc[row_na_frac <= MAX_ROW_NA_RATIO].copy()
print(f"Dropped {rows_before - len(df)} rows with > {MAX_ROW_NA_RATIO*100:.0f}% missing features")

# 3) Winsorize each feature (helps extreme outliers)
print("Winsorizing features at percentiles:", WINSOR_PCT)
for col in feature_cols:
    df[col] = winsorize_series(df[col], lower=WINSOR_PCT[0], upper=WINSOR_PCT[1])

# 4) Impute remaining missing feature values with median
print("Imputing remaining NaNs with median")
imputer = SimpleImputer(strategy="median")
df[feature_cols] = imputer.fit_transform(df[feature_cols])

# --------------------------
# Scaling
# --------------------------
print("Fitting RobustScaler on features")
scaler = RobustScaler()
df_scaled = df.copy()
df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

# --------------------------
# Train/test time split
# --------------------------
print("Creating time-based train/test split")
df_scaled = df_scaled.sort_values("date")
unique_dates = df_scaled["date"].sort_values().unique()
n_train_dates = int(len(unique_dates) * TRAIN_TEST_SPLIT_RATIO)
train_last_date = unique_dates[n_train_dates - 1]

df_scaled["split"] = np.where(df_scaled["date"] <= train_last_date, "train", "test")
print(f"Train dates up to (inclusive): {train_last_date} -> Train rows: {df_scaled['split'].value_counts().to_dict().get('train',0)}; Test rows: {df_scaled['split'].value_counts().to_dict().get('test',0)}")

# --------------------------
# Save outputs
# --------------------------
OUT_CSV = OUTPUT_DIR / "training_data.csv"
OUT_SCALER = ARTIFACTS_DIR / "scaler.pkl"
OUT_IMPUTER = ARTIFACTS_DIR / "imputer.pkl"
OUT_FEATURES = ARTIFACTS_DIR / "feature_list.json"

print("Saving training data to:", OUT_CSV)
df_scaled.to_csv(OUT_CSV, index=False)

print("Saving scaler and imputer to artifacts")
joblib.dump(scaler, OUT_SCALER)
joblib.dump(imputer, OUT_IMPUTER)

feature_meta = {
    "feature_columns": feature_cols,
    "horizon_days": HORIZON_DAYS,
    "train_test_split_date": str(train_last_date),
    "winsor_pct": WINSOR_PCT
}
with open(OUT_FEATURES, "w") as f:
    json.dump(feature_meta, f, indent=2)

print("Done. Artifacts saved to:", ARTIFACTS_DIR)
