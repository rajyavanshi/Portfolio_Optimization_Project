"""
scripts/feature_selector.py
Step 3.2: Feature Selection & Importance Ranking
Author: Suraj Prakash
---------------------------------------------------------
This script evaluates factor predictive strength and redundancy.
It computes:
 - Spearman Information Coefficient (IC)
 - Mutual Information (MI)
 - Variance Inflation Factor (VIF)
and selects top factors based on a combined score.

Outputs:
 - artifacts/feature_rankings.csv
 - artifacts/selected_features.csv
 - artifacts/selected_features.json
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import spearmanr
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\Portfolio Optimzation project")
TRAINING_FILE = PROJECT_ROOT / "data" / "training_data.csv"
FEATURE_META_FILE = PROJECT_ROOT / "artifacts" / "feature_list.json"
OUTPUT_DIR = PROJECT_ROOT / "artifacts"

# Selection Parameters
IC_THRESHOLD = 0.0        # No hard filter on IC (financial ICs are small)
TOP_N = 4                 # Keep all 4 since dataset small
VIF_THRESHOLD = 10.0      # Used for penalizing score
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print(" Loading training data ...")
df = pd.read_csv(TRAINING_FILE, parse_dates=["date"])
with open(FEATURE_META_FILE, "r") as f:
    feature_meta = json.load(f)

feature_cols = feature_meta["feature_columns"]
target_col = "target_return"

# Use only training part (avoid leakage)
train_df = df[df["split"] == "train"].copy()
X = train_df[feature_cols].copy()
y = train_df[target_col].copy()

print(f"Total features loaded: {len(feature_cols)}")

# Drop constant columns (zero variance)
zero_var_cols = X.columns[X.std() <= 1e-8]
if len(zero_var_cols) > 0:
    print(f" Dropping constant features: {list(zero_var_cols)}")
    X = X.drop(columns=zero_var_cols)

# Fill missing if any (safety)
X = X.fillna(X.median())
y = y.fillna(0)

# ============================================================
# 1️. MULTICOLLINEARITY CHECK (VIF)
# ============================================================

print(" Checking multicollinearity ...")
vif_data = pd.DataFrame({
    "feature": X.columns,
    "VIF": [variance_inflation_factor(X.values, i)
            for i in range(X.shape[1])]
})

# ============================================================
# 2️. INFORMATION COEFFICIENT (IC)
# ============================================================

print(" Calculating Spearman Information Coefficients ...")
ic_values = {}
for f in X.columns:
    ic, _ = spearmanr(X[f], y)
    ic_values[f] = ic

ic_df = pd.DataFrame({
    "feature": list(ic_values.keys()),
    "IC": list(ic_values.values())
})
ic_df["abs_IC"] = ic_df["IC"].abs()

# ============================================================
# 3️. MUTUAL INFORMATION (Nonlinear predictive strength)
# ============================================================

print(" Computing Mutual Information ...")
mi_scores = mutual_info_regression(X, y, random_state=42)
mi_df = pd.DataFrame({
    "feature": X.columns,
    "MI": mi_scores
})

# ============================================================
# 4️. COMBINE METRICS & SCORE FEATURES
# ============================================================

combined = ic_df.merge(mi_df, on="feature").merge(vif_data, on="feature")

# Handle NaNs if any
combined = combined.fillna(0)

# Compute composite score (higher = better)
combined["score"] = (
    0.6 * combined["abs_IC"].rank(pct=True) +
    0.4 * combined["MI"].rank(pct=True)
) / (1 + (combined["VIF"] / VIF_THRESHOLD))

combined = combined.sort_values("score", ascending=False).reset_index(drop=True)

# Always keep at least 3 features to avoid over-pruning
selected = combined.head(max(3, min(TOP_N, len(combined)))).reset_index(drop=True)

# ============================================================
# 5. SAVE RESULTS
# ============================================================

combined.to_csv(OUTPUT_DIR / "feature_rankings.csv", index=False)
selected.to_csv(OUTPUT_DIR / "selected_features.csv", index=False)

selected_features = selected["feature"].tolist()
with open(OUTPUT_DIR / "selected_features.json", "w") as f:
    json.dump({"selected_features": selected_features}, f, indent=2)

print(" Feature selection completed successfully.")
print(f" Saved results to: {OUTPUT_DIR}")
print("\nTop Features:")
print(selected[["feature", "IC", "MI", "VIF", "score"]])
