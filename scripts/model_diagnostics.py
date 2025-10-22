"""
Stage 3.5 — Model Diagnostics & Explainability
------------------------------------------------
Analyzes model interpretability via feature importance,
SHAP values, residual analysis, and temporal stability.

Author: Suraj Prakash
"""

import os
import joblib
import shap
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

# ===========================================================
# CONFIG
# ===========================================================
PROJECT_ROOT = Path(r"D:\Portfolio Optimzation project")
DATA_FILE = PROJECT_ROOT / "data" / "training_data.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_FILE = ARTIFACTS_DIR / "metrics_summary.csv"
FEATURES_FILE = ARTIFACTS_DIR / "selected_features.json"
DIAG_DIR = ARTIFACTS_DIR / "model_diagnostics"

DIAG_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================
# LOAD DATA & ARTIFACTS
# ===========================================================
data = pd.read_csv(DATA_FILE)
with open(FEATURES_FILE, "r") as f:
    features = json.load(f)["selected_features"]

metrics = pd.read_csv(METRICS_FILE)
best_model_name = metrics.groupby("model", as_index=False)["ic"].mean().sort_values("ic", ascending=False).iloc[0]["model"]
print(f" Best model for diagnostics: {best_model_name}")

model_files = sorted(MODELS_DIR.glob(f"{best_model_name}_fold*.pkl"))
model_path = model_files[-1]
model = joblib.load(model_path)
print(f" Loaded model: {model_path.name}")

# ===========================================================
# FEATURE IMPORTANCE
# ===========================================================
importance_df = pd.DataFrame(columns=["feature", "importance"])

if hasattr(model, "coef_"):  # Linear Models
    importance_df["feature"] = features
    importance_df["importance"] = np.abs(model.coef_)
elif hasattr(model, "feature_importances_"):  # Tree-based Models
    importance_df["feature"] = features
    importance_df["importance"] = model.feature_importances_
else:
    print(" Model does not expose direct feature importances.")
    importance_df = None

if importance_df is not None:
    importance_df = importance_df.sort_values("importance", ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="importance", y="feature", palette="coolwarm")
    plt.title(f"Feature Importance ({best_model_name.upper()})")
    plt.tight_layout()
    plt.savefig(DIAG_DIR / f"feature_importance_{best_model_name}.png", dpi=300)
    plt.close()
    print(" Feature importance plot saved.")

# ===========================================================
# SHAP EXPLAINABILITY
# ===========================================================
try:
    print(" Computing SHAP values (this may take time)...")
    sample = data.sample(min(5000, len(data)), random_state=42)
    X = sample[features]
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(DIAG_DIR / f"shap_summary_bar_{best_model_name}.png", dpi=300)
    plt.close()

    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(DIAG_DIR / f"shap_summary_violin_{best_model_name}.png", dpi=300)
    plt.close()
    print(" SHAP summary plots saved.")
except Exception as e:
    print(" SHAP analysis skipped:", e)

# ===========================================================
# PREDICTION ERROR ANALYSIS
# ===========================================================
print(" Analyzing residuals ...")
target = "target_return"
if target in data.columns:
    X_full = data[features]
    y_full = data[target]
    preds = model.predict(X_full)
    residuals = y_full - preds

    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, bins=50, kde=True)
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(DIAG_DIR / "residual_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_full, y=preds, alpha=0.3)
    plt.title("Actual vs Predicted Returns")
    plt.xlabel("Actual Returns")
    plt.ylabel("Predicted Returns")
    plt.tight_layout()
    plt.savefig(DIAG_DIR / "actual_vs_predicted.png", dpi=300)
    plt.close()
    print(" Residual analysis plots saved.")

# ===========================================================
# TEMPORAL STABILITY
# ===========================================================
print(" Checking IC stability over folds ...")
plt.figure(figsize=(8, 4))
sns.lineplot(data=metrics, x="fold", y="ic", hue="model", marker="o")
plt.title("IC Stability Over Time (By Fold)")
plt.tight_layout()
plt.savefig(DIAG_DIR / "ic_stability.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 4))
sns.lineplot(data=metrics, x="fold", y="hit_ratio", hue="model", marker="o")
plt.title("Hit Ratio Stability Over Time")
plt.tight_layout()
plt.savefig(DIAG_DIR / "hitratio_stability.png", dpi=300)
plt.close()

print(" Model diagnostics complete.")
print(f" All plots saved to: {DIAG_DIR}")
