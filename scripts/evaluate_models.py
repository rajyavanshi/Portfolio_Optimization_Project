"""
scripts/evaluate_models.py
Stage 3.4 - Model Evaluation & Interpretation

Reads metrics_summary.csv (output from Stage 3.3) and generates:
 - Aggregated performance summary
 - Plots: IC trend, Hit ratio trend, Model comparison
 - Saves all in artifacts/model_diagnostics/

Author: Suraj Prakash
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------------
# CONFIG
# ------------------------------------
PROJECT_ROOT = Path(r"D:\Portfolio Optimzation project")
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
METRICS_FILE = ARTIFACTS_DIR / "metrics_summary.csv"
DIAG_DIR = ARTIFACTS_DIR / "model_diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------
# LOAD DATA
# ------------------------------------
print(" Loading metrics summary...")
df = pd.read_csv(METRICS_FILE)
df = df[df["model"] != "ensemble_ic_weighted"]  # keep ensemble separate
print(f" Loaded {len(df)} rows for {df['model'].nunique()} models")

# ------------------------------------
# AGGREGATE STATS
# ------------------------------------
summary = (
    df.groupby("model")
    .agg({
        "mse": ["mean", "std"],
        "ic": ["mean", "std"],
        "hit_ratio": ["mean", "std"],
        "fold": "count"
    })
)
summary.columns = ["_".join(col) for col in summary.columns]
summary = summary.reset_index()
summary["ic_rank"] = summary["ic_mean"].rank(ascending=False)
summary = summary.sort_values("ic_mean", ascending=False)

# save table
summary_out = DIAG_DIR / "summary_table.csv"
summary.to_csv(summary_out, index=False)
print(f" Saved summary table to {summary_out}")

# ------------------------------------
# PLOTS
# ------------------------------------
sns.set(style="whitegrid", context="talk")

# 1. IC Trend per fold
plt.figure(figsize=(10,6))
sns.lineplot(data=df, x="fold", y="ic", hue="model", marker="o")
plt.title("Information Coefficient (IC) per Fold")
plt.xlabel("Fold")
plt.ylabel("IC (Spearman)")
plt.legend(title="Model")
plt.tight_layout()
plt.savefig(DIAG_DIR / "ic_trend.png", dpi=300)
plt.close()

# 2. Hit Ratio per fold
plt.figure(figsize=(10,6))
sns.lineplot(data=df, x="fold", y="hit_ratio", hue="model", marker="o")
plt.title("Hit Ratio per Fold")
plt.xlabel("Fold")
plt.ylabel("Hit Ratio")
plt.legend(title="Model")
plt.tight_layout()
plt.savefig(DIAG_DIR / "hitratio_trend.png", dpi=300)
plt.close()

# 3️. Average IC comparison
plt.figure(figsize=(9,5))
sns.barplot(data=summary, x="ic_mean", y="model", palette="viridis")
plt.title("Average Information Coefficient by Model")
plt.xlabel("Mean IC")
plt.ylabel("Model")
plt.tight_layout()
plt.savefig(DIAG_DIR / "model_comparison_bar.png", dpi=300)
plt.close()

# 4️. Average Hit Ratio comparison
plt.figure(figsize=(9,5))
sns.barplot(data=summary, x="hit_ratio_mean", y="model", palette="mako")
plt.title("Average Hit Ratio by Model")
plt.xlabel("Mean Hit Ratio")
plt.ylabel("Model")
plt.tight_layout()
plt.savefig(DIAG_DIR / "hitratio_comparison_bar.png", dpi=300)
plt.close()

# ------------------------------------
# PRINT INSIGHTS
# ------------------------------------
print("\n Model Performance Summary:")
print(summary[["model", "mse_mean", "ic_mean", "hit_ratio_mean", "ic_rank"]])

best_model = summary.iloc[0]["model"]
best_ic = summary.iloc[0]["ic_mean"]
print(f"\n Best model overall: {best_model} (avg IC = {best_ic:.5f})")
print(f" Plots and summary saved to: {DIAG_DIR}")
