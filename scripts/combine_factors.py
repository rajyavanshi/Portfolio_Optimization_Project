"""
combine_factors.py  |  Stage 2C – Factor Matrix Combination
Merges price-based and fundamental-based factor files into a single
combined_factor_matrix.csv and produces correlation heatmap.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- PATHS ----------
BASE_DIR = r"D:\Portfolio Optimzation project"
PRICE_PATH = os.path.join(BASE_DIR, "results", "factors", "price_factors.csv")
FUND_PATH = os.path.join(BASE_DIR, "results", "factors", "fundamental_factors_processed.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "factors", "combined_factor_matrix.csv")
HEATMAP_PATH = os.path.join(BASE_DIR, "reports", "plots", "factor_heatmap.png")

# ---------- MAIN PIPELINE ----------
def combine_factors():
    print("📥 Loading price and fundamental factor files...")
    price_df = pd.read_csv(PRICE_PATH, parse_dates=["date"])
    fund_df = pd.read_csv(FUND_PATH, parse_dates=["date"])

    print("🔗 Merging on [date, ticker]...")
    merged = pd.merge(price_df, fund_df, on=["date", "ticker"], how="inner")

    # Remove duplicates or missing data
    merged.dropna(inplace=True)
    print(f"✅ Combined dataset shape: {merged.shape}")

    # Save combined factor matrix
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"💾 Saved combined factor matrix → {OUTPUT_PATH}")

    # Compute correlations (cross-sectional across factors)
    factor_cols = [
        "momentum", "volatility", "liquidity", "size_proxy",
        "value_composite", "quality_composite"
    ]

    print("📊 Generating factor correlation heatmap...")
    corr = merged[factor_cols].corr()

    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Factor Correlation Heatmap")
    os.makedirs(os.path.dirname(HEATMAP_PATH), exist_ok=True)
    plt.tight_layout()
    plt.savefig(HEATMAP_PATH)
    plt.close()
    print(f"✅ Heatmap saved → {HEATMAP_PATH}")

    print("🎯 Stage 2C completed successfully!")

if __name__ == "__main__":
    combine_factors()
