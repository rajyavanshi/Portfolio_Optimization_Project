"""
factor_builder_fundamental.py | Stage 2B – Fundamental Factor Construction
Reads fundamental data from results/factors/, computes Value & Quality factors,
and saves normalized output to results/factors/fundamental_factors_processed.csv
"""

import os
import pandas as pd
import numpy as np

# ---------- PATHS ----------
BASE_DIR = r"D:\Portfolio Optimzation project"
INPUT_PATH = os.path.join(BASE_DIR, "results", "factors", "fundamental_factors.csv")        # raw dummy/real fundamentals
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "factors", "fundamental_factors_processed.csv")  # processed normalized version

# ---------- FACTOR FUNCTIONS ----------
def compute_value_factors(df):
    """Compute inverse value metrics: cheap = high score."""
    df["inv_pe"] = 1 / df["PE"].replace(0, np.nan)
    df["inv_pb"] = 1 / df["PB"].replace(0, np.nan)
    df["value_composite"] = df[["inv_pe", "inv_pb"]].mean(axis=1)
    return df

def compute_quality_factors(df):
    """Compute quality metrics (profitability and stability)."""
    df["quality_composite"] = (
        df[["ROE", "ROA"]].mean(axis=1) - df["Debt_to_Equity"]
    )
    return df

def normalize_factors(df, cols):
    """Z-score normalization per date (cross-sectionally)."""
    for col in cols:
        df[col] = df.groupby("date")[col].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0)
        )
    return df

# ---------- PIPELINE ----------
def build_fundamental_factors():
    print(" Loading fundamental data...")
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    df.sort_values(["ticker", "date"], inplace=True)

    # Ensure numeric columns
    num_cols = ["PE", "PB", "ROE", "ROA", "Debt_to_Equity"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    print(" Computing value and quality factors...")
    df = compute_value_factors(df)
    df = compute_quality_factors(df)

    # Normalize
    factor_cols = ["value_composite", "quality_composite"]
    df = normalize_factors(df, factor_cols)

    # Select relevant columns
    final_cols = ["date", "ticker"] + factor_cols
    factor_df = df[final_cols].copy()

    print(" Saving processed factors to:", OUTPUT_PATH)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    factor_df.to_csv(OUTPUT_PATH, index=False)
    print(" Fundamental-based factors saved successfully!")

if __name__ == "__main__":
    build_fundamental_factors()
