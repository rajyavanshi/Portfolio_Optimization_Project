"""
factor_builder.py  |  Stage 2A – Price-Based Factor Construction
Reads clean price data from processed_data/, computes technical factors,
and saves results to results/factors/price_factors.csv
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# ---------- PATHS ----------
BASE_DIR = r"D:\Portfolio Optimzation project"
PROCESSED_PATH = os.path.join(BASE_DIR, "processed_data", "merged_prices.csv")
RESULT_PATH = os.path.join(BASE_DIR, "results", "factors", "price_factors.csv")

# ---------- FACTOR FUNCTIONS ----------
def compute_returns(df):
    """Daily percentage returns."""
    df["returns"] = df.groupby("ticker")["close"].pct_change()
    return df

def compute_momentum(df, lookback=126):
    """Rolling cumulative return over the lookback window."""
    df["momentum"] = (
        df.groupby("ticker")["close"].pct_change(lookback)
    )
    return df

def compute_volatility(df, window=60):
    """Rolling standard deviation of daily returns."""
    df["volatility"] = (
        df.groupby("ticker")["returns"]
        .rolling(window)
        .std()
        .reset_index(level=0, drop=True)
    )
    return df

def compute_liquidity(df, window=60):
    """Rolling mean of volume."""
    df["liquidity"] = (
        df.groupby("ticker")["volume"]
        .rolling(window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df

def compute_size_proxy(df):
    """Log of market activity proxy (price × volume)."""
    df["size_proxy"] = np.log(df["close"] * df["volume"] + 1)
    return df

def normalize_factors(df, cols):
    """Cross-sectional z-score normalization (per date)."""
    for col in cols:
        df[col] = df.groupby("date")[col].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0)
        )
    return df

# ---------- PIPELINE ----------
def build_price_factors():
    print("📥 Loading merged price data...")
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    df.sort_values(["ticker", "date"], inplace=True)

    print("⚙️ Computing factors...")
    df = compute_returns(df)
    df = compute_momentum(df)
    df = compute_volatility(df)
    df = compute_liquidity(df)
    df = compute_size_proxy(df)

    # Drop NaN rows from lookback computations
    df.dropna(subset=["momentum", "volatility", "liquidity"], inplace=True)

    # Normalize factors per date
    factor_cols = ["momentum", "volatility", "liquidity", "size_proxy"]
    df = normalize_factors(df, factor_cols)

    # Keep essential columns
    final_cols = ["date", "ticker"] + factor_cols
    factor_df = df[final_cols].copy()

    print("💾 Saving to:", RESULT_PATH)
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    factor_df.to_csv(RESULT_PATH, index=False)
    print("✅ Price-based factor matrix saved successfully!")

if __name__ == "__main__":
    build_price_factors()
