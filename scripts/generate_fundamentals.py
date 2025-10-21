"""
generate_fundamentals.py
Creates a synthetic fundamental_factors.csv file in processed_data/
so Stage 2B can run without external data sources.
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = r"D:\Portfolio Optimzation project"
MERGED_PATH = os.path.join(BASE_DIR, "processed_data", "merged_prices.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "factors", "fundamental_factors.csv")

def generate_fundamentals():
    print("📥 Reading tickers from merged_prices.csv ...")
    df = pd.read_csv(MERGED_PATH, usecols=["date", "ticker"])
    df["date"] = pd.to_datetime(df["date"])
    latest_dates = df["date"].unique()[-4:]  # last 4 quarters

    # Generate dummy fundamentals
    fundamentals = []
    tickers = df["ticker"].unique()
    for d in latest_dates:
        for t in tickers:
            fundamentals.append({
                "date": d,
                "ticker": t,
                "PE": np.random.uniform(5, 60),
                "PB": np.random.uniform(0.5, 15),
                "ROE": np.random.uniform(0.05, 0.5),
                "ROA": np.random.uniform(0.02, 0.3),
                "Debt_to_Equity": np.random.uniform(0.0, 1.5)
            })
    fundamentals_df = pd.DataFrame(fundamentals)
    fundamentals_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved dummy fundamentals to: {OUTPUT_PATH}")
    print(f"📊 Shape: {fundamentals_df.shape} | Unique tickers: {fundamentals_df['ticker'].nunique()}")

if __name__ == "__main__":
    generate_fundamentals()
