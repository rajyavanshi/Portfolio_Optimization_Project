"""
merge_processed_data.py
Merges all individual processed stock CSVs into a single merged_prices.csv
Reads from processed_data/, writes to processed_data/merged_prices.csv
"""

import os
import pandas as pd
from tqdm import tqdm

# ---------- PATHS ----------
BASE_DIR = r"D:\Portfolio Optimzation project"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "merged_prices.csv")

# ---------- MERGE SCRIPT ----------
def merge_processed_files():
    print("📂 Reading processed_data directory...")
    files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".csv") and f != "fundamental_factors.csv"]

    all_data = []

    for file in tqdm(files, desc="Merging files"):
        file_path = os.path.join(PROCESSED_DIR, file)
        try:
            df = pd.read_csv(file_path)
            ticker = os.path.splitext(file)[0]  # extract filename without .csv
            df["ticker"] = ticker.upper()
            all_data.append(df)
        except Exception as e:
            print(f"⚠️ Skipped {file} due to error: {e}")

    # Combine all dataframes
    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df = merged_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Save output
    merged_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Merged dataset saved to: {OUTPUT_PATH}")
    print(f"📊 Total rows: {len(merged_df):,} | Unique tickers: {merged_df['ticker'].nunique()}")

if __name__ == "__main__":
    merge_processed_files()
