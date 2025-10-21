# D:\Portfolio_Project\scripts\data_cleaner.py
import os
import pandas as pd
from tqdm import tqdm
from scripts.data_loader import load_csv

def clean_and_resample_single(file_path, processed_dir):
    """Clean and resample a single raw CSV file."""
    symbol = os.path.basename(file_path).replace(".csv", "")
    df = load_csv(file_path)
    df = df.sort_values('date').drop_duplicates(subset='date')
    df = df.dropna(subset=['open', 'close', 'volume'])

    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(int)

    # Resample to daily frequency
    df = df.set_index('date').resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()

    os.makedirs(processed_dir, exist_ok=True)
    save_path = os.path.join(processed_dir, f"{symbol}_daily.csv")
    df.to_csv(save_path, index=False)

    print(f" Processed {symbol} → {save_path}")
    return df


def process_all_files(raw_dir, processed_dir):
    """Loop through all CSVs in raw_dir and process them."""
    os.makedirs(processed_dir, exist_ok=True)
    files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]

    for file in tqdm(files, desc="Processing all files"):
        file_path = os.path.join(raw_dir, file)
        try:
            clean_and_resample_single(file_path, processed_dir)
        except Exception as e:
            print(f" Error processing {file}: {e}")
