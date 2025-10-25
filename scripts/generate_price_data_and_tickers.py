import pandas as pd
from pathlib import Path
import os

# =========================
#  CONFIG
# =========================
PROJECT_ROOT = Path(r"D:\Portfolio Optimzation project")
MERGED_FILE = PROJECT_ROOT / "processed_data" / "merged_prices.csv"
DATA_DIR = PROJECT_ROOT / "data"
os.makedirs(DATA_DIR, exist_ok=True)

# =========================
#  STEP 1: LOAD MERGED DATA
# =========================
df = pd.read_csv(MERGED_FILE)
df.columns = df.columns.str.lower()

# sanity check
required_cols = {'date', 'close', 'ticker'}
if not required_cols.issubset(df.columns):
    raise ValueError(f" merged_prices.csv must have columns {required_cols}")

print(f" Loaded merged_prices.csv | Shape: {df.shape} | Columns: {list(df.columns)}")

# =========================
#  STEP 2: CLEAN AND CONVERT
# =========================
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date', 'close', 'ticker'])

# standardize ticker names
df['ticker'] = df['ticker'].astype(str).str.upper().str.replace('.NS', '', regex=False)

# keep only what we need
price_data = df[['date', 'ticker', 'close']].rename(columns={'close': 'close_price'})
price_data = price_data.sort_values(['date', 'ticker']).reset_index(drop=True)

# =========================
#  STEP 3: SAVE price_data.csv
# =========================
price_data.to_csv(DATA_DIR / "price_data.csv", index=False)
print(f" Saved: {DATA_DIR / 'price_data.csv'} | Shape: {price_data.shape}")

# =========================
#  STEP 4: CREATE tickers.csv
# =========================
tickers = (
    price_data[['ticker']]
    .drop_duplicates()
    .assign(sector='Unknown', name=lambda x: x['ticker'])
)
tickers.to_csv(DATA_DIR / "tickers.csv", index=False)
print(f" Saved: {DATA_DIR / 'tickers.csv'} | {len(tickers)} tickers found")
