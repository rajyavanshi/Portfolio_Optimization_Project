"""
scripts/generate_rolling_alpha.py

Generate rolling/monthly alpha predictions (next-month returns) using a LASSO model.
Saves per-period alpha CSV and a merged alpha_predictions.csv in artifacts/.

Assumes:
- data/price_data.csv exists with columns: date, ticker, close_price
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings

warnings.filterwarnings("ignore")

# ===== CONFIG =====
PROJECT_ROOT = Path(r"D:\Portfolio Optimzation project")
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUT_DIR = ARTIFACTS_DIR
os.makedirs(OUT_DIR, exist_ok=True)

PRICE_FILE = DATA_DIR / "price_data.csv"

# modeling params
LOOKBACK_DAYS = 252            # training window (in trading days)
PRED_HORIZON = 21              # predict 21 trading days ahead (approx 1 month)
REB_FREQ = 'M'                 # 'M' = month-end, 'W' = weekly, 'Q' = quarterly
MIN_TICKERS = 50               # skip periods with too few tickers
LASSO_ALPHA = 1e-3             # regularization strength (tune if needed)

# ===== HELPERS =====
def ensure_dates_tz_naive(df, col='date'):
    df[col] = pd.to_datetime(df[col])
    # drop tz info if present
    try:
        df[col] = df[col].dt.tz_localize(None)
    except Exception:
        # if already naive, this raises; ignore
        pass
    return df

def compute_features_from_prices(price_df):
    """
    price_df: long form DataFrame with columns date, ticker, close_price
    returns: features_df with columns:
      date, ticker, close, r1, r5, r21, vol21, mom21
    """
    df = price_df.copy()
    df = ensure_dates_tz_naive(df, 'date')
    # pivot to wide for rolling ops
    wide = df.pivot(index='date', columns='ticker', values='close_price').sort_index()
    # compute returns and features
    r1 = wide.pct_change(1)
    r5 = wide.pct_change(5)
    r21 = wide.pct_change(21)
    vol21 = wide.pct_change().rolling(21).std()  # daily vol
    mom21 = r21  # momentum over 21 days

    # melt features back to long
    feats = []
    for name, mat in [('r1', r1), ('r5', r5), ('r21', r21), ('vol21', vol21), ('mom21', mom21)]:
        temp = mat.stack().rename(name).reset_index()
        temp.columns = ['date', 'ticker', name]
        feats.append(temp)

    features_df = feats[0]
    for other in feats[1:]:
        features_df = features_df.merge(other, on=['date','ticker'], how='outer')

    # include last price
    last_price = wide.stack().rename('close').reset_index()
    last_price.columns = ['date','ticker','close_price']
    features_df = features_df.merge(last_price, on=['date','ticker'], how='left')

    # drop rows with insufficient data
    features_df = features_df.dropna(subset=['r1','r5','r21','vol21','close_price'])
    return features_df

def compute_targets_from_prices(price_df, horizon=PRED_HORIZON):
    """
    target is forward horizon pct change: (P_{t+h} / P_t) - 1
    returns DataFrame with columns: date, ticker, target
    """
    df = price_df.copy()
    df = ensure_dates_tz_naive(df, 'date')
    wide = df.pivot(index='date', columns='ticker', values='close_price').sort_index()
    forward = wide.shift(-horizon) / wide - 1.0
    target_df = forward.stack().rename('target').reset_index()
    target_df.columns = ['date', 'ticker', 'target']
    # target at date t = return from t to t+horizon
    return target_df

# ===== LOAD PRICES =====
print("Loading price data...")
price_df = pd.read_csv(PRICE_FILE, parse_dates=['date'])
price_df = ensure_dates_tz_naive(price_df, 'date')
price_df = price_df.rename(columns={'close_price':'close_price'})
print("Price rows:", len(price_df), "unique tickers:", price_df['ticker'].nunique())

# ===== BUILD FEATURES & TARGETS =====
print("Computing features...")
features_df = compute_features_from_prices(price_df)   # date,ticker,r1,r5,r21,vol21,mom21,close_price
print("Computed features rows:", len(features_df))

print("Computing targets (forward returns)...")
targets_df = compute_targets_from_prices(price_df, horizon=PRED_HORIZON)
print("Computed targets rows:", len(targets_df))

# merge features + targets (targets will be NaN at tail because forward shift)
data_df = features_df.merge(targets_df, on=['date','ticker'], how='left')
# drop rows where target is null (we can't train/predict where forward price missing)
data_df = data_df.dropna(subset=['target'])
print("Merged dataset rows (features with valid targets):", len(data_df))

# ===== Determine rebalance dates (period ends with enough history) =====
all_dates = sorted(data_df['date'].unique())
# convert to pandas Period grouping and take last date per period
dates_series = pd.Series(all_dates)
rebalance_period_ends = [
    pd.Timestamp(x) for x in dates_series.groupby(dates_series.dt.to_period(REB_FREQ)).last()
]
# make tz-naive
rebalance_period_ends = [pd.to_datetime(d).tz_localize(None) for d in rebalance_period_ends]
print("Total potential rebalance dates (period ends):", len(rebalance_period_ends))

# Filter rebalance dates to ensure we have LOOKBACK history and horizon ahead
valid_rebalances = []
min_date = min(all_dates)
max_date = max(all_dates)
for d in rebalance_period_ends:
    # need at least LOOKBACK_DAYS of history before d and PRED_HORIZON ahead for targets
    # find index positions in sorted all_dates
    try:
        idx = all_dates.index(pd.Timestamp(d))
    except ValueError:
        # if d not in all_dates (rare), find nearest
        idx = np.searchsorted(all_dates, pd.Timestamp(d))
    if idx >= LOOKBACK_DAYS and (idx + PRED_HORIZON) < len(all_dates):
        valid_rebalances.append(all_dates[idx])
print("Rebalance dates after lookback/horizon filter:", len(valid_rebalances))

# ===== Rolling training & prediction loop =====
saved_files = []
ic_stats = []
for reb_date in valid_rebalances:
    # define train_end = reb_date - 1 (use history up to day before rebalance)
    reb_date = pd.to_datetime(reb_date)
    train_end_pos = all_dates.index(reb_date)
    train_start_pos = train_end_pos - LOOKBACK_DAYS
    train_start_date = all_dates[train_start_pos]
    train_end_date = all_dates[train_end_pos - 1]  # last historical date before reb_date

    # Prepare training set: rows with date between train_start_date and train_end_date (inclusive)
    train_mask = (data_df['date'] >= train_start_date) & (data_df['date'] <= train_end_date)
    train_set = data_df.loc[train_mask].copy()

    # Test set: predict at reb_date for all tickers (features available at reb_date)
    test_mask = (data_df['date'] == reb_date)
    test_set = data_df.loc[test_mask].copy()

    if train_set.empty or test_set.empty:
        # skip if no data
        continue

    # optionally filter tickers with too many NaNs or low liquidity — not done here
    # Prepare X,y
    feature_cols = ['r1','r5','r21','vol21','mom21']
    X_train = train_set[feature_cols].values
    y_train = train_set['target'].values

    X_test = test_set[feature_cols].values

    # train model: pipeline (scaler + lasso)
    model = make_pipeline(StandardScaler(), Lasso(alpha=LASSO_ALPHA, max_iter=5000))
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Training failed at {reb_date.date()}: {e}")
        continue

    # predict
    preds = model.predict(X_test)

    test_set = test_set.copy()
    test_set['predicted_return'] = preds

    # compute realized forward returns for IC diagnostics (target already is forward return)
    # compute IC (Spearman or Pearson)
    try:
        ic_pearson = np.corrcoef(test_set['predicted_return'], test_set['target'])[0,1]
    except Exception:
        ic_pearson = np.nan

    ic_stats.append({'date': reb_date, 'ic_pearson': ic_pearson, 'n': len(test_set)})

    # Save per-period alpha file
    out_df = test_set[['date','ticker','predicted_return']].copy()
    # Ensure date column is string-normalized for filenames
    out_fn = OUT_DIR / f"alpha_predictions_{reb_date.date()}.csv"
    out_df.to_csv(out_fn, index=False)
    saved_files.append(out_fn)
    print(f"Saved alpha predictions for {reb_date.date()} | n={len(out_df)} | IC={ic_pearson:.4f}")

# ===== Merge into single alpha_predictions.csv =====
if saved_files:
    all_alpha = pd.concat([pd.read_csv(f, parse_dates=['date']) for f in saved_files], ignore_index=True)
    all_alpha['date'] = pd.to_datetime(all_alpha['date']).dt.tz_localize(None)
    master_file = OUT_DIR / "alpha_predictions.csv"
    all_alpha.to_csv(master_file, index=False)
    print(f"Saved merged alpha file: {master_file} | rows={len(all_alpha)}")

# ===== Summary IC printout =====
ic_df = pd.DataFrame(ic_stats)
if not ic_df.empty:
    print("IC stats (first 10 rows):")
    print(ic_df.head(10))
else:
    print("No IC stats computed (no saved periods).")

print("Done.")
