import pandas as pd
import numpy as np
from pathlib import Path
import os

# --- CONFIG ---
PROJECT_ROOT = Path(r"D:\Portfolio Optimzation project")
ALPHA_FILE = PROJECT_ROOT / "artifacts" / "alpha_predictions.csv"
VOL_FILE = PROJECT_ROOT / "results" / "volatility_matrix.csv"
OUT_DIR = PROJECT_ROOT / "results" / "portfolios"
os.makedirs(OUT_DIR, exist_ok=True)

REB_FREQ = 'M'  # 'M' for monthly, 'W' for weekly

# --- LOAD INPUTS ---
alpha_df = pd.read_csv(ALPHA_FILE, parse_dates=['date'])
vol_df = pd.read_csv(VOL_FILE, parse_dates=['date'])

#  Drop timezone info to make comparison valid
alpha_df['date'] = pd.to_datetime(alpha_df['date']).dt.tz_localize(None)
vol_df['date'] = pd.to_datetime(vol_df['date']).dt.tz_localize(None)

alpha_df.columns = alpha_df.columns.str.lower()
vol_df.columns = vol_df.columns.str.lower()


# expected columns check
assert {'date', 'ticker', 'predicted_return'}.issubset(alpha_df.columns)
assert {'date', 'ticker', 'volatility'}.issubset(vol_df.columns)

# --- Expand static alpha across vol dates (for testing only) ---
if alpha_df['date'].nunique() == 1:
    unique_vol_dates = sorted(vol_df['date'].unique())
    base_alpha = alpha_df.copy()
    all_alpha = []
    for d in unique_vol_dates:
        tmp = base_alpha.copy()
        tmp['date'] = d
        all_alpha.append(tmp)
    alpha_df = pd.concat(all_alpha, ignore_index=True)
    print(f" Simulated dynamic alpha predictions for {len(unique_vol_dates)} dates.")


# merge alpha + vol
df = alpha_df.merge(vol_df, on=['date', 'ticker'], how='inner').dropna(subset=['predicted_return', 'volatility'])

# identify rebalance dates
rebalance_dates = df['date'].dt.to_period(REB_FREQ).drop_duplicates().dt.to_timestamp()
print(f" Total rebalance periods: {len(rebalance_dates)}")

# --- OPTIMIZATION HELPERS ---
def mean_variance_optimize(sub_df):
    """
    Performs mean-variance optimization for given snapshot.
    Returns DataFrame with columns [ticker, w_mvo_best_sharpe, w_minvar].
    """
    tickers = sub_df['ticker'].values
    mu = sub_df['predicted_return'].values
    sigma = sub_df['volatility'].values

    cov = np.diag(sigma ** 2)
    inv_cov = np.linalg.pinv(cov)

    ones = np.ones(len(tickers))
    A = ones @ inv_cov @ ones
    B = ones @ inv_cov @ mu
    C = mu @ inv_cov @ mu

    lam_sharpe = (C - B) / (A * C - B ** 2 + 1e-8)
    lam_minvar = (B) / (A * C - B ** 2 + 1e-8)

    # Min variance weights
    w_minvar = (inv_cov @ ones) / A
    # Max Sharpe (risk-return efficient frontier)
    w_mvo_best_sharpe = (inv_cov @ (mu - lam_sharpe * ones))
    w_mvo_best_sharpe /= np.sum(np.abs(w_mvo_best_sharpe))

    w_mvo_best_sharpe = np.clip(w_mvo_best_sharpe, 0, None)
    w_mvo_best_sharpe /= np.sum(w_mvo_best_sharpe)

    w_minvar = np.clip(w_minvar, 0, None)
    w_minvar /= np.sum(w_minvar)

    return pd.DataFrame({
        'ticker': tickers,
        'predicted_return': mu,
        'volatility': sigma,
        'w_mvo_best_sharpe': w_mvo_best_sharpe,
        'w_minvar': w_minvar
    })

# --- LOOP OVER REBALANCE DATES ---
for rdate in rebalance_dates:
    sub = df[df['date'] <= rdate]
    latest = sub[sub['date'] == sub['date'].max()]
    if latest.empty:
        continue

    res = mean_variance_optimize(latest)
    res['date'] = rdate

    out_file = OUT_DIR / f"optimized_weights_{rdate.date()}.csv"
    res.to_csv(out_file, index=False)
    print(f" Saved {out_file.name} | {len(res)} tickers")

print(" Dynamic weights generated successfully!")
