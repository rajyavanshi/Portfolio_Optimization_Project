# ==============================
#  CONFIGURATION FILE
# Portfolio Optimization Project
# ==============================

from pathlib import Path
import os

# ---- Project Root ----
PROJECT_ROOT = Path(r"D:\Portfolio Optimzation project")

# ---- Data Paths ----
DATA_DIR = PROJECT_ROOT / "data"
PRICE_FILE = DATA_DIR / "price_data.csv"         # historical prices
TICKER_FILE = DATA_DIR / "tickers.csv"           # optional

# ---- Artifacts (from ML Models) ----
ALPHA_FILE = PROJECT_ROOT / "artifacts" / "alpha_predictions.csv"   # predicted returns
VOL_FILE = PROJECT_ROOT / "results" / "volatility_matrix.csv"       # vol/cov matrix
FACTORS_FILE = PROJECT_ROOT / "results" / "factors" / "price_factors.csv"  # optional

# ---- Optimization Outputs ----
OUT_DIR = PROJECT_ROOT / "results" / "portfolios"
OPT_WEIGHTS_DIR = OUT_DIR / "optimized_weights"

# ---- Rebalancing / Backtesting Outputs ----
BACKTEST_DIR = PROJECT_ROOT / "results" / "backtest"
os.makedirs(BACKTEST_DIR, exist_ok=True)

# ---- Parameters ----
REB_FREQ = 'monthly'            # rebalancing frequency
INITIAL_CAPITAL = 1_000_000     # starting capital
TRANSACTION_COST = 0.001        # 0.1% per trade
RISK_FREE_RATE = 0.05 / 252     # daily risk-free rate (5% annual)

# ---- Column Names ----
DATE_COL = "date"
TICKER_COL = "ticker"
RETURN_COL = "predicted_return"

# ---- Plot Settings ----
PLOT_DPI = 120
