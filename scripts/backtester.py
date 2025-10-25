"""
scripts/backtest.py
Runs the backtest using an in-regime optimized portfolio (re-optimizes per regime).
Outputs daily returns, portfolio value, and advanced metrics (Sharpe, Sortino, Calmar, VaR, CVaR).
Saves regime results under results/versions/run_<scenario>.

Author: Suraj Prakash 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# --- Config paths ---
PROJECT_ROOT = Path(r"D:\Portfolio Optimzation project")
PRICE_FILE = PROJECT_ROOT / "data" / "price_data.csv"
# original global optimized weights kept for reference but not required
OPT_WEIGHTS = PROJECT_ROOT / "results" / "portfolios" / "optimized_weights.csv"
RESULTS_BASE = PROJECT_ROOT / "results" / "versions"
RESULTS_BASE.mkdir(parents=True, exist_ok=True)

TRANSACTION_COST = 0.001
INITIAL_CAPITAL = 1.0
TRADING_DAYS_PER_YEAR = 252
RISK_FREE = 0.0  # annual risk-free rate used in ratios (0 by default)

# --- Helper functions ---
def compute_portfolio_return(weights, returns):
    """Compute daily portfolio returns given weight vector and stock returns (DataFrame)."""
    # weights: pd.Series indexed by ticker
    # returns: DataFrame indexed by date with same ticker columns
    return (returns * weights.reindex(returns.columns).fillna(0)).sum(axis=1)


def max_drawdown(series):
    """Max drawdown of a cumulative series (not returns)."""
    cummax = series.cummax()
    dd = (series / cummax) - 1
    return dd.min()


def compute_advanced_metrics(ret_series, risk_free_rate_annual=0.0):
    """
    Compute advanced metrics from a daily return series:
    CAGR, Volatility, Sharpe, Sortino, Calmar, VaR(95%), CVaR(95%)
    """
    if ret_series.empty:
        return {
            "CAGR": np.nan,
            "Volatility": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "Calmar": np.nan,
            "VaR95": np.nan,
            "CVaR95": np.nan,
            "MaxDrawdown": np.nan,
        }

    # daily stats
    mean_daily = ret_series.mean()
    std_daily = ret_series.std(ddof=0)

    # annualize
    cagr = (1 + mean_daily) ** TRADING_DAYS_PER_YEAR - 1
    vol = std_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Sharpe (annual)
    rf_daily = (1 + risk_free_rate_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess_daily = ret_series - rf_daily
    sharpe = (excess_daily.mean() / excess_daily.std(ddof=0)) * np.sqrt(TRADING_DAYS_PER_YEAR) if excess_daily.std(ddof=0) > 0 else np.nan

    # Sortino ratio: downside deviation
    neg_returns = ret_series[ret_series < 0]
    if len(neg_returns) > 0:
        downside_std = neg_returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
        sortino = (cagr - risk_free_rate_annual) / downside_std if downside_std > 0 else np.nan
    else:
        sortino = np.nan

    # Max drawdown (use cumulative returns)
    cum = (1 + ret_series).cumprod()
    mdd = max_drawdown(cum)

    # Calmar ratio
    calmar = (cagr / abs(mdd)) if (mdd != 0 and not np.isnan(mdd)) else np.nan

    # VaR and CVaR (daily)
    var_95 = np.percentile(ret_series.dropna(), 5) if len(ret_series.dropna()) > 0 else np.nan
    cvar_95 = ret_series[ret_series <= var_95].mean() if not np.isnan(var_95) else np.nan

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "VaR95": var_95,
        "CVaR95": cvar_95,
        "MaxDrawdown": mdd,
    }


def optimize_weights_mv(returns_df, shrink=1e-5):
    """
    Simple mean-variance tangency-style optimizer (no constraints).
    Weighted solution: w ∝ inv(C) * (mean - rf)
    Post-process: set negatives to zero and normalize (long-only).
    If optimization fails, fallback to equal weights.
    """
    try:
        rets = returns_df.dropna(axis=0, how="any")
        if rets.empty:
            return pd.Series(dtype=float)

        mu = rets.mean()
        cov = rets.cov()
        # shrink covariance for numerical stability
        cov += np.eye(cov.shape[0]) * shrink
        invcov = np.linalg.inv(cov.values)
        # use excess mean (assuming rf ~ 0 daily)
        target = mu.values
        raw = invcov.dot(target)
        w = pd.Series(raw, index=mu.index)
        # enforce long-only: clip negatives -> 0, renormalize
        w = w.clip(lower=0)
        if w.sum() == 0 or np.isnan(w.sum()):
            # fallback equal-weight
            w = pd.Series(1.0 / len(mu), index=mu.index)
        else:
            w /= w.sum()
        return w
    except Exception:
        # fallback equal weights
        if returns_df.shape[1] == 0:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / returns_df.shape[1], index=returns_df.columns)


# --- Load full data once ---
prices = pd.read_csv(PRICE_FILE, parse_dates=["date"])
prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)

# Try to load global optimized weights if present (for reference)
if OPT_WEIGHTS.exists():
    try:
        opt_w_global = pd.read_csv(OPT_WEIGHTS, parse_dates=["date"])
        opt_w_global["date"] = pd.to_datetime(opt_w_global["date"]).dt.tz_localize(None)
    except Exception:
        opt_w_global = None
else:
    opt_w_global = None

# --- Define market regimes ---
stress_periods = {
    "Bull_Market_2017_2019": ("2017-01-01", "2019-12-31"),
    "Bear_Market_2020": ("2020-02-01", "2020-09-30"),
    "Sideways_Market_2021_2022": ("2021-01-01", "2022-12-31"),
    "Recent_Volatility_2023_2025": ("2023-01-01", "2025-10-01")
}

# --- Identify benchmark tickers (if present) ---
available_tickers = prices["ticker"].unique().tolist()
benchmark_ticker = None
for cand in ["NIFTY50", "NSEI", "NIFTY", "NIFTY500"]:
    if cand in available_tickers:
        benchmark_ticker = cand
        break

if benchmark_ticker:
    print(f" Benchmark detected: {benchmark_ticker}")
else:
    print(" No benchmark ticker detected in price data (NIFTY). Benchmark comparison will be skipped.")

# --- Run backtest for each regime ---
for scenario_name, (START_DATE, END_DATE) in stress_periods.items():
    print(f"\n Running backtest for {scenario_name} ({START_DATE} → {END_DATE})")

    # Filter data for the regime
    subset_prices = prices[(prices["date"] >= START_DATE) & (prices["date"] <= END_DATE)].copy()
    if subset_prices.empty:
        print(f" Skipping {scenario_name} — no price rows in range.")
        continue

    # Build returns matrix (wide)
    returns = (
        subset_prices.pivot(index="date", columns="ticker", values="close_price")
        .pct_change(fill_method=None)
        .dropna(how="all")
    )

    # If returns is empty after pct_change, skip
    if returns.empty:
        print(f" No returns available for {scenario_name} after pct_change. Skipping.")
        continue

    # Re-optimize weights using historical returns in this regime
    weights = optimize_weights_mv(returns)
    if weights.empty:
        print(f" Optimizer returned no weights for {scenario_name}. Skipping.")
        continue

    # Compute daily portfolio returns
    pf_daily = compute_portfolio_return(weights, returns)
    # Drop NaNs and ensure index sorted
    pf_daily = pf_daily.dropna().sort_index()
    if pf_daily.empty:
        print(f" No portfolio returns computed for {scenario_name}. Skipping.")
        continue

    # Portfolio cumulative value
    pf_value = (1 + pf_daily).cumprod() * INITIAL_CAPITAL

    # Compute advanced metrics for portfolio
    metrics = compute_advanced_metrics(pf_daily, risk_free_rate_annual=RISK_FREE)

    # Compute benchmark metrics for the regime (if available)
    benchmark_metrics = {}
    if benchmark_ticker:
        bench_df = subset_prices[subset_prices["ticker"] == benchmark_ticker][["date", "close_price"]].copy()
        if not bench_df.empty:
            bench_df.set_index("date", inplace=True)
            bench_ret = bench_df["close_price"].pct_change(fill_method=None).dropna()
            benchmark_metrics = compute_advanced_metrics(bench_ret, risk_free_rate_annual=RISK_FREE)
        else:
            # no benchmark data in this regime
            benchmark_metrics = {k: np.nan for k in ["CAGR", "Volatility", "Sharpe", "Sortino", "Calmar", "VaR95", "CVaR95", "MaxDrawdown"]}

    # Prepare summary DataFrame (portfolio + benchmark)
    summary = pd.DataFrame({
        "metric": [
            "CAGR", "Volatility", "Sharpe", "Sortino", "Calmar", "VaR95", "CVaR95", "MaxDrawdown"
        ],
        "portfolio": [
            metrics["CAGR"], metrics["Volatility"], metrics["Sharpe"], metrics["Sortino"],
            metrics["Calmar"], metrics["VaR95"], metrics["CVaR95"], metrics["MaxDrawdown"]
        ],
        "benchmark": [
            benchmark_metrics.get("CAGR", np.nan), benchmark_metrics.get("Volatility", np.nan),
            benchmark_metrics.get("Sharpe", np.nan), benchmark_metrics.get("Sortino", np.nan),
            benchmark_metrics.get("Calmar", np.nan), benchmark_metrics.get("VaR95", np.nan),
            benchmark_metrics.get("CVaR95", np.nan), benchmark_metrics.get("MaxDrawdown", np.nan),
        ]
    }).set_index("metric")

    # --- Save outputs ---
    run_dir = RESULTS_BASE / f"run_{scenario_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save daily returns (index = date, column = return)
    pf_daily.to_frame(name="return").to_csv(run_dir / "daily_returns.csv", index=True)
    # Save portfolio value (index = date, column = value). Use column name '0' or 'portfolio_value'
    pf_value.to_frame(name="portfolio_value").to_csv(run_dir / "portfolio_value.csv", index=True)
    # Save summary as wide CSV for ease
    summary.to_csv(run_dir / "backtest_summary.csv")
    # Save weights used for this regime
    w_df = weights.rename("w").reset_index().rename(columns={"index": "ticker"})
    w_df.to_csv(run_dir / "optimized_weights.csv", index=False)

    # --- Print summary ---
    print(f" {scenario_name} Results (Portfolio):")
    print(summary["portfolio"].to_string())
    if benchmark_ticker:
        print(f"\n {scenario_name} Results (Benchmark: {benchmark_ticker}):")
        print(summary["benchmark"].to_string())

    print(f"Saved to: {run_dir}")

    # --- Plot cumulative portfolio value (and benchmark if available) ---
    plt.figure(figsize=(9, 4))
    plt.plot(pf_value.index, pf_value.values, label="Portfolio", linewidth=1.5)
    if benchmark_ticker and (not bench_df.empty):
        # align benchmark to normalized 1.0 initial to compare shapes
        bench_cum = (1 + bench_ret).cumprod()
        bench_cum = bench_cum.reindex(pf_value.index).ffill().fillna(method="ffill")
        if bench_cum.isna().all():
            pass
        else:
            plt.plot(bench_cum.index, bench_cum.values, label=benchmark_ticker, linewidth=1.0, alpha=0.9)
    plt.title(f"Cumulative Portfolio Value — {scenario_name}")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / f"portfolio_value_{scenario_name}.png", dpi=150)
    plt.close()

print("\n All regime backtests complete. Outputs saved in:")
print(RESULTS_BASE)
