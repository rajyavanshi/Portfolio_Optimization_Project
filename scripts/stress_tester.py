"""
scripts/stress_tester.py
Aggregates regime-wise backtests and visualizes performance comparison.
Now includes advanced risk metrics (Sharpe, Sortino, Calmar, VaR, CVaR)
and a 4th comparative chart for risk-adjusted ratios.

Part of Stage 6 — Stress Testing.

Author: Suraj Prakash
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

plt.style.use("seaborn-v0_8")
pd.options.display.float_format = "{:,.4f}".format

# === CONFIG ===
PROJECT_ROOT = Path(r"D:\Portfolio Optimzation project")
BASE_DIR = PROJECT_ROOT / "results" / "versions"

# === HELPER FUNCTIONS ===
def compute_advanced_metrics(df, risk_free_rate_annual=0.0):
    """Compute advanced risk-adjusted metrics from daily return series."""
    if df.empty:
        return [np.nan] * 8

    mean_daily = df.mean()
    std_daily = df.std()
    ann_ret = (1 + mean_daily) ** 252 - 1
    ann_vol = std_daily * np.sqrt(252)

    # Sharpe Ratio (annualized)
    rf_daily = (1 + risk_free_rate_annual) ** (1 / 252) - 1
    excess = df - rf_daily
    sharpe = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else np.nan

    # Sortino Ratio
    downside_std = df[df < 0].std() * np.sqrt(252)
    sortino = (ann_ret - risk_free_rate_annual) / downside_std if downside_std > 0 else np.nan

    # Max Drawdown
    cum = (1 + df).cumprod()
    mdd = (cum / cum.cummax() - 1).min()

    # Calmar Ratio
    calmar = ann_ret / abs(mdd) if mdd != 0 else np.nan

    # VaR and CVaR at 95%
    var_95 = np.percentile(df, 5)
    cvar_95 = df[df <= var_95].mean()

    return [ann_ret, ann_vol, sharpe, sortino, calmar, var_95, cvar_95, mdd]


def safe_read_csv(path):
    try:
        return pd.read_csv(path, parse_dates=["date"])
    except Exception:
        return pd.read_csv(path)


# === MAIN EXECUTION ===
print(f"\n Searching for regime runs under: {BASE_DIR}")
regime_folders = [
    p for p in BASE_DIR.glob("run_*")
    if p.is_dir() and "Market" in p.name
]

if not regime_folders:
    raise FileNotFoundError(" No 'run_*' regime folders found. Run Stage 5 first!")

summary_records = []

for run_path in regime_folders:
    scenario_name = run_path.name.replace("run_", "").replace("_", " ")

    pv_file = run_path / "portfolio_value.csv"
    if not pv_file.exists():
        print(f" Skipping {scenario_name}: no portfolio_value.csv found.")
        continue

    df = safe_read_csv(pv_file)
    val_col = [c for c in df.columns if c != "date"][0]
    df["return"] = df[val_col].pct_change(fill_method=None).fillna(0)

    # Compute metrics
    cagr, vol, sharpe, sortino, calmar, var95, cvar95, mdd = compute_advanced_metrics(df["return"])
    summary_records.append({
        "Scenario": scenario_name,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "VaR(95%)": var95,
        "CVaR(95%)": cvar95,
        "MaxDrawdown": mdd
    })

    print(f" Processed {scenario_name}: "
          f"CAGR={cagr:.2%}, Sharpe={sharpe:.2f}, Sortino={sortino:.2f}, Calmar={calmar:.2f}, MDD={mdd:.2%}")

# === AGGREGATE RESULTS ===
scenario_df = pd.DataFrame(summary_records).set_index("Scenario").sort_index()
scenario_df = scenario_df.round(4)
OUTPUT_FILE = BASE_DIR / "stress_test_summary.csv"
scenario_df.to_csv(OUTPUT_FILE)

print("\n Final Stress Test Summary:")
print(scenario_df)

# === VISUALIZATION ===
fig, axes = plt.subplots(4, 1, figsize=(10, 13))

# CAGR comparison
scenario_df["CAGR"].plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("Portfolio CAGR Across Market Regimes")
axes[0].set_ylabel("CAGR")
axes[0].grid(alpha=0.3)

# Volatility comparison
scenario_df["Volatility"].plot(kind="bar", ax=axes[1], color="orange")
axes[1].set_title("Portfolio Volatility Across Market Regimes")
axes[1].set_ylabel("Volatility")
axes[1].grid(alpha=0.3)

# Max Drawdown comparison
scenario_df["MaxDrawdown"].plot(kind="bar", ax=axes[2], color="red")
axes[2].set_title("Portfolio Max Drawdown Across Market Regimes")
axes[2].set_ylabel("Max Drawdown")
axes[2].grid(alpha=0.3)

# NEW: Risk Ratios Comparison (Sharpe, Sortino, Calmar)
scenario_df[["Sharpe", "Sortino", "Calmar"]].plot(kind="bar", ax=axes[3])
axes[3].set_title("Risk-Adjusted Ratios Across Market Regimes")
axes[3].set_ylabel("Ratio Value")
axes[3].grid(alpha=0.3)
axes[3].legend(title="Metric", loc="best")

for ax in axes:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
    try:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=2)
    except Exception:
        pass

plt.tight_layout()
plt.savefig(BASE_DIR / "stress_test_summary.png", dpi=600, bbox_inches="tight")
plt.show()

print(f"\n Stress test summary saved to:\n{OUTPUT_FILE}")
print(f" Chart exported as: {BASE_DIR / 'stress_test_summary.png'}")
