"""
scripts/generate_report.py
---------------------------------------
Automated Reporting & Versioned Results
Creates timestamped result folders and summarizes run outputs.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil
import os

# === CONFIG ===
PROJECT_ROOT = Path(r"D:\Portfolio Optimzation project")
RESULTS_DIR = PROJECT_ROOT / "results"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
VERSIONS_DIR = RESULTS_DIR / "versions"
VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

# === TIMESTAMPED FOLDER ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
run_dir = VERSIONS_DIR / f"run_{timestamp}"
run_dir.mkdir(parents=True, exist_ok=True)

print(f" Creating versioned result folder: {run_dir}")

# === FILES TO COPY ===
files_to_copy = [
    (ARTIFACTS_DIR / "alpha_predictions.csv"),
    (RESULTS_DIR / "portfolios" / "optimized_weights.csv"),
    (RESULTS_DIR / "backtest" / "portfolio_value.csv"),
    (RESULTS_DIR / "backtest" / "backtest_summary.csv"),
]

for file in files_to_copy:
    if file.exists():
        shutil.copy(file, run_dir / file.name)
        print(f" Copied: {file.name}")
    else:
        print(f" Missing: {file.name}")

# === GENERATE RUN SUMMARY ===
summary = {}

# === Load and summarize backtest results ===
backtest_file = run_dir / "backtest_summary.csv"
if backtest_file.exists():
    backtest = pd.read_csv(backtest_file)

    #  Clean and convert columns safely (handles %, strings, NaN)
    for col in ["CAGR", "Volatility", "Sharpe", "MaxDrawdown"]:
        if col in backtest.columns:
            backtest[col] = (
                backtest[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            backtest[col] = pd.to_numeric(backtest[col], errors="coerce")

    #  Build summary dictionary (with graceful fallback)
    summary.update({
        "Annual Return": f"{backtest['CAGR'].iloc[0]:.2f}%" if "CAGR" in backtest.columns else "N/A",
        "Volatility": f"{backtest['Volatility'].iloc[0]:.2f}%" if "Volatility" in backtest.columns else "N/A",
        "Sharpe Ratio": f"{backtest['Sharpe'].iloc[0]:.2f}" if "Sharpe" in backtest.columns else "N/A",
        "Max Drawdown": f"{backtest['MaxDrawdown'].iloc[0]:.2f}%" if "MaxDrawdown" in backtest.columns else "N/A",
    })

else:
    summary["Status"] = " No backtest summary found."



# === Save Markdown Report ===
report_path = run_dir / "run_summary.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"# Portfolio Optimization Run Summary — {timestamp}\n\n")
    for k, v in summary.items():
        f.write(f"- **{k}:** {v}\n")
    f.write("\n Files saved in this version:\n")
    for file in os.listdir(run_dir):
        f.write(f"- {file}\n")

print(f"\n Run summary saved: {report_path}")
print(" Versioned run archive complete.")
