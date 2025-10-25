"""
scripts/run_pipeline.py
------------------------------------
Master pipeline runner for the Portfolio Optimization Project.

Stages:
1️. Generate Rolling Alpha (ML model training & predictions)
2️. Optimize Portfolio Weights (MVO / MinVar)
3️. Generate Dynamic Weights (rebalancing engine)
4️. Backtest Portfolio Performance
5️. Benchmark Comparison (vs NIFTY 500)

Author: Suraj Prakash
"""

import subprocess
import os
from datetime import datetime
from pathlib import Path

# === CONFIG ===
PROJECT_ROOT = Path(r"D:\Portfolio Optimzation project")
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
log_file = LOG_DIR / f"pipeline_log_{timestamp}.txt"

# === Helper ===
def run_stage(name: str, script: str):
    print(f"\n Running Stage: {name}")
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"\n=== {name} | {datetime.now()} ===\n")
        try:
            subprocess.run(["python", script], cwd=PROJECT_ROOT, check=True, text=True, stdout=log, stderr=log)
            print(f" {name} completed successfully.")
            log.write(f" {name} completed successfully.\n")
        except subprocess.CalledProcessError:
            print(f" Error in {name}. Check log file: {log_file}")
            log.write(f" Error in {name}\n")
            raise

# === PIPELINE EXECUTION ORDER ===
if __name__ == "__main__":
    print("\n Portfolio Optimization Unified Pipeline Started")
    print(f"Logs: {log_file}\n")

    stages = [
        ("Stage 1: Rolling Alpha Generation", "scripts/generate_rolling_alpha.py"),
        ("Stage 2: Portfolio Optimization", "scripts/portfolio_optimizer.py"),
        ("Stage 3: Dynamic Weight Generation", "scripts/generate_dynamic_weights.py"),
        ("Stage 4: Backtesting", "scripts/backtester.py")
    ]

    for name, script in stages:
        run_stage(name, script)

        # === Benchmark Comparison (optional) ===
    benchmark_script = PROJECT_ROOT / "scripts" / "benchmark_compare.py"
    if benchmark_script.exists():
        run_stage("Stage 5: Benchmark Comparison", str(benchmark_script))
    else:
        print(" Benchmark comparison script not found. Skipping.")

    # === Automated Reporting ===
    run_stage("Stage 6: Automated Reporting", "scripts/generate_report.py")

    print("\n Pipeline execution completed.")
    print(f"Full logs saved to: {log_file}")

