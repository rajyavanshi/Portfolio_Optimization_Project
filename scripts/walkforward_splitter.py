"""
scripts/walkforward_splitter.py
Utility for generating walk-forward cross-validation splits.
Works directly with training_data.csv from Step 3.1
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# --------------------------
# CONFIG
# --------------------------
PROJECT_ROOT = Path(r"D:\Portfolio Optimzation project")
TRAINING_FILE = PROJECT_ROOT / "data" / "training_data.csv"
SPLIT_OUTPUT = PROJECT_ROOT / "artifacts" / "walkforward_splits.json"

# Walk-forward parameters
N_SPLITS = 5              # number of folds
MIN_TRAIN_YEARS = 2       # minimum years for training window (only used for rolling)
WINDOW_TYPE = "expanding"  # or "rolling"
VERBOSE = True

# --------------------------
# LOAD DATA
# --------------------------
df = pd.read_csv(TRAINING_FILE, parse_dates=["date"])
df = df.sort_values("date")

dates = df["date"].sort_values().unique()
n_dates = len(dates)
fold_size = n_dates // (N_SPLITS + 1)   # leave enough room for last test fold

print(f"Total unique dates: {n_dates}, Fold size: {fold_size}")

splits = []
for i in range(1, N_SPLITS + 1):
    train_end_idx = i * fold_size
    test_end_idx = (i + 1) * fold_size if i < N_SPLITS else n_dates

    if WINDOW_TYPE == "expanding":
        train_dates = dates[:train_end_idx]
    elif WINDOW_TYPE == "rolling":
        # rolling window keeps last MIN_TRAIN_YEARS * ~252 trading days
        approx_days = int(MIN_TRAIN_YEARS * 252)
        train_start_idx = max(0, train_end_idx - approx_days)
        train_dates = dates[train_start_idx:train_end_idx]
    else:
        raise ValueError("WINDOW_TYPE must be 'expanding' or 'rolling'")

    test_dates = dates[train_end_idx:test_end_idx]

    train_idx = df.index[df["date"].isin(train_dates)].tolist()
    test_idx = df.index[df["date"].isin(test_dates)].tolist()

    splits.append({
        "fold": i,
        "train_start": str(train_dates[0].date()),
        "train_end": str(train_dates[-1].date()),
        "test_start": str(test_dates[0].date()),
        "test_end": str(test_dates[-1].date()),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "train_indices": train_idx,
        "test_indices": test_idx
    })

    if VERBOSE:
        print(f"Fold {i}: Train [{train_dates[0].date()} → {train_dates[-1].date()}], "
              f"Test [{test_dates[0].date()} → {test_dates[-1].date()}], "
              f"({len(train_idx)} train, {len(test_idx)} test)")

# --------------------------
# SAVE SPLITS
# --------------------------
with open(SPLIT_OUTPUT, "w") as f:
    json.dump(splits, f, indent=2)

print(f"Saved {len(splits)} walk-forward splits to {SPLIT_OUTPUT}")
