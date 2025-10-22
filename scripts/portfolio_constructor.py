# scripts/portfolio_constructor.py
import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
TOP_N = 50  # change this if you want top 50/100/ etc.
PROJECT_ROOT = Path(r"D:\Portfolio Optimzation project")

PRED_FILE = PROJECT_ROOT / "artifacts" / "alpha_predictions.csv"   # expected: date,ticker,predicted_return
VOL_FILE = PROJECT_ROOT / "results" / "volatility_matrix.csv"     # expected: date,ticker,volatility
OUT_DIR = PROJECT_ROOT / "results" / "portfolios"                             
OUT_WEIGHTS = OUT_DIR / "weekly_weights.csv"
OUT_SUMMARY = OUT_DIR / "summary.csv"
DATE_COL = "date"
TICKER_COL = "ticker"
PRED_COL = "predicted_return"
VOL_COL = "volatility"
# ----------------------------------------


# Setup logging
os.makedirs(OUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(os.path.join(OUT_DIR, "portfolio_constructor.log")), logging.StreamHandler()],
)

def safe_read_csv(path):
    if not Path(path).exists():
        logging.error("File not found: %s", path)
        return None
    return pd.read_csv(path)

def build_portfolios(top_n=TOP_N, pred_file=PRED_FILE, vol_file=VOL_FILE):
    # Load predictions
    preds = safe_read_csv(pred_file)
    if preds is None:
        raise FileNotFoundError(f"Predictions file missing: {pred_file}")
    if DATE_COL not in preds.columns or TICKER_COL not in preds.columns or PRED_COL not in preds.columns:
        raise ValueError(f"Predictions file must contain columns: {DATE_COL}, {TICKER_COL}, {PRED_COL}")
    preds[DATE_COL] = pd.to_datetime(preds[DATE_COL])
    preds = preds[[DATE_COL, TICKER_COL, PRED_COL]].copy()

    # Load volatility if present
    vol = safe_read_csv(vol_file)
    if vol is not None:
        if DATE_COL not in vol.columns or TICKER_COL not in vol.columns or VOL_COL not in vol.columns:
            logging.warning("Vol file present but missing expected columns; ignoring volatility. Expected: %s,%s,%s", DATE_COL, TICKER_COL, VOL_COL)
            vol = None
        else:
            vol[DATE_COL] = pd.to_datetime(vol[DATE_COL])
            vol = vol[[DATE_COL, TICKER_COL, VOL_COL]].copy()

    # Merge
    if vol is not None:
        df = preds.merge(vol, on=[DATE_COL, TICKER_COL], how="left")
    else:
        df = preds.copy()
        df[VOL_COL] = np.nan  # placeholder

    # Basic cleaning
    df = df.dropna(subset=[PRED_COL])  # predictions are mandatory
    df = df[np.isfinite(df[PRED_COL])]

    # --- Select Top N stocks per date (retain date column robustly) ---
    def select_top_n(group):
        topn_grp = group.nlargest(top_n, PRED_COL).copy()
        # ensure the date column exists in the returned frame (some pandas versions drop it)
        try:
            # group.name is the date
            topn_grp[DATE_COL] = group.name
        except Exception:
            # if something goes wrong, just pass (we'll ensure later)
            pass
        return topn_grp

    # use group_keys=False so that apply returns a concatenated frame indexed by original index (no MultiIndex)
    grouped = df.groupby(DATE_COL, group_keys=False)
    # include_groups kwarg added in newer pandas to avoid warnings; attempt to use it if available
    try:
        topn = grouped.apply(select_top_n, include_groups=False).reset_index(drop=True)
    except TypeError:
        # fallback for older pandas versions
        topn = grouped.apply(select_top_n).reset_index(drop=True)

    # Ensure date column present
    if DATE_COL not in topn.columns:
        # try to extract from index if present
        if isinstance(topn.index, pd.MultiIndex) and DATE_COL in topn.index.names:
            topn = topn.reset_index()
        else:
            # if still missing, attempt to merge with df to retrieve date by index (last resort)
            logging.warning("Date column missing from topn after grouping; attempting to recover from df.")
            topn = topn.merge(df[[DATE_COL, TICKER_COL, PRED_COL]], on=[TICKER_COL, PRED_COL], how="left", suffixes=("", "_orig"))
            if DATE_COL not in topn.columns:
                raise KeyError("Unable to recover date column in topn. Inspect grouping logic.")

    # Compute equal weight
    topn["_count"] = topn.groupby(DATE_COL)[TICKER_COL].transform("count")
    topn["w_eq"] = 1.0 / topn["_count"]

    # --- Predicted-return weights (PRW) using transform (robust) ---
    def compute_prw_series(x):
        preds_pos = x.clip(lower=0.0)
        s = preds_pos.sum()
        if s <= 0:
            # fallback: equal allocation within group
            return np.repeat(1.0 / len(preds_pos), len(preds_pos))
        return preds_pos / s

    topn["w_pred"] = topn.groupby(DATE_COL)[PRED_COL].transform(compute_prw_series)

    # --- Volatility-adjusted weights (VAW) robustly aligned ---
    topn[VOL_COL] = topn[VOL_COL].astype(float)

    def compute_vaw(group):
        preds_pos = group[PRED_COL].clip(lower=0.0)
        vol_series = group[VOL_COL].copy()

        # fill missing / invalid volatilities
        if vol_series.isnull().all() or (vol_series <= 0).all():
            vol_fixed = pd.Series(1.0, index=group.index)
        else:
            date_median = vol_series[vol_series > 0].median()
            if np.isnan(date_median) or date_median <= 0:
                date_median = 1.0
            vol_fixed = vol_series.fillna(date_median).clip(lower=1e-9)

        score = preds_pos / vol_fixed
        total = score.sum()
        if total <= 0:
            return pd.Series(np.repeat(1.0 / len(group), len(group)), index=group.index)
        return score / total

    # Use group_keys=False to avoid extra index level; this returns a Series indexed by original rows in most pandas versions
    try:
        vaw_scores = topn.groupby(DATE_COL, group_keys=False).apply(lambda g: compute_vaw(g), include_groups=False)
    except TypeError:
        # older pandas: no include_groups kwarg
        vaw_scores = topn.groupby(DATE_COL, group_keys=False).apply(lambda g: compute_vaw(g))

    # vaw_scores might be a Series indexed by the original topn.index or a MultiIndex (date, idx).
    # Normalize to ensure alignment: always reindex to topn.index and fill any missing with equal weights
    if isinstance(vaw_scores, pd.DataFrame):
        # If by mistake a DataFrame returned, try to select appropriate single column or reduce
        if vaw_scores.shape[1] == 1:
            vaw_scores = vaw_scores.iloc[:, 0]
        else:
            # If multiple columns returned, collapse by taking first (unexpected)
            vaw_scores = vaw_scores.iloc[:, 0]

    # If MultiIndex (date, orig_index), drop the first level if present
    if isinstance(vaw_scores.index, pd.MultiIndex):
        try:
            # try to drop the outer (date) level
            vaw_scores = vaw_scores.droplevel(0)
        except Exception:
            # if that fails, reset and attempt to align by position
            vaw_scores = vaw_scores.reset_index(level=0, drop=True)

    # Now reindex to topn to guarantee perfect alignment
    try:
        vaw_scores = vaw_scores.reindex(topn.index)
    except Exception:
        # final fallback: use equal weights
        logging.warning("vaw_scores reindex failed; falling back to equal weights for VAW.")
        vaw_scores = pd.Series(np.repeat(1.0 / topn.groupby(DATE_COL)[TICKER_COL].transform("count").iloc[0], len(topn)), index=topn.index)

    # Fill any remaining NaNs in vaw_scores with equal allocation per group
    mask_na = vaw_scores.isna()
    if mask_na.any():
        # fill by group equal weights
        for date_key, subidx in topn[mask_na].groupby(DATE_COL).groups.items():
            cnt = len(topn[topn[DATE_COL] == date_key])
            vaw_scores.loc[subidx] = 1.0 / cnt

    topn["w_vol"] = vaw_scores.values.astype(float)

    # Final normalization safety (ensure each weight column sums to 1 per date)
    for wcol in ["w_eq", "w_pred", "w_vol"]:
        topn[wcol] = topn[wcol].astype(float)
        sums = topn.groupby(DATE_COL)[wcol].transform("sum")
        zero_mask = sums == 0
        if zero_mask.any():
            # replace zero-sum groups with equal weights
            for date_key in topn[zero_mask][DATE_COL].unique():
                mask = topn[DATE_COL] == date_key
                topn.loc[mask, wcol] = 1.0 / mask.sum()
            sums = topn.groupby(DATE_COL)[wcol].transform("sum")
        topn[wcol] = topn[wcol] / sums

    # Drop helper cols
    topn = topn.drop(columns=["_count"])

    # Save weights
    topn.to_csv(OUT_WEIGHTS, index=False)
    logging.info("Saved weekly weights to: %s", OUT_WEIGHTS)

    # Create summary: number of stocks, avg volatility, turnover per scheme
    summary_rows = []
    weight_cols = ["w_eq", "w_pred", "w_vol"]
    dates_sorted = sorted(topn[DATE_COL].unique())
    prev_weights = {col: pd.Series(dtype=float) for col in weight_cols}

    for date in dates_sorted:
        subset = topn[topn[DATE_COL] == date].copy()
        row = {
            DATE_COL: date,
            "n_stocks": subset[TICKER_COL].nunique(),
            "avg_volatility": float(subset[VOL_COL].mean(skipna=True)) if not subset[VOL_COL].isnull().all() else np.nan,
        }
        for col in weight_cols:
            curr = subset.set_index(TICKER_COL)[col]
            if prev_weights[col].empty:
                turnover = np.nan
            else:
                union_idx = curr.index.union(prev_weights[col].index)
                prev = prev_weights[col].reindex(union_idx).fillna(0.0)
                cur = curr.reindex(union_idx).fillna(0.0)
                turnover = 0.5 * float((cur - prev).abs().sum())
            row[f"turnover_{col}"] = turnover
            prev_weights[col] = curr
            row[f"avg_weight_{col}"] = float(curr.mean())
            row[f"max_weight_{col}"] = float(curr.max())
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    logging.info("Saved portfolio summary to: %s", OUT_SUMMARY)

    # Basic checks and logs
    checks = topn.groupby(DATE_COL)[["w_eq", "w_pred", "w_vol"]].sum().reset_index()
    bad_weeks = checks[(checks["w_eq"].round(6) != 1) | (checks["w_pred"].round(6) != 1) | (checks["w_vol"].round(6) != 1)]
    if not bad_weeks.empty:
        logging.warning("Found weeks where weight sums != 1 (rounded). Inspect: %s", bad_weeks.head())
    else:
        logging.info("All weight schemes normalized (sum ~ 1 per week).")

    return topn, summary_df

if __name__ == "__main__":
    logging.info("Starting portfolio construction (TOP_N=%s)", TOP_N)
    topn_df, summary_df = build_portfolios()
    logging.info("Done. Produced %d rows of weights and %d summary rows.", len(topn_df), len(summary_df))
