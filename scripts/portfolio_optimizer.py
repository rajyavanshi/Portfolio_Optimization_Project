# scripts/portfolio_optimizer.py
"""
Portfolio optimizer (long-only, per-stock cap) with:
 - MVO sweep (choose best Sharpe)
 - Min-variance fallback
 - Turnover penalty (L1) to reduce trading
 - Transaction cost simulation in summary
 - Factor-based covariance if factors present, else diagonal
 - Plots: weight distribution (last date) + summary timeseries

Outputs:
 - results/portfolios/optimized_weights.csv
 - results/portfolios/opt_summary.csv
 - results/plots/opt_weights_last_date.png
 - results/plots/opt_summary_timeseries.png
"""

import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import cvxpy as cp
except Exception as e:
    raise ImportError("cvxpy is required. Install with: pip install cvxpy") from e

# ---------------- CONFIG ----------------
TOP_N = 50
# Project root directory
PROJECT_ROOT = Path(r"D:\Portfolio Optimzation project")

# Input files (joined with PROJECT_ROOT for absolute safety)
ALPHA_FILE = PROJECT_ROOT / "artifacts" / "alpha_predictions.csv"          # date, ticker, predicted_return
VOL_FILE = PROJECT_ROOT / "results" / "volatility_matrix.csv"              # date, ticker, volatility
FACTORS_FILE = PROJECT_ROOT / "results" / "factors" / "price_factors.csv"  # optional; used to estimate correlations

# Output directories and files
OUT_DIR = PROJECT_ROOT / "results" / "portfolios"
OUT_OPT_WEIGHTS = OUT_DIR / "optimized_weights.csv"
OUT_OPT_SUMMARY = OUT_DIR / "opt_summary.csv"
PLOTS_DIR = OUT_DIR / "plots"

# Column names
DATE_COL = "date"
TICKER_COL = "ticker"
ALPHA_COL = "predicted_return"
VOL_COL = "volatility"
# ----------------------------------------

# Constraints
WEIGHT_CAP = 0.10   # max weight per stock
LONG_ONLY = True

# Solver and sweep settings
CVX_SOLVER = cp.SCS  # try cp.OSQP or cp.ECOS if you prefer
LAMBDA_GRID = np.logspace(-4, 2, 25)
EPS = 1e-12

# Turnover penalty + transaction costs
TURNOVER_PENALTY_GAMMA = 0.1  # convex L1 penalty coefficient applied in objective
TRANSACTION_COST_RATE = 0.001  # 0.1% per turnover (example). Applied to turnover * portfolio value (assume 1)

# Plot settings
PLOT_DPI = 120
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(os.path.join(OUT_DIR, "portfolio_optimizer.log")),
                              logging.StreamHandler()])

def safe_read_csv(path):
    p = Path(path)
    if not p.exists():
        logging.warning("File not found: %s", path)
        return None
    return pd.read_csv(p)

def build_covariance_matrix(tickers, vols_series, factors_df=None):
    """
    Build covariance matrix for tickers.
    vols_series: pd.Series indexed by ticker
    factors_df: DataFrame indexed by ticker with factor columns (optional)
    """
    n = len(tickers)
    D = np.diag(vols_series.reindex(tickers).fillna(vols_series.median()).values)
    Sigma = D @ np.eye(n) @ D  # diag fallback

    if factors_df is not None:
        # factors_df: index=ticker, columns=factors
        f = factors_df.reindex(tickers)
        if f.dropna(how='all').shape[0] == 0:
            return Sigma
        # fill small missing with column medians
        f = f.fillna(f.median())
        try:
            corr = f.T.corr().values
            corr = np.nan_to_num(corr, nan=0.0)
            corr = (corr + corr.T) / 2.0
            np.fill_diagonal(corr, 1.0)
            Sigma = D @ corr @ D
        except Exception as ex:
            logging.warning("Factor-based correlation failed: %s. Using diagonal Sigma.", ex)
            Sigma = D @ np.eye(n) @ D
    return Sigma

def solve_min_variance(Sigma, prev_w=None, cap=WEIGHT_CAP, gamma=0.0):
    n = Sigma.shape[0]
    w = cp.Variable(n)
    # objective: minimize w^T Sigma w + gamma * ||w - prev_w||_1  (if prev_w provided)
    obj = cp.quad_form(w, Sigma)
    if gamma > 0 and prev_w is not None:
        obj = obj + gamma * cp.norm1(w - prev_w)
    objective = cp.Minimize(obj)
    constraints = [cp.sum(w) == 1]
    if LONG_ONLY:
        constraints += [w >= 0]
    if cap is not None:
        constraints += [w <= cap]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=CVX_SOLVER, warm_start=True)
    if w.value is None:
        raise RuntimeError("Min-variance optimization failed.")
    wv = np.array(w.value).flatten()
    # enforce non-neg & cap & renormalize to tolerate numerical noise
    if LONG_ONLY:
        wv = np.maximum(0.0, wv)
    if cap is not None:
        wv = np.minimum(wv, cap)
    if wv.sum() <= 0:
        wv = np.repeat(1.0 / len(wv), len(wv))
    else:
        wv = wv / wv.sum()
    return wv

def solve_mvo_sweep(mu, Sigma, prev_w=None, cap=WEIGHT_CAP, gamma=0.0, lambda_grid=LAMBDA_GRID):
    """
    Solve convex problems: minimize lam * w^T Σ w - mu^T w + gamma * ||w - prev_w||1
    (equivalently maximize mu^T w - lam w^T Σ w - gamma*||...||)
    Choose solution with highest realized Sharpe (mu^T w)/(sqrt(w^T Σ w))
    """
    n = Sigma.shape[0]
    best_sharpe = -np.inf
    best_w = None

    mu = np.array(mu).flatten()
    for lam in lambda_grid:
        w = cp.Variable(n)
        quad = lam * cp.quad_form(w, Sigma)
        linear = - mu.T @ w  # minimization form
        reg = 0
        if gamma > 0 and prev_w is not None:
            reg = gamma * cp.norm1(w - prev_w)
        objective = cp.Minimize(quad + linear + reg)
        constraints = [cp.sum(w) == 1]
        if LONG_ONLY:
            constraints += [w >= 0]
        if cap is not None:
            constraints += [w <= cap]
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=CVX_SOLVER, warm_start=True, verbose=False)
        except Exception as e:
            logging.debug("Solver failed for lambda=%s: %s", lam, e)
            continue
        if w.value is None:
            continue
        wv = np.array(w.value).flatten()
        if LONG_ONLY:
            wv = np.maximum(0.0, wv)
        if cap is not None:
            wv = np.minimum(wv, cap)
        if wv.sum() <= 0:
            continue
        wv = wv / wv.sum()
        port_ret = mu.dot(wv)
        port_var = max(wv.dot(Sigma.dot(wv)), 0.0)
        port_vol = np.sqrt(port_var)
        sharpe = port_ret / (port_vol + EPS)
        if np.isfinite(sharpe) and sharpe > best_sharpe:
            best_sharpe = sharpe
            best_w = wv
    if best_w is None:
        logging.warning("MVO sweep found no feasible solution; falling back to min-variance.")
        best_w = solve_min_variance(Sigma, prev_w=prev_w, cap=cap, gamma=gamma)
        # compute sharpe
        port_ret = mu.dot(best_w)
        port_vol = np.sqrt(max(best_w.dot(Sigma.dot(best_w)), 0.0))
        best_sharpe = port_ret / (port_vol + EPS) if port_vol > 0 else np.nan
    return best_w, best_sharpe

def run_optimizer():
    alphas = safe_read_csv(ALPHA_FILE)
    if alphas is None:
        raise FileNotFoundError(ALPHA_FILE + " not found")
    vol = safe_read_csv(VOL_FILE)
    factors = safe_read_csv(FACTORS_FILE)

    # Validate
    for col in [DATE_COL, TICKER_COL, ALPHA_COL]:
        if col not in alphas.columns:
            raise ValueError(f"Alpha file must contain column: {col}")
    alphas[DATE_COL] = pd.to_datetime(alphas[DATE_COL])

    if vol is not None:
        if not {DATE_COL, TICKER_COL, VOL_COL}.issubset(vol.columns):
            logging.warning("Vol file missing expected columns; ignoring vol file.")
            vol = None
        else:
            vol[DATE_COL] = pd.to_datetime(vol[DATE_COL])

    if factors is not None:
        if not {DATE_COL, TICKER_COL}.issubset(factors.columns):
            logging.warning("Factors file missing expected columns; ignoring factors.")
            factors = None
        else:
            factors[DATE_COL] = pd.to_datetime(factors[DATE_COL])

    results = []
    summary_rows = []
    prev_weights_mvo = {}
    prev_weights_minvar = {}

    dates = sorted(alphas[DATE_COL].unique())
    for dt in dates:
        df_alpha = alphas[alphas[DATE_COL] == dt].copy()
        if vol is not None:
            df = df_alpha.merge(vol[vol[DATE_COL] == dt], on=[DATE_COL, TICKER_COL], how="left")
        else:
            df = df_alpha.copy()
            df[VOL_COL] = np.nan

        # Top N
        if TOP_N is not None:
            df = df.nlargest(TOP_N, ALPHA_COL).reset_index(drop=True)

        df = df.dropna(subset=[ALPHA_COL]).reset_index(drop=True)
        tickers = df[TICKER_COL].tolist()
        if len(tickers) == 0:
            logging.warning("No tickers for date %s. Skipping.", dt)
            continue

        mu = df[ALPHA_COL].astype(float).values
        # fix vol series safely (this was the original bug)
        if VOL_COL in df.columns:
            median_vol = np.nanmedian(df[VOL_COL].values)
            if np.isnan(median_vol) or np.isclose(median_vol, 0.0):
                median_vol = 1.0
            vols = pd.Series(df[VOL_COL].fillna(median_vol).values, index=tickers)
        else:
            vols = pd.Series(np.repeat(1.0, len(tickers)), index=tickers)

        # prepare factor exposures for date if available
        factors_for_date = None
        if factors is not None:
            fdate = factors[factors[DATE_COL] == dt].copy()
            # take factor columns
            drop_cols = {DATE_COL, TICKER_COL}
            fact_cols = [c for c in fdate.columns if c not in drop_cols]
            if fact_cols:
                fsmall = fdate[[TICKER_COL] + fact_cols].drop_duplicates(TICKER_COL).set_index(TICKER_COL)
                factors_for_date = fsmall.reindex(tickers)

        # Build covariance
        Sigma = build_covariance_matrix(tickers, vols, factors_for_date)
        # Make PSD
        Sigma = (Sigma + Sigma.T) / 2.0
        try:
            eigs = np.linalg.eigvalsh(Sigma)
            min_eig = eigs.min()
            if min_eig < 1e-8:
                Sigma += np.eye(len(Sigma)) * (1e-8 - min_eig)
        except Exception:
            Sigma = np.diag(vols.values ** 2)

        # previous weights for turnover penalty
        prev_w_mvo = None
        prev_w_minvar = None
        # build vector of previous w if exists (from prev_weights dict keyed by ticker)
        if prev_weights_mvo:
            prev_w_mvo = np.array([prev_weights_mvo.get(t, 0.0) for t in tickers], dtype=float)
        if prev_weights_minvar:
            prev_w_minvar = np.array([prev_weights_minvar.get(t, 0.0) for t in tickers], dtype=float)

        # Solve min-variance (with turnover penalty)
        try:
            w_minvar = solve_min_variance(Sigma, prev_w=prev_w_minvar, cap=WEIGHT_CAP, gamma=TURNOVER_PENALTY_GAMMA)
        except Exception as e:
            logging.warning("MinVar failed on %s: %s. Falling back equal.", dt, e)
            w_minvar = np.repeat(1.0 / len(tickers), len(tickers))

        # Solve MVO sweep (with turnover penalty)
        try:
            w_mvo, best_sharpe = solve_mvo_sweep(mu, Sigma, prev_w=prev_w_mvo, cap=WEIGHT_CAP, gamma=TURNOVER_PENALTY_GAMMA)
        except Exception as e:
            logging.warning("MVO sweep failed on %s: %s. Falling back to minvar.", dt, e)
            w_mvo = w_minvar
            best_sharpe = np.nan

        # compute turnovers and transaction costs (assuming portfolio value = 1)
        prev_vector = np.array([prev_weights_mvo.get(t, 0.0) for t in tickers]) if prev_weights_mvo else np.zeros(len(tickers))
        turnover_mvo = 0.5 * np.abs(w_mvo - prev_vector).sum()
        tc_cost_mvo = turnover_mvo * TRANSACTION_COST_RATE  # fraction of portfolio lost to trading

        prev_vector2 = np.array([prev_weights_minvar.get(t, 0.0) for t in tickers]) if prev_weights_minvar else np.zeros(len(tickers))
        turnover_minvar = 0.5 * np.abs(w_minvar - prev_vector2).sum()
        tc_cost_minvar = turnover_minvar * TRANSACTION_COST_RATE

        # metrics
        port_ret_mvo = float(np.dot(mu, w_mvo) - tc_cost_mvo)
        port_var_mvo = float(np.dot(w_mvo, Sigma.dot(w_mvo)))
        port_vol_mvo = float(np.sqrt(max(port_var_mvo, 0.0)))
        sharpe_mvo = float(port_ret_mvo / (port_vol_mvo + EPS)) if port_vol_mvo > 0 else np.nan

        port_ret_minvar = float(np.dot(mu, w_minvar) - tc_cost_minvar)
        port_var_minvar = float(np.dot(w_minvar, Sigma.dot(w_minvar)))
        port_vol_minvar = float(np.sqrt(max(port_var_minvar, 0.0)))
        sharpe_minvar = float(port_ret_minvar / (port_vol_minvar + EPS)) if port_vol_minvar > 0 else np.nan

        # store
        for i, t in enumerate(tickers):
            results.append({
                DATE_COL: dt,
                TICKER_COL: t,
                ALPHA_COL: float(mu[i]),
                VOL_COL: float(vols.iloc[i]) if i < len(vols) else np.nan,
                "w_mvo_best_sharpe": float(w_mvo[i]),
                "w_minvar": float(w_minvar[i])
            })

        summary_rows.append({
            DATE_COL: dt,
            "n_assets": len(tickers),
            "mvo_expected_return_net_tc": port_ret_mvo,
            "mvo_vol": port_vol_mvo,
            "mvo_sharpe_net_tc": sharpe_mvo,
            "mvo_turnover": turnover_mvo,
            "mvo_tc_cost": tc_cost_mvo,
            "minvar_expected_return_net_tc": port_ret_minvar,
            "minvar_vol": port_vol_minvar,
            "minvar_sharpe_net_tc": sharpe_minvar,
            "minvar_turnover": turnover_minvar,
            "minvar_tc_cost": tc_cost_minvar
        })

        # update prev weight dicts for next date (map ticker->weight)
        prev_weights_mvo = {t: float(w) for t, w in zip(tickers, w_mvo)}
        prev_weights_minvar = {t: float(w) for t, w in zip(tickers, w_minvar)}

        logging.info("Date %s: %d assets. MVO_sharpe=%.6f (tc=%.6f), MinVar_sharpe=%.6f (tc=%.6f)",
                     dt, len(tickers),
                     sharpe_mvo if np.isfinite(sharpe_mvo) else -999, tc_cost_mvo,
                     sharpe_minvar if np.isfinite(sharpe_minvar) else -999, tc_cost_minvar)

    # write outputs
    res_df = pd.DataFrame(results)
    if res_df.empty:
        logging.warning("No optimization results produced.")
    else:
        res_df.to_csv(OUT_OPT_WEIGHTS, index=False)
        logging.info("Saved optimized weights to %s", OUT_OPT_WEIGHTS)

    opt_summary_df = pd.DataFrame(summary_rows)
    opt_summary_df.to_csv(OUT_OPT_SUMMARY, index=False)
    logging.info("Saved optimization summary to %s", OUT_OPT_SUMMARY)

    # PLOTS
    try:
        if not opt_summary_df.empty:
        # timeseries plot: MVO & MinVar Sharpe and turnover
            fig, ax1 = plt.subplots(figsize=(8, 4), dpi=PLOT_DPI)
            dates = pd.to_datetime(opt_summary_df[DATE_COL])

        # Sharpe (net TC)
            ax1.plot(dates, opt_summary_df["mvo_sharpe_net_tc"], label="MVO Sharpe (net TC)", marker='o', linewidth=1.5)
            ax1.plot(dates, opt_summary_df["minvar_sharpe_net_tc"], label="MinVar Sharpe (net TC)", marker='s', linewidth=1.5)
            ax1.set_ylabel("Sharpe (net TC)")
            ax1.legend(loc="upper left")

        # Expand Y-axis range slightly for visibility
            ymin, ymax = ax1.get_ylim()
            ax1.set_ylim(ymin - abs(ymax - ymin) * 5, ymax + abs(ymax - ymin) * 5)

        # Turnover (right axis)
            ax2 = ax1.twinx()
            ax2.plot(dates, opt_summary_df["mvo_turnover"], color="gray", linestyle="--", linewidth=1.2, label="MVO Turnover")
            ax2.plot(dates, opt_summary_df["minvar_turnover"], color="black", linestyle="--", linewidth=1.2, label="MinVar Turnover")
            ax2.set_ylabel("Turnover")

        # Expand turnover range a bit
            ymin2, ymax2 = ax2.get_ylim()
            ax2.set_ylim(ymin2 - abs(ymax2 - ymin2) * 0.2, ymax2 + abs(ymax2 - ymin2) * 0.2)

            fig.autofmt_xdate()
            ax1.set_title("Optimizer summary timeseries")
            fig.tight_layout()
            fig_path = os.path.join(PLOTS_DIR, "opt_summary_timeseries.png")
            fig.savefig(fig_path, bbox_inches='tight')
            plt.close(fig)
            logging.info("Saved plot: %s", fig_path)


        # weight distribution for last date
        if not res_df.empty:
            last_date = res_df[DATE_COL].max()
            last_df = res_df[res_df[DATE_COL] == last_date]
            fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=PLOT_DPI)
            ax[0].barh(last_df[TICKER_COL], last_df["w_mvo_best_sharpe"])
            ax[0].set_title(f"Wts MVO (last {last_date})")
            ax[0].set_xlabel("weight")
            ax[1].barh(last_df[TICKER_COL], last_df["w_minvar"])
            ax[1].set_title(f"Wts MinVar (last {last_date})")
            ax[1].set_xlabel("weight")
            plt.tight_layout()
            wpath = os.path.join(PLOTS_DIR, "opt_weights_last_date.png")
            fig.savefig(wpath)
            plt.close(fig)
            logging.info("Saved plot: %s", wpath)
    except Exception as ex:
        logging.warning("Plotting failed: %s", ex)

    return res_df, opt_summary_df

if __name__ == "__main__":
    logging.info("Starting portfolio optimizer (TOP_N=%s, cap=%s, long-only=%s)", TOP_N, WEIGHT_CAP, LONG_ONLY)
    opt_w, opt_summary = run_optimizer()
    logging.info("Done. Produced %d optimized-weight rows for %d dates", len(opt_w) if opt_w is not None else 0,
                 opt_summary.shape[0] if opt_summary is not None else 0)
