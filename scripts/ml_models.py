"""
scripts/ml_models.py — STABLE VERSION (Stage 3.3)

Fully optimized for Windows + limited RAM environments.

Key Fixes:
 - No nested multiprocessing (n_jobs=1 inside RandomizedSearch)
 - float32 arrays to cut memory usage
 - reduced n_estimators defaults
 - robust exception handling
 - IC-weighted ensemble + stacking meta-learner (Ridge)
"""

import os
import json
import time
import copy
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LassoCV, RidgeCV, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Optional models
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

# ---------------------------
# CONFIG
# ---------------------------
PROJECT_ROOT = Path(r"D:\Portfolio Optimzation project")
DATA_FILE = PROJECT_ROOT / "data" / "training_data.csv"
SELECTED_FEATURES_FILE = PROJECT_ROOT / "artifacts" / "selected_features.json"
SPLITS_FILE = PROJECT_ROOT / "artifacts" / "walkforward_splits.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_CSV = ARTIFACTS_DIR / "metrics_summary.csv"

RANDOM_SEED = 42
DO_RANDOM_SEARCH = True
N_RANDOM_SEARCH_ITERS = 8    # reduced to avoid memory issues
RANDOM_SEARCH_CV = 2         # small inner CV
N_STACK_SPLITS = 3
KEEP_TOP_N_ENS = 3

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# HELPERS
# ---------------------------
def spearman_ic(a, b):
    try:
        ic, _ = spearmanr(a, b)
        return float(ic if not np.isnan(ic) else 0.0)
    except Exception:
        return 0.0

def hit_ratio(y_true, y_pred):
    signs_true = np.sign(y_true)
    signs_pred = np.sign(y_pred)
    return float((signs_true == signs_pred).mean())

def save_model(obj, path):
    joblib.dump(obj, path)

# ---------------------------
# MODELS
# ---------------------------
def build_base_models():
    models = {
        "lasso": LassoCV(cv=3, n_jobs=1, random_state=RANDOM_SEED, max_iter=3000),
        "ridge": RidgeCV(cv=3),
        "rf": RandomForestRegressor(
            n_estimators=100, n_jobs=1, random_state=RANDOM_SEED, max_depth=8
        ),
    }
    if XGB_AVAILABLE:
        models["xgb"] = xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, n_jobs=1, verbosity=0
        )
    if LGB_AVAILABLE:
        models["lgbm"] = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, n_jobs=1
        )
    return models

# ---------------------------
# RANDOM SEARCH (safe mode)
# ---------------------------
def safe_random_search(estimator, param_dist, X, y):
    """RandomizedSearchCV with n_jobs=1 to avoid joblib conflict"""
    try:
        cv = TimeSeriesSplit(n_splits=RANDOM_SEARCH_CV)
        rs = RandomizedSearchCV(
            estimator, param_distributions=param_dist,
            n_iter=N_RANDOM_SEARCH_ITERS, cv=cv,
            random_state=RANDOM_SEED, n_jobs=1, verbose=0
        )
        rs.fit(X, y)
        return rs.best_estimator_, rs.best_params_
    except Exception as e:
        print("     ⚠️ Random search failed:", e)
        return estimator, None

# ---------------------------
# STACKING FEATURES
# ---------------------------
def build_stacking_meta_features(base_models, X_train, y_train):
    tss = TimeSeriesSplit(n_splits=N_STACK_SPLITS)
    meta_preds = {n: np.zeros(len(X_train)) for n in base_models}
    for tr_idx, val_idx in tss.split(X_train):
        X_tr, y_tr, X_val = X_train.iloc[tr_idx], y_train.iloc[tr_idx], X_train.iloc[val_idx]
        for n, m in base_models.items():
            cloned = copy.deepcopy(m)
            cloned.fit(X_tr, y_tr)
            meta_preds[n][val_idx] = cloned.predict(X_val)
    meta_X = np.vstack([meta_preds[n] for n in base_models]).T
    return meta_X.astype(np.float32), y_train.values, list(base_models.keys())

# ---------------------------
# MAIN
# ---------------------------
def main():
    print("📥 Loading data...")
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.lower()

    with open(SELECTED_FEATURES_FILE) as f:
        features = json.load(f)["selected_features"]
    with open(SPLITS_FILE) as f:
        splits = json.load(f)

    base_template = build_base_models()
    results = []

    for fold in splits:
        fid = fold["fold"]
        print(f"\n===== Fold {fid} =====")
        tr_idx, te_idx = fold["train_indices"], fold["test_indices"]

        train, test = df.loc[tr_idx], df.loc[te_idx]
        X_train, y_train = train[features], train["target_return"]
        X_test, y_test = test[features], test["target_return"]

        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features).astype(np.float32)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=features).astype(np.float32)
        save_model(scaler, MODELS_DIR / f"scaler_fold{fid}.pkl")

        models = {k: copy.deepcopy(v) for k, v in base_template.items()}
        # Random Search only for RF/XGB (if enough memory)
        if DO_RANDOM_SEARCH:
            if "rf" in models:
                print("  🔍 RF tuning...")
                rf_params = {
                    "max_depth": [4, 6, 8],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                }
                models["rf"], _ = safe_random_search(models["rf"], rf_params, X_train, y_train)

            if "xgb" in models and XGB_AVAILABLE:
                print("  🔍 XGB tuning...")
                xgb_params = {
                    "learning_rate": [0.02, 0.05, 0.1],
                    "max_depth": [3, 4, 5],
                    "subsample": [0.7, 0.8, 1.0],
                    "colsample_bytree": [0.7, 0.8, 1.0],
                }
                models["xgb"], _ = safe_random_search(models["xgb"], xgb_params, X_train, y_train)

        preds_dict, ic_map = {}, {}
        for name, model in models.items():
            try:
                start = time.time()
                model.fit(X_train, y_train)
                tr_t = time.time() - start
                preds = model.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                ic = spearman_ic(preds, y_test)
                hr = hit_ratio(y_test, preds)
                ic_map[name] = ic
                preds_dict[name] = preds
                save_model(model, MODELS_DIR / f"{name}_fold{fid}.pkl")
                results.append({
                    "fold": fid, "model": name,
                    "mse": mse, "ic": ic, "hit_ratio": hr,
                    "train_size": len(X_train), "test_size": len(X_test),
                    "train_time": tr_t
                })
                print(f"   {name}: MSE={mse:.2e}, IC={ic:.4f}, Hit={hr:.3f}")
            except Exception as e:
                print(f"   {name} failed: {e}")

        # ENSEMBLE
        if preds_dict:
            ic_series = pd.Series(ic_map)
            ic_series = ic_series.clip(lower=0).sort_values(ascending=False)
            if ic_series.sum() == 0:
                weights = {n: 1 / len(preds_dict) for n in preds_dict}
            else:
                topk = ic_series.head(KEEP_TOP_N_ENS)
                weights = {n: (topk[n] / topk.sum()) if n in topk else 0 for n in preds_dict}

            preds_matrix = np.vstack([preds_dict[n] for n in preds_dict]).T
            weight_vec = np.array([weights[n] for n in preds_dict])
            ens_preds = preds_matrix.dot(weight_vec)

            mse_e = mean_squared_error(y_test, ens_preds)
            ic_e = spearman_ic(ens_preds, y_test)
            hr_e = hit_ratio(y_test, ens_preds)
            results.append({
                "fold": fid, "model": "ensemble_ic_weighted",
                "mse": mse_e, "ic": ic_e, "hit_ratio": hr_e,
                "train_size": len(X_train), "test_size": len(X_test)
            })
            print(f"   Ensemble: MSE={mse_e:.2e}, IC={ic_e:.4f}, Hit={hr_e:.3f}")

        # STACKING
        try:
            meta_X, meta_y, base_order = build_stacking_meta_features(models, X_train, y_train)
            meta = Ridge()
            meta.fit(meta_X, meta_y)
            test_meta = np.vstack([models[n].predict(X_test) for n in base_order]).T.astype(np.float32)
            meta_preds = meta.predict(test_meta)
            mse_m = mean_squared_error(y_test, meta_preds)
            ic_m = spearman_ic(meta_preds, y_test)
            hr_m = hit_ratio(y_test, meta_preds)
            results.append({
                "fold": fid, "model": "stacking_ridge",
                "mse": mse_m, "ic": ic_m, "hit_ratio": hr_m,
                "train_size": len(X_train), "test_size": len(X_test)
            })
            save_model(meta, MODELS_DIR / f"stacking_ridge_fold{fid}.pkl")
            print(f"   Stacking: MSE={mse_m:.2e}, IC={ic_m:.4f}, Hit={hr_m:.3f}")
        except Exception as e:
            print("   Stacking skipped:", e)

    pd.DataFrame(results).to_csv(METRICS_CSV, index=False)
    print(f"\n Results saved to {METRICS_CSV}\n Training completed successfully.")

if __name__ == "__main__":
    main()
