import pandas as pd, json, numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

PROJECT_ROOT = r"D:\Portfolio Optimzation project"
df = pd.read_csv(f"{PROJECT_ROOT}\\data\\training_data.csv", parse_dates=["date"])
with open(f"{PROJECT_ROOT}\\artifacts\\feature_list.json") as f:
    feature_meta = json.load(f)

feature_cols = feature_meta["feature_columns"]
train = df[df["split"] == "train"]

X, y = train[feature_cols], train["target_return"]

print("\n Dataset Summary")
print(X.describe().T[["mean","std","min","max"]])
print("\nUnique values per column:")
print(X.nunique())

print("\n Checking NaN counts:")
print(X.isna().sum())

# Spearman IC
print("\n Spearman ICs:")
for f in feature_cols:
    ic, _ = spearmanr(X[f], y)
    print(f"{f:25s}: {ic}")

# Mutual info
print("\n Mutual Information Scores:")
mi = mutual_info_regression(X.fillna(0), y.fillna(0), random_state=42)
for f, score in zip(feature_cols, mi):
    print(f"{f:25s}: {score}")

# VIF
print("\n VIF values:")
X_ = X.fillna(X.median())
vif_df = pd.DataFrame({
    "feature": X_.columns,
    "VIF": [variance_inflation_factor(X_.values, i) for i in range(X_.shape[1])]
})
print(vif_df)
