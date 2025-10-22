import pandas as pd
import numpy as np

# Load price factors
factors = pd.read_csv("results/factors/price_factors.csv")
factors['date'] = pd.to_datetime(factors['date'])

# Step 1: Shift volatility to positive scale (e.g., z-score → [0.5, 1.5])
factors['vol_adj'] = (factors['volatility'] - factors['volatility'].min()) + 0.0001

# Step 2: Optional normalization (avoid too extreme scaling)
factors['vol_adj'] = factors.groupby('date')['vol_adj'].transform(lambda x: x / x.mean())

# Step 3: Export usable volatility proxy
vol_proxy = factors[['date', 'ticker', 'vol_adj']].rename(columns={'vol_adj': 'volatility'})
vol_proxy.to_csv("results/volatility_matrix.csv", index=False)

print(" Proxy volatility matrix saved to results/volatility_matrix.csv")
