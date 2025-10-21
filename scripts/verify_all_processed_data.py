import pandas as pd
import os
import random

# ---- PATHS ----
raw_dir = r"D:\Portfolio Optimzation project\raw_data"
processed_dir = r"D:\Portfolio Optimzation project\processed_data"

# Create processed directory if it doesn't exist
print(f"Raw path exists: {os.path.exists(raw_dir)}")
print(f"Processed path exists: {os.path.exists(processed_dir)}")

print("Files in raw_data folder:", os.listdir(raw_dir))
print("Files in processed_data folder:", os.listdir(processed_dir))

# ---- STORAGE ----
summary = []

# ---- LOOP OVER ALL FILES ----
for file in os.listdir(raw_dir):
    if file.endswith(".csv"):
        symbol = file.replace(".csv", "")
        raw_path = os.path.join(raw_dir, file)
        processed_path = os.path.join(processed_dir, f"{symbol}_daily.csv")

        if not os.path.exists(processed_path):
            print(f" Skipping {symbol}: Processed file not found.")
            continue

        try:
            # Load both datasets
            raw = pd.read_csv(raw_path, parse_dates=['date'])
            processed = pd.read_csv(processed_path, parse_dates=['date'])

            # Gather info
            raw_rows = len(raw)
            processed_rows = len(processed)
            reduction_ratio = (1 - (processed_rows / raw_rows)) * 100

            raw_start, raw_end = raw['date'].min(), raw['date'].max()
            proc_start, proc_end = processed['date'].min(), processed['date'].max()

            # Missing check
            raw_missing = raw.isna().sum().sum()
            proc_missing = processed.isna().sum().sum()

            # Random date sample to check volume consistency
            date_sample = processed['date'].iloc[random.randint(0, len(processed)-1)]
            raw_day = raw[raw['date'].dt.date == date_sample.date()]
            raw_vol_sum = raw_day['volume'].sum()
            proc_vol_sum = processed.loc[processed['date'] == date_sample, 'volume'].iloc[0]
            vol_match = abs(raw_vol_sum - proc_vol_sum) < 1e-6  # tolerance

            # Append summary
            summary.append({
                "Symbol": symbol,
                "Raw Rows": raw_rows,
                "Processed Rows": processed_rows,
                "Reduction (%)": round(reduction_ratio, 2),
                "Raw Start": str(raw_start)[:10],
                "Raw End": str(raw_end)[:10],
                "Proc Start": str(proc_start)[:10],
                "Proc End": str(proc_end)[:10],
                "Raw Missing": raw_missing,
                "Proc Missing": proc_missing,
                "Volume OK": "✅" if vol_match else "❌"
            })

        except Exception as e:
            print(f" Error processing {symbol}: {e}")

# ---- CREATE SUMMARY DATAFRAME ----
summary_df = pd.DataFrame(summary)
summary_df = summary_df.sort_values("Symbol").reset_index(drop=True)

# ---- SAVE REPORT ----
save_path = os.path.join(processed_dir, "verification_summary.csv")
summary_df.to_csv(save_path, index=False)
print(f"\n Verification summary saved to: {save_path}")

# ---- PRINT A SMALL SAMPLE ----
print("\n --- SAMPLE VERIFICATION REPORT ---")
print(summary_df.head(10).to_string(index=False))
