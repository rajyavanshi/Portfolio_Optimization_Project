# D:\Portfolio_Project\scripts\data_loader.py
import pandas as pd

def load_csv(file_path, chunksize=None):
    """Load one CSV (optionally in chunks)."""
    if chunksize:
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=chunksize, parse_dates=['date']):
            chunks.append(chunk)
        df = pd.concat(chunks)
    else:
        df = pd.read_csv(file_path, parse_dates=['date'])
    return df
