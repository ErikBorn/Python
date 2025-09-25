# src/clean.py

import pandas as pd
import numpy as np

# -----------------------------
# Currency / number cleaning
# -----------------------------

def to_num_currency(series: pd.Series) -> pd.Series:
    """
    Convert a Series of salary-like strings (with $, commas, etc.) into floats.
    Non-numeric entries become NaN.
    """
    def _convert(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        s = s.replace("$", "").replace(",", "").replace("\xa0", " ")
        s = s.replace("--", "").replace("—", "").strip()
        try:
            return float(s)
        except ValueError:
            return np.nan

    return series.map(_convert)


# -----------------------------
# Column standardization
# -----------------------------

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with:
      - stripped whitespace
      - lowercased
      - spaces replaced with underscores
    """
    out = df.copy()
    out.columns = (
        out.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return out