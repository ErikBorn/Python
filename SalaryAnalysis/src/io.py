# src/io.py

import pandas as pd
import numpy as np
import re

# -----------------------------
# Staff loader
# -----------------------------

def _to_num_currency(x):
    """Convert salary-like strings to float (handles $ and commas)."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace("$", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan

def _to_float(x):
    """Coerce to float, return NaN if invalid."""
    try:
        return float(x)
    except Exception:
        return np.nan

def load_staff(path: str) -> pd.DataFrame:
    """
    Load staff CSV and clean key numeric columns.
    Expected columns include: '25-26 Salary', 'Years of Exp', 'Seniority', etc.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]  # normalize headers

    # Clean salaries
    if "25-26 Salary" in df.columns:
        df["25-26 Salary"] = df["25-26 Salary"].map(_to_num_currency)

    # Clean experience years
    if "Years of Exp" in df.columns:
        df["Years of Exp"] = df["Years of Exp"].map(_to_float)

    if "Seniority" in df.columns:
        df["Seniority"] = df["Seniority"].map(_to_float)

    return df


# -----------------------------
# Cohort loader
# -----------------------------

def load_cohort(path: str) -> pd.DataFrame:
    """
    Load state cohort CSV and return tidy long format.
    Columns in the input: 'State', optional base column, and experience bands.
    Output columns: ['experience_band', 'percentile', 'salary'].
    """
    raw = pd.read_csv(path)

    # Identify label column and optional base
    label_col = "State"
    base_col = next((c for c in raw.columns if "Base Salaries" in c), None)

    # Bands = all non-label, non-base
    band_cols = [c for c in raw.columns if c not in [label_col, base_col] and c is not None]

    # Clean numeric
    def _to_num(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).replace("$", "").replace(",", "").replace("\xa0", " ").strip()
        return pd.to_numeric(s, errors="coerce")

    clean = raw.copy()
    if base_col:
        clean[base_col] = clean[base_col].map(_to_num)
    for c in band_cols:
        clean[c] = clean[c].map(_to_num)

    # Normalize percentile labels
    norm = clean[label_col].astype(str).str.lower().str.strip()
    norm = norm.replace({
        "90th percentile": "p90",
        "75th percentile": "p75",
        "50th percentile (median)": "p50",
        "25th percentile": "p25",
        "10th percentile": "p10",
    })
    clean["percentile_label"] = norm
    clean["percentile"] = clean["percentile_label"].map(
        {"p10": 10, "p25": 25, "p50": 50, "p75": 75, "p90": 90}
    )

    # Reshape
    long = clean.melt(
        id_vars=["percentile_label", "percentile"] + ([base_col] if base_col else []),
        value_vars=band_cols,
        var_name="experience_band",
        value_name="salary"
    ).dropna(subset=["percentile", "salary"])

    return long[["experience_band", "percentile", "salary"]]


# -----------------------------
# Writer
# -----------------------------

def write_csv(df: pd.DataFrame, path: str) -> None:
    """Write DataFrame to CSV with index suppressed."""
    df.to_csv(path, index=False)