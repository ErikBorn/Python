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
    Load a cohort CSV and return tidy long format.

    Accepts first-column labels named 'State', 'Local', 'Percentile', or any
    column that appears to contain percentile rows (e.g., '90th Percentile',
    '50th Percentile (Median)', etc.).

    Output columns: ['experience_band', 'percentile', 'salary'] with one row
    per (band, percentile).
    """
    import re
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip() for c in raw.columns]  # normalize headers

    cols_lower = {c: c.lower() for c in raw.columns}

    # --- find the label column (percentile names) ----------------------------
    # Direct matches first
    label_candidates = [c for c in raw.columns
                        if cols_lower[c] in {"state", "local", "percentile"}]

    # If not found, pick the column whose values look like percentile labels
    if not label_candidates:
        def looks_like_pct_series(s: pd.Series) -> bool:
            sample = s.dropna().astype(str).str.lower().head(6)
            pat = re.compile(r"(?:\d{1,2}0th|median).*percentile")
            return sample.str.contains(pat).any()
        for c in raw.columns:
            if looks_like_pct_series(raw[c]):
                label_candidates = [c]
                break

    if not label_candidates:
        raise ValueError(
            f"No label column found in {path}. "
            "Expected a column named 'State' or 'Local' or something that contains percentile labels."
        )

    label_col = label_candidates[0]

    # --- find optional base column (starting salaries) -----------------------
    base_col = None
    for c in raw.columns:
        lc = cols_lower[c]
        if c != label_col and ("base" in lc or "starting" in lc):
            base_col = c
            break

    # --- all remaining columns are the experience bands ----------------------
    band_cols = [c for c in raw.columns if c not in {label_col, base_col}]

    # --- numeric cleaning helper ---------------------------------------------
    def _to_num(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        s = (str(x).replace("$", "").replace(",", "")
                   .replace("\xa0", " ").strip())
        return pd.to_numeric(s, errors="coerce")

    clean = raw.copy()
    if base_col:
        clean[base_col] = clean[base_col].map(_to_num)
    for c in band_cols:
        clean[c] = clean[c].map(_to_num)

    # --- normalize percentile labels to p10/p25/p50/p75/p90 ------------------
    norm = clean[label_col].astype(str).str.lower().str.strip()
    norm = norm.replace({
        "90th percentile": "p90",
        "75th percentile": "p75",
        "50th percentile (median)": "p50",
        "25th percentile": "p25",
        "10th percentile": "p10",
    })
    # also handle minor variants
    norm = (norm.str.replace("percentile", "", regex=False)
                .str.replace("(median)", "", regex=False)
                .str.strip()
                .replace({"90th": "p90", "75th": "p75", "50th": "p50",
                          "25th": "p25", "10th": "p10"}))

    clean["percentile_label"] = norm
    clean["percentile"] = clean["percentile_label"].map(
        {"p10": 10, "p25": 25, "p50": 50, "p75": 75, "p90": 90}
    )

    # sanity-check we actually mapped something
    if clean["percentile"].isna().all():
        raise ValueError(
            f"Could not parse percentile labels from column '{label_col}' in {path}."
        )

    # --- reshape to tidy long -------------------------------------------------
    id_vars = ["percentile_label", "percentile"]
    if base_col:
        id_vars.append(base_col)

    long = (clean
            .melt(id_vars=id_vars,
                  value_vars=band_cols,
                  var_name="experience_band",
                  value_name="salary")
            .dropna(subset=["percentile", "salary"]))

    return long[["experience_band", "percentile", "salary"]]


# -----------------------------
# Writer
# -----------------------------

def write_csv(df: pd.DataFrame, path: str) -> None:
    """Write DataFrame to CSV with index suppressed."""
    df.to_csv(path, index=False)