# src/metrics.py

from __future__ import annotations
from typing import List, Dict
from .cohort import build_band_bins_closed, assign_band_from_years_closed

import numpy as np
import pandas as pd

from .cohort import (
    build_band_bins_closed,
    assign_band_from_years_closed,
)


# -------------------------------
# Inverse percentile (strict)
# -------------------------------

def percentile_for_salary_strict(long: pd.DataFrame, band_label: str, salary: float) -> float:
    """
    Given a tidy cohort table `long` and a (band, salary), return the cohort percentile.
    - Uses only rows where long.experience_band == band_label.
    - Columns required in `long`: 'experience_band', 'percentile', 'salary'.
    - Piecewise-linear inverse of (percentile -> salary), with salary clamped to table range.

    Returns np.nan if band is missing/no data or salary is NaN.
    """
    if pd.isna(salary):
        return np.nan

    need = {"experience_band", "percentile", "salary"}
    if not need.issubset(long.columns):
        missing = need - set(long.columns)
        raise ValueError(f"`long` missing required columns: {missing}")

    sub = (
        long.loc[long["experience_band"] == band_label, ["percentile", "salary"]]
            .dropna(subset=["percentile", "salary"])
            .astype({"percentile": "float64", "salary": "float64"})
            .sort_values("salary")
    )
    if sub.empty:
        return np.nan

    ys = sub["salary"].to_numpy()
    xs = sub["percentile"].to_numpy()

    s = float(np.clip(float(salary), ys.min(), ys.max()))
    return float(np.interp(s, ys, xs))


# -------------------------------
# Achieved percentiles (overall)
# -------------------------------

# src/metrics.py

def achieved_percentiles(
    long: pd.DataFrame,
    staff: pd.DataFrame,
    salary_cols: list[str],
    years_col: str = "Years of Exp",
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """
    Returns a 2-row DataFrame with index ["Mean %ile", "Median %ile"] and
    one column per salary series in `salary_cols`. If `labels` is provided,
    it is used as the column names (must match length).
    """
    if labels is not None and len(labels) != len(salary_cols):
        raise ValueError("labels must be same length as salary_cols")

    years = pd.to_numeric(staff[years_col], errors="coerce")
    out_cols = labels if labels is not None else salary_cols
    rows_mean, rows_median = [], []

    # Reuse your existing inverse percentile util:
    # percentile_for_salary_strict(long, band_label, salary)
    from .cohort import build_band_bins_closed
    from .metrics import percentile_for_salary_strict  # or wherever it lives in your repo

    bins, band_order = build_band_bins_closed(long)

    def assign_band(y: float):
        if pd.isna(y): return np.nan
        yy = float(y)
        for s, e, lab in bins:
            if (np.isfinite(e) and s <= yy <= e) or (not np.isfinite(e) and yy >= s):
                return lab
        return np.nan

    bands = years.apply(assign_band)

    for col in salary_cols:
        vals = pd.to_numeric(staff[col], errors="coerce")
        pct = []
        for b, v in zip(bands, vals):
            if pd.isna(b) or pd.isna(v):
                pct.append(np.nan)
            else:
                pct.append(percentile_for_salary_strict(long, b, float(v)))
        pct = pd.Series(pct, index=staff.index, dtype=float)
        rows_mean.append(float(pct.mean(skipna=True)))
        rows_median.append(float(pct.median(skipna=True)))

    out = pd.DataFrame(
        [rows_mean, rows_median],
        index=["Mean %ile", "Median %ile"],
        columns=out_cols
    )
    return out


# -------------------------------
# Band medians table
# -------------------------------

def band_medians_table(
    long: pd.DataFrame,
    staff: pd.DataFrame,
    salary_cols: List[str],
    years_col: str = "Years of Exp",
) -> pd.DataFrame:
    """
    Return a per-band table of medians for the requested salary columns,
    plus a Headcount row.

    - Uses CLOSED band logic from `long`.
    - Rows: each model in `salary_cols` (median per band) + final "Headcount" row.
    - Columns: bands in cohort order (as appear in `long`).
    """
    if years_col not in staff.columns:
        raise ValueError(f"`staff` missing required column: {years_col!r}")
    for c in salary_cols:
        if c not in staff.columns:
            raise ValueError(f"`staff` missing salary column: {c!r}")

    bins, band_order = build_band_bins_closed(long)

    years = pd.to_numeric(staff[years_col], errors="coerce")
    bands = years.apply(lambda y: assign_band_from_years_closed(y, bins))

    cols_needed = [years_col] + salary_cols
    bw = staff[cols_needed].copy()
    bw["Band"] = bands
    bw = bw[bw["Band"].isin(band_order)].copy()

    # Headcount per band
    headcount = bw.groupby("Band", as_index=True).size().reindex(band_order, fill_value=0)
    headcount.name = "Headcount"

    # Medians per band for each series
    med = (
        bw.groupby("Band", as_index=True)[salary_cols]
          .median(numeric_only=True)
          .reindex(band_order)
          .T  # rows = models, cols = bands
    )

    # Append headcount as last row (not currency)
    table = med.copy()
    table.loc["Headcount"] = headcount.to_numpy()

    return table

def per_band_median_percentiles(
    long: pd.DataFrame,
    staff: pd.DataFrame,
    cols_dict: dict[str, str],
    years_col: str = "Years of Exp",
    decimals: int = 1,
) -> pd.DataFrame:
    """
    For each band (from cohort `long`), compute the median achieved cohort percentile
    of each requested salary series.

    Args
    ----
    long:      tidy cohort df with columns ['experience_band','percentile','salary']
    staff:     staff dataframe
    cols_dict: mapping {display_label: column_name_in_staff}, in desired row order
               e.g. {"Real":"25-26 Salary","RPB":"RPB Salary","NLM":"Model NL Salary","PW":"PW"}
    years_col: staff column with years of experience
    decimals:  round result to this many decimals

    Returns
    -------
    pd.DataFrame with index = models (rows), columns = band labels (ordered),
    values = median achieved percentile within each band.
    """
    # Validate inputs
    missing = [c for c in [years_col] + list(cols_dict.values()) if c not in staff.columns]
    if missing:
        raise ValueError(f"Missing required columns in staff: {missing}")

    # Build bands from cohort and assign staff to CLOSED bands
    bins, band_order = build_band_bins_closed(long)
    df = staff[[years_col] + list(cols_dict.values())].copy()
    df[years_col] = pd.to_numeric(df[years_col], errors="coerce")
    df["Band"] = df[years_col].apply(lambda y: assign_band_from_years_closed(y, bins))
    df = df[df["Band"].isin(band_order)].copy()
    if df.empty:
        return pd.DataFrame(index=list(cols_dict.keys()), columns=band_order, dtype=float)

    # Compute median achieved percentile per (model, band)
    out = pd.DataFrame(index=list(cols_dict.keys()), columns=band_order, dtype=float)

    for model_label, col in cols_dict.items():
        # ensure numeric salaries
        df[col] = pd.to_numeric(df[col], errors="coerce")
        for band in band_order:
            sub = df[df["Band"] == band][col].dropna()
            if sub.empty:
                out.loc[model_label, band] = np.nan
                continue
            # per-person achieved percentiles in this band
            pcts = [percentile_for_salary_strict(long, band, s) for s in sub.to_numpy()]
            out.loc[model_label, band] = float(np.nanmedian(pcts)) if len(pcts) else np.nan

    return out.round(decimals)