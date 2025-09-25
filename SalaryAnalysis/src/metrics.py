# src/metrics.py

from __future__ import annotations
from typing import List, Dict
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

def achieved_percentiles(
    long: pd.DataFrame,
    staff: pd.DataFrame,
    salary_cols: List[str],
    years_col: str = "Years of Exp",
) -> pd.DataFrame:
    """
    For each salary column in `salary_cols`, compute per-person achieved cohort percentile
    (using CLOSED band logic), then return a table with overall mean/median percentiles.

    Returns a DataFrame with index = model name (salary col) and columns ["Mean %ile", "Median %ile"].
    """
    if years_col not in staff.columns:
        raise ValueError(f"`staff` missing required column: {years_col!r}")
    for c in salary_cols:
        if c not in staff.columns:
            raise ValueError(f"`staff` missing salary column: {c!r}")

    bins, band_order = build_band_bins_closed(long)

    years = pd.to_numeric(staff[years_col], errors="coerce")
    bands = years.apply(lambda y: assign_band_from_years_closed(y, bins))

    out_rows: Dict[str, Dict[str, float]] = {}

    for col in salary_cols:
        vals = pd.to_numeric(staff[col], errors="coerce")
        df = pd.DataFrame({"Band": bands, "Salary": vals}).dropna(subset=["Band", "Salary"])

        if df.empty:
            out_rows[col] = {"Mean %ile": np.nan, "Median %ile": np.nan}
            continue

        per_person_pct = df.apply(
            lambda r: percentile_for_salary_strict(long, r["Band"], r["Salary"]), axis=1
        ).astype(float)

        out_rows[col] = {
            "Mean %ile": float(per_person_pct.mean()) if len(per_person_pct) else np.nan,
            "Median %ile": float(per_person_pct.median()) if len(per_person_pct) else np.nan,
        }

    return pd.DataFrame.from_dict(out_rows, orient="index")


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