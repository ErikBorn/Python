# src/rpb.py

from __future__ import annotations
import numpy as np
import pandas as pd

from .cohort import (
    build_band_bins_closed,
    assign_band_from_years_closed,
    interpolate_salary_strict,
)


def compute_rpb(
    long: pd.DataFrame,
    staff: pd.DataFrame,
    target_percentile: float,
    target_inflation: float = 0.0,
    years_col: str = "Years of Exp",
    out_col: str = "RPB Salary",
) -> pd.Series:
    """
    Compute Rough Percentile/Band (RPB) salaries for each person.

    - Uses CLOSED band logic (e.g., 0–5, 6–10, …, 41+ Years).
    - For each band, interpolates the cohort salary at `target_percentile`,
      then optionally applies `target_inflation` (e.g., 0.03 for +3%).
    - Assigns each staff member to a band using CLOSED intervals and
      returns the band target as their RPB.

    Returns:
      pd.Series indexed like `staff`, named `out_col`.
    """
    if years_col not in staff.columns:
        raise ValueError(f"`staff` missing required column: {years_col!r}")

    # Build bands & order from cohort table
    bins, band_order = build_band_bins_closed(long)

    # Compute per-band target salaries (interpolate + inflation)
    band_targets: dict[str, float] = {}
    for _, _, band_label in bins:
        y = interpolate_salary_strict(long, band_label, float(target_percentile))
        if not np.isnan(y):
            band_targets[band_label] = float(y) * (1.0 + float(target_inflation))

    # Assign band per person (CLOSED intervals)
    years = pd.to_numeric(staff[years_col], errors="coerce")
    bands_for_staff = years.apply(lambda v: assign_band_from_years_closed(v, bins))

    # Map to targets
    rpb_vals = bands_for_staff.map(band_targets).astype(float)

    # Return a Series aligned to staff index
    rpb_vals.name = out_col
    return rpb_vals


def validate_rpb(
    long: pd.DataFrame,
    staff: pd.DataFrame,
    rpb_col: str = "RPB Salary",
    years_col: str = "Years of Exp",
) -> pd.DataFrame:
    """
    Per-band comparison of the assigned RPB against the cohort table.

    Returns a DataFrame with one row per band containing:
      - headcount
      - mean_rpb (mean of assigned RPB salaries in that band)
      - cohort_P10, cohort_P25, cohort_P50, cohort_P75, cohort_P90  (from `long`)

    This helps you spot bands where the assigned RPBs land relative to the cohort curve.
    """
    need_staff = {years_col, rpb_col}
    missing = [c for c in need_staff if c not in staff.columns]
    if missing:
        raise ValueError(f"`staff` missing required columns: {missing}")

    # Build CLOSED-interval bins from cohort
    bins, band_order = build_band_bins_closed(long)

    # Assign band to each row in staff
    years = pd.to_numeric(staff[years_col], errors="coerce")
    staff_bands = years.apply(lambda v: assign_band_from_years_closed(v, bins))

    # Frame with bands + RPB
    df = pd.DataFrame({
        "Band": staff_bands,
        "RPB": pd.to_numeric(staff[rpb_col], errors="coerce"),
    }).dropna(subset=["Band"])

    # Per-band headcount and mean RPB
    agg = (
        df.groupby("Band", as_index=True)
          .agg(headcount=("RPB", "size"), mean_rpb=("RPB", "mean"))
          .reindex(band_order)
    )

    # Pull cohort salaries at standard percentiles for each band
    cohort_pts = long.dropna(subset=["experience_band", "percentile", "salary"]).copy()
    cohort_pivot = (
        cohort_pts.pivot_table(
            index="experience_band",
            columns="percentile",
            values="salary",
            aggfunc="first",
        )
        .reindex(band_order)
        .rename_axis(index="Band", columns=None)
    )

    # Ensure columns are in conventional order if present
    for p in [10, 25, 50, 75, 90]:
        if p not in cohort_pivot.columns:
            cohort_pivot[p] = np.nan
    cohort_pivot = cohort_pivot[[10, 25, 50, 75, 90]]
    cohort_pivot.columns = ["cohort_P10", "cohort_P25", "cohort_P50", "cohort_P75", "cohort_P90"]

    # Merge and return
    out = agg.join(cohort_pivot, how="left")

    # Optional: round for readability (do not force formatting)
    return out