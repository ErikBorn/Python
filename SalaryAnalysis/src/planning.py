# src/planning.py

from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd

from .cohort import (
    build_band_bins_closed,
    assign_band_from_years_closed,
    interpolate_salary_strict,
)


# -----------------------------
# Raise rule (pure function)
# -----------------------------

def apply_raise_rule(real: float, model: float, bump: float = 0.02, tol: float = 0.02) -> float:
    """
    If real salary is below (model * (1 - tol)), raise to model.
    Otherwise, give a bump (e.g., +2%) on real.

    Returns the planned salary for that person.
    """
    if pd.isna(real) or pd.isna(model):
        return np.nan
    real = float(real)
    model = float(model)
    if real < model * (1.0 - float(tol)):
        return model
    return real * (1.0 + float(bump))


# -----------------------------
# Column-level planner (pure)
# -----------------------------

def plan_salary_column(
    staff: pd.DataFrame,
    real_col: str,
    model_col: str,
    out_col: str,
    bump: float = 0.02,
    tol: float = 0.02,
) -> pd.Series:
    """
    Apply `apply_raise_rule` row-wise using `real_col` and `model_col`.

    Returns a Series (aligned to staff.index) named `out_col`.
    Does not mutate `staff`.
    """
    if real_col not in staff.columns or model_col not in staff.columns:
        missing = [c for c in (real_col, model_col) if c not in staff.columns]
        raise ValueError(f"Missing columns in staff: {missing}")

    real = pd.to_numeric(staff[real_col], errors="coerce")
    model = pd.to_numeric(staff[model_col], errors="coerce")

    planned = [
        apply_raise_rule(r, m, bump=bump, tol=tol)
        for r, m in zip(real, model)
    ]
    s = pd.Series(planned, index=staff.index, name=out_col)
    return s


# -----------------------------
# Cost summary vs. real
# -----------------------------

def plan_costs(staff: pd.DataFrame, real_col: str, planned_col: str) -> Dict[str, float]:
    """
    Compute headline costs for a planned salary column vs real salaries.

    Returns:
      {
        "Total Cost": float,             # sum(max(planned - real, 0))
        "Avg Cost per Person": float,    # average uplift across all rows
        "Median Cost per Person": float,
        "Num Raised": int,               # count of rows with uplift > 0
        "Headcount": int
      }
    """
    if real_col not in staff.columns or planned_col not in staff.columns:
        missing = [c for c in (real_col, planned_col) if c not in staff.columns]
        raise ValueError(f"Missing columns in staff: {missing}")

    real = pd.to_numeric(staff[real_col], errors="coerce").to_numpy(dtype=float)
    planned = pd.to_numeric(staff[planned_col], errors="coerce").to_numpy(dtype=float)

    m = np.isfinite(real) & np.isfinite(planned)
    if not np.any(m):
        return {
            "Total Cost": 0.0,
            "Avg Cost per Person": 0.0,
            "Median Cost per Person": 0.0,
            "Num Raised": 0,
            "Headcount": 0,
        }

    uplift = np.maximum(planned[m] - real[m], 0.0)
    return {
        "Total Cost": float(np.sum(uplift)),
        "Avg Cost per Person": float(np.mean(uplift)) if uplift.size else 0.0,
        "Median Cost per Person": float(np.median(uplift)) if uplift.size else 0.0,
        "Num Raised": int(np.sum(uplift > 0)),
        "Headcount": int(uplift.size),
    }


# -----------------------------
# Cohort band plan: keep-above rule
# -----------------------------

def cost_to_percentile_keep_above(
    long: pd.DataFrame,
    staff: pd.DataFrame,
    target_percentile: float,
    target_inflation: float = 0.0,
    keep_above_uplift: float = 0.02,
    years_col: str = "Years of Exp",
    salary_col: str = "25-26 Salary",
) -> pd.DataFrame:
    """
    Bring each person to the cohort band target at `target_percentile` (with forward inflation),
    but if their current salary is already above the band target, keep them above by
    applying `keep_above_uplift` (e.g., +2%) to their current salary.

      person_target = max(band_target, current * (1 + keep_above_uplift))

    Returns a DataFrame with per-band rows and a TOTAL summary row:
      columns:
        - band
        - headcount
        - num_above_band
        - current_total
        - band_target_salary (per-band value)
        - target_total
        - delta
        - delta_per_head
    """
    # Validate inputs
    need_staff = {years_col, salary_col}
    missing = [c for c in need_staff if c not in staff.columns]
    if missing:
        raise ValueError(f"`staff` missing required columns: {missing}")

    # Build CLOSED-interval bins from cohort long
    bins, band_order = build_band_bins_closed(long)

    # Staff subset for calculation
    view = staff[[years_col, salary_col]].copy()
    view[years_col] = pd.to_numeric(view[years_col], errors="coerce")
    view[salary_col] = pd.to_numeric(view[salary_col], errors="coerce")
    view = view.dropna(subset=[years_col, salary_col])

    # Assign band to each staff member
    view["band"] = view[years_col].apply(lambda y: assign_band_from_years_closed(y, bins))
    view = view.dropna(subset=["band"]).copy()

    # Compute band targets (interpolate + inflation)
    band_targets: dict[str, float] = {}
    for _, _, band in bins:
        t = interpolate_salary_strict(long, band, float(target_percentile))
        if not np.isnan(t):
            band_targets[band] = float(t) * (1.0 + float(target_inflation))

    # Attach per-person band target
    view["band_target_salary"] = view["band"].map(band_targets)

    # Per-person plan under "keep-above" rule
    cur = view[salary_col].to_numpy(dtype=float)
    bt = view["band_target_salary"].to_numpy(dtype=float)
    bt = np.where(np.isfinite(bt), bt, -np.inf)  # if missing band target, treat as -inf -> keep_above path

    keep_above = cur * (1.0 + float(keep_above_uplift))
    person_target = np.maximum(bt, keep_above)

    view["person_target_salary"] = person_target
    view["above_band"] = cur > bt

    # Aggregate per band
    per_band = (
        view.groupby("band", as_index=False)
            .agg(
                headcount=(salary_col, "size"),
                num_above_band=("above_band", "sum"),
                current_total=(salary_col, "sum"),
                target_total=("person_target_salary", "sum"),
            )
    )

    # Include the common band target value per band (for readability)
    # If a band has no target in band_targets (unlikely), leave NaN
    per_band["band_target_salary"] = per_band["band"].map(band_targets)

    # Deltas
    per_band["delta"] = per_band["target_total"] - per_band["current_total"]
    per_band["delta_per_head"] = per_band["delta"] / per_band["headcount"].replace({0: np.nan})

    # Order by band start (using CLOSED parsing from cohort)
    def _band_start(band_label: str) -> float:
        # reuse closed parser via bins list (faster than reparsing)
        for s, e, lab in bins:
            if lab == band_label:
                return float(s)
        return 1e9

    per_band["__start__"] = per_band["band"].apply(_band_start)
    per_band = per_band.sort_values("__start__").drop(columns="__start__")

    # TOTAL row
    totals = {
        "band": "TOTAL",
        "headcount": int(per_band["headcount"].sum()) if len(per_band) else 0,
        "num_above_band": int(per_band["num_above_band"].sum()) if len(per_band) else 0,
        "current_total": float(per_band["current_total"].sum()) if len(per_band) else 0.0,
        "target_total": float(per_band["target_total"].sum()) if len(per_band) else 0.0,
    }
    totals["band_target_salary"] = np.nan
    totals["delta"] = totals["target_total"] - totals["current_total"]
    totals["delta_per_head"] = totals["delta"] / max(totals["headcount"], 1)

    result = pd.concat([per_band, pd.DataFrame([totals])], ignore_index=True)
    return result