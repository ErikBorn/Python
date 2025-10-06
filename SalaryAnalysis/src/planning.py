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

def plan_costs(
    staff: pd.DataFrame,
    real_col: str,
    planned_col: str,
    bump: float = 0.02,
    *,
    real_headcount: int | float | None = None,   # NEW: optional calibration target
) -> Dict[str, float]:
    """
    Compute headline costs for a planned salary column vs real salaries.

    If `real_headcount` is provided and > 0, scales the Total Cost by
    (real_headcount / observed_headcount) so totals reflect the full faculty size.

    Returns:
      {
        "Total Cost": float,             # sum(max(planned - real, 0)) * scale
        "Avg Cost per Person": float,    # average uplift across observed rows (unscaled)
        "Num Raised": int,               # count of rows with uplift > baseline bump
        "Headcount": int,                # observed rows used in calc
        "Real Headcount": int | None,    # echo of calibration target
        "Scale Factor": float,           # Real/Observed (1.0 if no calibration)
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
            "Num Raised": 0,
            "Headcount": 0,
            "Real Headcount": int(real_headcount) if real_headcount is not None else None,
            "Scale Factor": 1.0,
        }

    uplift   = np.maximum(planned[m] - real[m], 0.0)
    baseline = real[m] * bump
    obs_hc   = int(uplift.size)

    # --- calibration scale (applies to Total Cost only) ---
    if real_headcount is not None and float(real_headcount) > 0 and obs_hc > 0:
        scale = float(real_headcount) / float(obs_hc)
    else:
        scale = 1.0

    return {
        "Total Cost": float(np.sum(uplift)) * scale,
        "Avg Cost per Person": float(np.mean(uplift)) if uplift.size else 0.0,
        "Num Raised": int(np.sum(uplift > baseline)),   # not scaled
        "Headcount": obs_hc,
        "Real Headcount": int(real_headcount) if real_headcount is not None else None,
        "Scale Factor": float(scale),
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

# --- Money helpers ------------------------------------------------------------

import numpy as np
import pandas as pd

def _round_to_10(x: float | np.ndarray) -> float | np.ndarray:
    """Round to nearest $10 (handles scalars or arrays)."""
    return np.round(np.asarray(x, dtype=float) / 10.0) * 10.0

def format_money_10(x: float) -> str:
    """Round to nearest $10 and format like $12,340 (string)."""
    if pd.isna(x):
        return ""
    v = _round_to_10(x)
    return f"${int(v):,}"


# --- Comparison row: 'Current +2% (all)' -------------------------------------

def global_bump_costs(staff: pd.DataFrame, real_col: str, bump: float = 0.02) -> dict:
    """
    Baseline comparison: everyone gets the same global bump percentage.
    """
    real = pd.to_numeric(staff[real_col], errors="coerce").to_numpy(dtype=float)
    uplift = real * float(bump)
    total = float(np.nansum(uplift))
    avg   = float(np.nanmean(uplift))
    num   = int(np.sum(np.isfinite(uplift)))  # everyone with a salary is 'raised'
    return {"Total Cost": total, "Avg Cost per Person": avg, "Num Raised": num}


# --- Convenience: build a full, formatted costs table ------------------------

def plan_costs_table(
    staff: pd.DataFrame,
    real_col: str,
    planned_cols: list[str],
    labels: list[str] | None = None,
    bump_compare: float | None = 0.02,
    format_output: bool = True,
    *,
    real_headcount: int | float | None = None,   # NEW: calibrate totals
    bump_for_plans: float | None = None,         # optional: override bump used inside plan_costs
) -> pd.DataFrame:
    """
    Build a table of plan costs (+ an optional 'Current +X%' comparison row).

    - planned_cols are the columns already created by plan_salary_column.
    - labels are the row names to show (defaults to planned_cols).
    - If real_headcount is provided, Total Cost is scaled to that HC in plan_costs.
    - If bump_compare is None, the comparison row is omitted.
    """
    if labels is None:
        labels = planned_cols

    rows = {}
    for lab, col in zip(labels, planned_cols):
        rows[lab] = plan_costs(
            staff, real_col, col,
            bump=(0.02 if bump_for_plans is None else float(bump_for_plans)),
            real_headcount=real_headcount
        )

    # Optional comparison row: Current +X% (all)
    if bump_compare is not None:
        rows[f"Current +{int(round(float(bump_compare)*100))}% (all)"] = \
            global_bump_costs(staff, real_col, float(bump_compare))

    df = pd.DataFrame.from_dict(rows, orient="index")

    if format_output:
        fmt = df.copy()
        for c in ["Total Cost", "Avg Cost per Person"]:
            if c in fmt.columns:
                fmt[c] = fmt[c].apply(format_money_10)
        if "Num Raised" in fmt.columns:
            fmt["Num Raised"] = fmt["Num Raised"].astype(int)
        return fmt

    return df

def plan_salary_fte(
    staff: pd.DataFrame,
    *,
    real_actual_col: str,        # e.g. "25-26 Salary (real)"
    model_fte_col: str,          # e.g. "CONS_CAP Salary" (FTE)
    fte_col: str = "Time Value", # fraction (1.0 for FT)
    out_actual_col: str | None = None,  # planned actual $
    out_fte_col: str | None = None,     # planned FTE $ (for plots/tables)
    bump: float = 0.02,
    tol: float = 0.02,
) -> dict[str, pd.Series]:
    # --- required columns present? ---
    for c in (real_actual_col, model_fte_col, fte_col):
        if c not in staff.columns:
            raise ValueError(f"Missing column: {c!r}")

    # --- robust cleaners -----------------------------------------------------
    def _numify_money(s: pd.Series) -> pd.Series:
        """
        Convert a money-like series to float:
        - strips $ and commas
        - trims whitespace
        - coerces to float
        """
        if s.dtype == object:
            s = s.astype(str).str.replace(r"[,\$]", "", regex=True).str.strip()
        return pd.to_numeric(s, errors="coerce")

    def _numify_fte(s: pd.Series) -> pd.Series:
        """
        Convert FTE to float in [0,1]:
        - handles '80%' -> 0.8
        - handles '0.8', ' 1 ', etc.
        - fills NaN with 1.0 (assume full-time if missing)
        """
        if s.dtype == object:
            st = s.astype(str).str.strip()
            pct_mask = st.str.endswith("%", na=False)
            # percent strings -> divide by 100
            s_pct = pd.to_numeric(st.str.rstrip("%"), errors="coerce") / 100.0
            s_num = pd.to_numeric(st, errors="coerce")
            s = np.where(pct_mask, s_pct, s_num)
            s = pd.Series(s, index=s.index)
        else:
            s = pd.to_numeric(s, errors="coerce")

        # sometimes people store 80 instead of 0.8; if values > 1.5, treat like percent
        big = s > 1.5
        s.loc[big] = s.loc[big] / 100.0
        return s.fillna(1.0)

    # --- sanitize inputs -----------------------------------------------------
    real_actual = _numify_money(staff[real_actual_col]).to_numpy(dtype=float)
    model_fte   = _numify_money(staff[model_fte_col]).to_numpy(dtype=float)
    fte         = _numify_fte(staff[fte_col]).to_numpy(dtype=float)

    # Guard against bad rows: if either model or fte is non-finite, model_actual = NaN
    with np.errstate(invalid="ignore"):
        model_actual = np.where(np.isfinite(model_fte) & np.isfinite(fte), model_fte * fte, np.nan)

    # --- apply rule on ACTUAL dollars ---------------------------------------
    planned_actual = np.array(
        [apply_raise_rule(r, m, bump=bump, tol=tol) for r, m in zip(real_actual, model_actual)],
        dtype=float
    )

    # --- outputs -------------------------------------------------------------
    out: dict[str, pd.Series] = {}
    if out_actual_col:
        out[out_actual_col] = pd.Series(planned_actual, index=staff.index, name=out_actual_col)

    if out_fte_col:
        with np.errstate(divide="ignore", invalid="ignore"):
            planned_fte = np.where(fte > 0, planned_actual / fte, np.nan)
        out[out_fte_col] = pd.Series(planned_fte, index=staff.index, name=out_fte_col)

    return out