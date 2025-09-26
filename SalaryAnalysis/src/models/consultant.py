# src/models/consultant.py
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import re

from src.cohort import build_band_bins_closed, interpolate_salary_strict

# Reuse the same degree parsing style we used elsewhere
_MA_PAT = re.compile(r"(?i)\b(?:MA|M\.A\.|MS|M\.S\.|MEd|M\.Ed)\b")
_PHD_PAT = re.compile(r"(?i)\b(?:PhD|Ph\.D\.|EdD|Ed\.D\.)\b")

def _coerce_flag(s: Optional[pd.Series]) -> np.ndarray:
    if s is None:
        return None
    # Accept 0/1, booleans, or truthy strings; treat NaN as 0
    return pd.to_numeric(s, errors="coerce").fillna(0).to_numpy(dtype=float)

def _degree_multiplier_positive(edu: pd.Series, *, ma_pct: float, phd_pct: float) -> np.ndarray:
    """
    Positive-only degree multipliers. BA = 0% bump, MA = +ma_pct, PhD = +phd_pct.
    PhD overrides MA if both appear.
    """
    edu = edu.astype(str).str.strip()
    has_phd = edu.str.contains(_PHD_PAT, na=False)
    has_ma  = edu.str.contains(_MA_PAT,  na=False)

    mult = np.ones(len(edu), dtype=float)
    mult[has_ma.to_numpy()]  *= (1.0 + float(ma_pct))
    mult[has_phd.to_numpy()] *= (1.0 + float(phd_pct))  # overrides MA if both
    # If you want explicit override (not multiply), uncomment the next two lines:
    # mult[has_ma.to_numpy()] = 1.0 + float(ma_pct)
    # mult[has_phd.to_numpy()] = 1.0 + float(phd_pct)
    return mult

def _expected_uplift_from_mix(edu: pd.Series, *, ma_pct: float, phd_pct: float) -> float:
    """
    Estimate the average *percentage* uplift implied by the current staff degree mix
    using positive-only logic (BA=0, MA=+ma, PhD=+phd, PhD overrides MA).
    Returns a fraction (e.g., 0.028 => +2.8%).
    """
    edu = edu.astype(str).str.strip()
    has_phd = edu.str.contains(_PHD_PAT, na=False)
    has_ma  = edu.str.contains(_MA_PAT,  na=False)

    n = max(len(edu), 1)
    p_phd = has_phd.sum() / n
    p_ma  = (has_ma & ~has_phd).sum() / n  # MA-only share

    # Average multiplier – 1:
    return p_ma * float(ma_pct) + p_phd * float(phd_pct)

def _interp_with_tail(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """
    1D linear interpolation with linear extrapolation on BOTH sides.
    Right side uses the last-segment slope; left side uses the first-segment slope.
    """
    x  = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)

    if xp.size < 2:
        # degenerate: no slope; just return the single value
        return np.full_like(x, fp[0] if fp.size else np.nan, dtype=float)

    # Interpolate within the anchor range
    y = np.interp(np.clip(x, xp[0], xp[-1]), xp, fp)

    # Right-side linear extrapolation
    right = x > xp[-1]
    if np.any(right):
        mR = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]) if xp[-1] != xp[-2] else 0.0
        y[right] = fp[-1] + mR * (x[right] - xp[-1])

    # Left-side linear extrapolation
    left = x < xp[0]
    if np.any(left):
        mL = (fp[1] - fp[0]) / (xp[1] - xp[0]) if xp[1] != xp[0] else 0.0
        y[left] = fp[0] + mL * (x[left] - xp[0])

    return y

def _build_anchors(long: pd.DataFrame,
                   *,
                   target_percentile: float,
                   inflation: float,
                   base_start: float | None) -> tuple[np.ndarray, np.ndarray]:
    """
    Anchor points (years, salaries):
      - year 0 -> base_start * (1 + inflation)  (if provided)
      - for each band [start..end], anchor at the *band midpoint*:
          midpoint = (start + end)/2  (use end = start+5 for the open-ended last band)
        with salary = cohort band @ P{target} * (1 + inflation)
    """
    bins, _ = build_band_bins_closed(long)

    years: list[float] = []
    vals:  list[float] = []

    if base_start is not None and np.isfinite(base_start):
        years.append(0.0)
        vals.append(float(base_start) * (1.0 + float(inflation)))

    for start, end, label in bins:
        # midpoint of the band (treat open-ended as 5-year wide)
        if np.isfinite(end):
            anchor_y = 0.5 * (float(start) + float(end))
        else:
            anchor_y = float(start) + 2.5

        s = interpolate_salary_strict(long, label, float(target_percentile))
        if not np.isnan(s):
            years.append(anchor_y)
            vals.append(float(s) * (1.0 + float(inflation)))

    # Ensure strictly increasing xp for interpolation
    xp = np.array(sorted(set(years)), dtype=float)
    # When duplicates occurred, keep the last value seen for that year
    latest_for_year = {y: v for y, v in zip(years, vals)}
    fp = np.array([latest_for_year[y] for y in xp], dtype=float)

    return xp, fp

def consultant_predict(
    staff: pd.DataFrame,
    long: pd.DataFrame,
    *,
    target_percentile: float = 50.0,
    inflation: float = 0.04,
    base_start: float | None = None,
    # positive-only bumps (defaults are examples):
    deg_ma_pct: float = 0.02,
    deg_phd_pct: float = 0.04,
    years_col: str = "Years of Exp",
    edu_col: str = "Education Level",
    # flat bonus
    prep_col: str = "Prep",
    prep_bonus: float = 2500.0,
    # NEW: pre-degree downshift of baseline
    pre_degree_down_pct: float = 0.03,   # e.g., 3% downward aim
    auto_downshift: bool = False,        # if True, compute from staff’s degree mix
) -> pd.Series:
    """
    Consultant model (revised):
      1) Build/interpolate a *baseline* curve from anchors (bands @ target %ile, inflated).
      2) Apply a global downward factor (1 - pre_degree_down_pct) so that adding positive degree
         bumps doesn't blow the budget; optionally compute that factor from the staff mix.
      3) Apply positive-only degree multipliers: MA=+ma_pct, PhD=+phd_pct (PhD overrides).
      4) Add flat Prep bonus (+prep_bonus) for rows with Prep==1.
    """
    # Anchors & interpolation
    xp, fp = _build_anchors(
        long,
        target_percentile=target_percentile,
        inflation=inflation,
        base_start=base_start,
    )
    yrs = pd.to_numeric(staff[years_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    base_curve = _interp_with_tail(yrs, xp, fp)

    # Pre-degree downshift (either fixed value or auto from degree mix)
    if auto_downshift:
        edu_series = staff.get(edu_col, pd.Series(index=staff.index, dtype="object"))
        est = _expected_uplift_from_mix(edu_series, ma_pct=deg_ma_pct, phd_pct=deg_phd_pct)
        k = max(0.0, min(1.0, 1.0 - est))   # clamp to [0,1]
    else:
        k = 1.0 - float(pre_degree_down_pct)
    k = max(0.0, k)  # safety
    base_curve = base_curve * k

    # Degree bumps (positive only)
    mult = _degree_multiplier_positive(
        staff.get(edu_col, pd.Series(index=staff.index, dtype="object")),
        ma_pct=deg_ma_pct, phd_pct=deg_phd_pct
    )
    y = base_curve * mult

    # Flat Prep bonus
    pf = _coerce_flag(staff.get(prep_col, staff.get("Prep Rating")))
    if pf is not None and np.any(pf > 0):
        y = y + pf * float(prep_bonus)

    return pd.Series(y, index=staff.index, name="CONS Salary")


def consultant_predict_capped(
    staff: pd.DataFrame,
    long: pd.DataFrame,
    *,
    target_percentile: float,
    inflation: float,
    base_start: float | None = None,
    deg_ma_pct: float = 0.02,
    deg_phd_pct: float = 0.04,
    max_salary: float | None = 100_000.0,
    years_col: str = "Years of Exp",
    edu_col: str = "Education Level",
    prep_col: str = "Prep",
    prep_bonus: float = 2500.0,
    pre_degree_down_pct: float = 0.03,
    auto_downshift: bool = False,
) -> pd.Series:
    """
    Same logic as consultant_predict (with positive degree bumps + pre-downshift),
    then cap at `max_salary`.
    """
    y = consultant_predict(
        staff, long,
        target_percentile=target_percentile,
        inflation=inflation,
        base_start=base_start,
        deg_ma_pct=deg_ma_pct,
        deg_phd_pct=deg_phd_pct,
        years_col=years_col,
        edu_col=edu_col,
        prep_col=prep_col,
        prep_bonus=prep_bonus,
        pre_degree_down_pct=pre_degree_down_pct,
        auto_downshift=auto_downshift,
    ).astype(float)

    if max_salary is not None and np.isfinite(max_salary):
        y = np.minimum(y.to_numpy(dtype=float), float(max_salary))

    return pd.Series(y, index=staff.index, name="CONS_CAP Salary")