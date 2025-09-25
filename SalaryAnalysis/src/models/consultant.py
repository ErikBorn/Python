# src/models/consultant.py
from __future__ import annotations
import numpy as np
import pandas as pd
import re

from src.cohort import build_band_bins_closed, interpolate_salary_strict

# Reuse the same degree parsing style we used elsewhere
_MA_PAT = re.compile(r"(?i)\b(?:MA|M\.A\.|MS|M\.S\.|MEd|M\.Ed)\b")
_PHD_PAT = re.compile(r"(?i)\b(?:PhD|Ph\.D\.|EdD|Ed\.D\.)\b")

def _degree_multiplier(edu: pd.Series, *, ba_pct: float, ma_pct: float, phd_pct: float) -> np.ndarray:
    edu = edu.astype(str).str.strip()
    has_phd = edu.str.contains(_PHD_PAT, na=False)
    has_ma  = edu.str.contains(_MA_PAT,  na=False)
    # default BA; MA overrides BA; PhD overrides both
    mult = np.full(len(edu), 1.0 + float(ba_pct), dtype=float)
    mult[has_ma.to_numpy()]  = 1.0 + float(ma_pct)
    mult[has_phd.to_numpy()] = 1.0 + float(phd_pct)
    return mult

def _interp_with_tail(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """
    1D linear interpolation with linear extrapolation on the right using the
    last segment slope (for 41+ years, etc.). Left of the first anchor clamps to fp[0].
    """
    x  = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float); fp = np.asarray(fp, dtype=float)

    y = np.interp(np.clip(x, xp.min(), xp.max()), xp, fp)

    # Right-side linear extrapolation
    right = x > xp.max()
    if np.any(right):
        m = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]) if xp[-1] != xp[-2] else 0.0
        y[right] = fp[-1] + m * (x[right] - xp[-1])

    # Left-side clamp
    left = x < xp.min()
    if np.any(left):
        y[left] = fp[0]

    return y

def _build_anchors(long: pd.DataFrame,
                   *,
                   target_percentile: float,
                   inflation: float,
                   base_start: float | None) -> tuple[np.ndarray, np.ndarray]:
    """
    Anchor points (years, salaries):
      - year 0 -> base_start * (1 + inflation)  (if provided)
      - for each band [start..end], anchor at (start+3) with salary = cohort band @ P{target} * (1 + inflation)
    """
    bins, _ = build_band_bins_closed(long)

    years = []
    vals  = []

    if base_start is not None and np.isfinite(base_start):
        years.append(0.0)
        vals.append(float(base_start) * (1.0 + float(inflation)))

    for start, end, label in bins:
        anchor_y = float(start + 3)      # 0-5 -> 3, 6-10 -> 8, 11-15 -> 14, ...
        s = interpolate_salary_strict(long, label, float(target_percentile))
        if not np.isnan(s):
            years.append(anchor_y)
            vals.append(float(s) * (1.0 + float(inflation)))

    # Ensure strictly increasing xp for interpolation
    xp = np.array(sorted(set(years)), dtype=float)
    # When duplicates occurred, keep the last value for that year
    fp_map = {y: v for y, v in zip(years, vals)}
    fp = np.array([fp_map[y] for y in xp], dtype=float)

    return xp, fp

def consultant_predict(
    staff: pd.DataFrame,
    long: pd.DataFrame,
    *,
    target_percentile: float = 50.0,
    inflation: float = 0.04,
    base_start: float | None = None,
    deg_ba_pct: float = -0.04,
    deg_ma_pct: float = 0.00,
    deg_phd_pct: float = 0.04,
    years_col: str = "Years of Exp",
    edu_col: str = "Education Level",
) -> pd.Series:
    """
    Consultant model:
      - Linear interpolation across anchors:
            (0, base_start) and (start+3, band_target) for each 5y band.
      - Apply degree multiplier: BA -4%, MA 0%, PhD +4% (configurable)
      - Inflation applied to all anchors.

    Returns a pd.Series aligned to staff.index named "CONS Salary".
    """
    xp, fp = _build_anchors(
        long,
        target_percentile=target_percentile,
        inflation=inflation,
        base_start=base_start,
    )

    yrs = pd.to_numeric(staff[years_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    base_curve = _interp_with_tail(yrs, xp, fp)

    mult = _degree_multiplier(
        staff.get(edu_col, pd.Series(index=staff.index, dtype="object")),
        ba_pct=deg_ba_pct, ma_pct=deg_ma_pct, phd_pct=deg_phd_pct
    )

    out = base_curve * mult
    return pd.Series(out, index=staff.index, name="CONS Salary")