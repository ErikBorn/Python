# src/models/median_target.py
from __future__ import annotations
from typing import Dict, Optional
import re
import numpy as np
import pandas as pd

from src.cohort import build_band_bins_closed

# --- degree parsing (match consultant.py behavior) ---------------------------
_MA_PAT  = re.compile(r"(?i)\b(?:MA|M\.A\.|MS|M\.S\.|MEd|M\.Ed)\b")
_PHD_PAT = re.compile(r"(?i)\b(?:PhD|Ph\.D\.|EdD|Ed\.D\.)\b")

def _num(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.replace(r"[,\$]", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")

def _anchor_year(start: float) -> float:
    # Pin at band start + 2 years (the "3rd year" of the band)
    return float(start) + 2.0

def _build_anchors(long: pd.DataFrame, band_targets: Dict[str, float], *, inflation: float = 0.0):
    bins, _ = build_band_bins_closed(long)
    X, Y = [], []
    for start, _, label in bins:
        if label in band_targets and pd.notna(band_targets[label]):
            X.append(_anchor_year(start))
            Y.append(float(band_targets[label]) * (1.0 + float(inflation)))
    if not X:
        raise ValueError("No usable anchors: check band_targets against cohort bands.")
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    m = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[m], Y[m]
    if X.size < 2:
        X = np.array([X[0]-2.5, X[0], X[0]+2.5], dtype=float)
        Y = np.array([Y[0],     Y[0], Y[0]],     dtype=float)
    return X, Y

def _interp_clamped(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return np.interp(x, xp, fp, left=fp[0], right=fp[-1])

def _degree_multiplier_positive(edu: pd.Series, *, ma_pct: float, phd_pct: float) -> np.ndarray:
    edu = edu.astype(str).str.strip()
    has_phd = edu.str.contains(_PHD_PAT, na=False)
    has_ma  = edu.str.contains(_MA_PAT,  na=False)
    mult = np.ones(len(edu), dtype=float)
    mult[has_ma.to_numpy()]  *= (1.0 + float(ma_pct))
    mult[has_phd.to_numpy()] *= (1.0 + float(phd_pct))  # overrides MA if both
    return mult

def _expected_uplift_from_mix(edu: pd.Series, *, ma_pct: float, phd_pct: float) -> float:
    edu = edu.astype(str).str.strip()
    has_phd = edu.str.contains(_PHD_PAT, na=False)
    has_ma  = edu.str.contains(_MA_PAT,  na=False)
    n = max(len(edu), 1)
    p_phd = has_phd.sum() / n
    p_ma  = (has_ma & ~has_phd).sum() / n  # MA-only share
    return p_ma * float(ma_pct) + p_phd * float(phd_pct)

def _add_flag_bonuses(y: np.ndarray, staff: pd.DataFrame, bonus_pairs: list[tuple[str, float]]) -> np.ndarray:
    out = y.copy()
    for col, amt in bonus_pairs:
        if not col or not (amt or 0):
            continue
        flags = pd.to_numeric(staff.get(col), errors="coerce").fillna(0).to_numpy(dtype=float)
        out = out + (flags > 0).astype(float) * float(amt)
    return out

def median_target_predict(
    staff: pd.DataFrame,
    long: pd.DataFrame,
    *,
    band_targets: Dict[str, float],        # {"0-5": 62000, "6-10": 68000, ...}
    inflation: float = 0.0,
    years_col: str = "Years of Exp",
    edu_col: str = "Education Level",
    prep_col: str = "Prep",
    prep_bonus: float = 0.0,
    deg_ma_pct: float = 0.04,
    deg_phd_pct: float = 0.08,
    skill_col: Optional[str] = None,
    skill_bonus: float = 0.0,
    leadership_col: Optional[str] = None,
    leadership_bonus: float = 0.0,
    # NEW: match consultant "aim down" behavior
    pre_degree_down_pct: float = 0.04,     # fixed downshift (e.g., 3%)
    auto_downshift: bool = False,          # if True, compute from degree mix
) -> pd.Series:
    """
    Median-target curve:
      anchors (band medians @ start+2) -> smooth baseline
      -> OPTIONAL pre-degree downshift
      -> degree multipliers
      -> flat stipends
    Returns FTE-level salary.
    """
    if years_col not in staff.columns:
        raise ValueError(f"Missing years column: {years_col!r}")

    # 1) baseline from anchors
    X, Y = _build_anchors(long, band_targets, inflation=inflation)
    years = _num(staff[years_col]).fillna(0.0).to_numpy(dtype=float)
    base  = _interp_clamped(years, X, Y)

    # 2) aim baseline down BEFORE degree bumps (like consultant)
    if auto_downshift:
        edu_series = staff.get(edu_col, pd.Series(index=staff.index, dtype="object"))
        est = _expected_uplift_from_mix(edu_series, ma_pct=deg_ma_pct, phd_pct=deg_phd_pct)
        k = max(0.0, min(1.0, 1.0 - est))
    else:
        k = max(0.0, 1.0 - float(pre_degree_down_pct))
    base *= k

    # 3) degree multipliers (positive-only)
    mult = _degree_multiplier_positive(staff.get(edu_col, pd.Series(index=staff.index, dtype="object")),
                                       ma_pct=deg_ma_pct, phd_pct=deg_phd_pct)
    y = base * mult

    # 4) flat stipends
    pairs: list[tuple[str, float]] = []
    if prep_col and float(prep_bonus or 0) != 0:      pairs.append((prep_col, float(prep_bonus)))
    if skill_col and float(skill_bonus or 0) != 0:    pairs.append((skill_col, float(skill_bonus)))
    if leadership_col and float(leadership_bonus or 0) != 0: pairs.append((leadership_col, float(leadership_bonus)))
    y = _add_flag_bonuses(y, staff, pairs)

    return pd.Series(y, index=staff.index, name="MEDIAN_TARGET Salary")

def median_target_predict_capped(
    staff: pd.DataFrame,
    long: pd.DataFrame,
    *,
    band_targets: Dict[str, float],
    inflation: float = 0.0,
    max_salary: float = 100_000.0,
    years_col: str = "Years of Exp",
    edu_col: str = "Education Level",
    prep_col: str = "Prep",
    prep_bonus: float = 0.0,
    deg_ma_pct: float = 0.04,
    deg_phd_pct: float = 0.08,
    skill_col: Optional[str] = None,
    skill_bonus: float = 0.0,
    leadership_col: Optional[str] = None,
    leadership_bonus: float = 0.0,
    pre_degree_down_pct: float = 0.04,
    auto_downshift: bool = False,
) -> pd.Series:
    s = median_target_predict(
        staff, long,
        band_targets=band_targets,
        inflation=inflation,
        years_col=years_col,
        edu_col=edu_col,
        prep_col=prep_col,
        prep_bonus=prep_bonus,
        deg_ma_pct=deg_ma_pct,
        deg_phd_pct=deg_phd_pct,
        skill_col=skill_col,
        skill_bonus=skill_bonus,
        leadership_col=leadership_col,
        leadership_bonus=leadership_bonus,
        pre_degree_down_pct=pre_degree_down_pct,
        auto_downshift=auto_downshift,
    ).astype(float)
    s = np.minimum(s, float(max_salary))
    return pd.Series(s, index=s.index, name="MEDIAN_TARGET_CAP Salary")

# src/models/mean_targeted.py
def build_mt_band_targets(long: pd.DataFrame, targets: dict[str, float], inflation: float) -> dict[str, float]:
    """
    Return {band_label: base_y} for the overlay.
    `targets` should already be your per-band medians (pre- or post-inflation; you decide).
    """
    # if you store raw medians, inflate here:
    return {band: float(val) * (1.0 + float(inflation)) for band, val in targets.items()}