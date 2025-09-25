# src/models/nonlinear.py
import numpy as np
import pandas as pd
import re

# -------- helpers --------

def exp_cumulative(years, base_per_year, half_life):
    years = np.asarray(years, dtype=float)
    years = np.maximum(0.0, years)
    if half_life <= 0 or base_per_year == 0:
        return np.zeros_like(years)
    k = np.log(2.0) / float(half_life)
    return base_per_year * (1.0 - np.exp(-k * years)) / k

# robust degree parsing
_MA_PATTERNS  = re.compile(r"(?i)\b(?:MA|M\.A\.|MS|M\.S\.|MEd|M\.Ed)\b")
_PHD_PATTERNS = re.compile(r"(?i)\b(?:PhD|Ph\.D\.|EdD|Ed\.D\.)\b")

def degree_multiplier_series(edu: pd.Series, ma_pct: float, phd_pct: float, stack: bool=False) -> np.ndarray:
    s = (edu if edu is not None else pd.Series("", index=None)).astype(str).str.strip()
    has_ma  = s.str.contains(_MA_PATTERNS,  na=False)
    has_phd = s.str.contains(_PHD_PATTERNS, na=False)
    ma_mult  = 1.0 + float(ma_pct)
    phd_mult = 1.0 + float(phd_pct)
    if stack:
        return (np.where(has_ma,  ma_mult, 1.0) *
                np.where(has_phd, phd_mult, 1.0))
    return np.where(has_phd, phd_mult, np.where(has_ma, ma_mult, 1.0))

def total_years(yrs, sen, f_non_sen: float) -> np.ndarray:
    """Blend outside experience at fraction f_non_sen into a single 'total years'."""
    yrs = pd.to_numeric(yrs, errors="coerce").fillna(0.0).to_numpy()
    sen = pd.to_numeric(sen, errors="coerce").fillna(0.0).to_numpy()
    f   = float(f_non_sen)
    return np.maximum(0.0, sen + f * (yrs - sen))

# -------- main model --------

def nonlinear_predict(
    staff,
    *,
    base_salary,
    exp_base_per_year,
    exp_half_life_years,
    sen_base_per_year,
    sen_half_life_years,
    ma_pct,
    phd_pct,
    stack_degrees=False,
    w_skill=0.0,
    w_prep=0.0,
    w_knowledge=0.0,
    level_adders=None,
    aim_multiplier=1.0,
    f_non_sen=None,           # <-- now accepted
):
    """
    If f_non_sen is provided (e.g., 0.67), use blended 'total years' for the
    EXPERIENCE curve and degree growth; Seniority curve still uses raw 'Seniority'.
    """
    level_adders = level_adders or {"LS": 0.0, "MS": 0.0, "HS": 0.0}

    sf  = staff.copy()
    yrs = pd.to_numeric(sf.get("Years of Exp", 0), errors="coerce").fillna(0.0)
    sen = pd.to_numeric(sf.get("Seniority",   0), errors="coerce").fillna(0.0)

    yrs_eff = yrs if f_non_sen is None else total_years(yrs, sen, f_non_sen)

    exp_contrib = exp_cumulative(yrs_eff, exp_base_per_year, exp_half_life_years)
    sen_contrib = exp_cumulative(sen,     sen_base_per_year, sen_half_life_years)

    # level adders
    lvl = sf.get("Level", pd.Series(index=sf.index, dtype="object")).astype(str).str.upper().str.strip()
    lvl_add = (
        np.where(lvl.eq("LS"), level_adders.get("LS", 0.0), 0.0) +
        np.where(lvl.eq("MS"), level_adders.get("MS", 0.0), 0.0) +
        np.where(lvl.eq("HS"), level_adders.get("HS", 0.0), 0.0)
    )

    # optional linear adds
    skill = pd.to_numeric(sf.get("Skill Rating",      0), errors="coerce").fillna(0.0).to_numpy()
    prep  = pd.to_numeric(sf.get("Prep Rating",       0), errors="coerce").fillna(0.0).to_numpy()
    know  = pd.to_numeric(sf.get("Knowledge Rating",  0), errors="coerce").fillna(0.0).to_numpy()
    linear_adds = w_skill*skill + w_prep*prep + w_knowledge*know + lvl_add

    base_curve = base_salary + exp_contrib + sen_contrib + linear_adds

    # degree multiplier applied to the base curve
    edu = sf.get("Education Level", pd.Series(index=sf.index, dtype="object"))
    deg_mult = degree_multiplier_series(edu, ma_pct=ma_pct, phd_pct=phd_pct, stack=stack_degrees)

    out = aim_multiplier * (base_curve * deg_mult)
    return pd.Series(out, index=sf.index, name="Model NL Salary")