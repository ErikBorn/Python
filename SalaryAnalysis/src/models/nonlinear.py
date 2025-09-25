# src/models/nonlinear.py

from __future__ import annotations
import numpy as np
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------

def _num(series_or_val, default=0.0) -> np.ndarray:
    """
    Coerce a pd.Series or scalar to float ndarray with NaNs -> default.
    """
    if isinstance(series_or_val, pd.Series):
        arr = pd.to_numeric(series_or_val, errors="coerce").fillna(default).to_numpy(dtype=float)
    else:
        try:
            arr = np.asarray(series_or_val, dtype=float)
        except Exception:
            arr = np.asarray(default, dtype=float)
    return arr


def total_years(yrs, sen, f_non_sen: float) -> np.ndarray:
    """
    'Total years' = seniority years (full weight) + non-seniority years * f_non_sen.
      yrs: total experience years (anywhere)
      sen: years at the school
      f_non_sen: fraction for non-seniority years (e.g., 0.67)
    """
    yrs = _num(yrs, 0.0)
    sen = _num(sen, 0.0)
    f = float(f_non_sen)
    non_sen = np.maximum(yrs - sen, 0.0)
    return np.maximum(sen, 0.0) + f * non_sen


# ----------------------------
# Core model components
# ----------------------------

def exp_cumulative(years, base_per_year, half_life) -> np.ndarray:
    """
    Continuous-time cumulative with exponentially decaying marginal:
      marginal(t)  = base_per_year * exp(-k t),  k = ln(2)/half_life
      cumulative(y)= base_per_year * (1 - exp(-k y)) / k
    """
    years = _num(years, 0.0)
    base_per_year = float(base_per_year)
    half_life = float(half_life)

    if half_life <= 0.0 or base_per_year == 0.0:
        return np.zeros_like(years, dtype=float)
    k = np.log(2.0) / half_life
    return base_per_year * (1.0 - np.exp(-k * np.maximum(years, 0.0))) / k


# Robust degree pattern checks (vectorized, tolerant to None/NaN)
def _has_ma(edu: pd.Series) -> np.ndarray:
    s = edu.fillna("").astype(str).str.upper()
    # Common MA/MS/M.Ed variants
    return s.str.contains(r"\b(MA|M\.A\.|MS|M\.S\.|MED|M\.ED)\b", regex=True).to_numpy()


def _has_phd(edu: pd.Series) -> np.ndarray:
    s = edu.fillna("").astype(str).str.upper()
    # Common PhD/Ph.D/EdD variants
    return s.str.contains(r"\b(PHD|PH\.D\.|EDD|ED\.D\.)\b", regex=True).to_numpy()


def degree_multiplier_series(edu: pd.Series, ma_pct: float, phd_pct: float, stack: bool=False) -> np.ndarray:
    """
    Degree multiplier applied multiplicatively to the whole salary:
      BA: 1.0
      MA: (1 + ma_pct)
      PhD: (1 + phd_pct)
      If stack=True, MA and PhD multipliers multiply (rare; usually False).
      If stack=False, PhD overrides MA.
    """
    ma_mult = 1.0 + float(ma_pct)
    phd_mult = 1.0 + float(phd_pct)

    has_ma = _has_ma(edu)
    has_phd = _has_phd(edu)

    if stack:
        # Multiply applicable multipliers (BA = 1.0)
        return np.where(has_ma, ma_mult, 1.0) * np.where(has_phd, phd_mult, 1.0)
    else:
        # Highest only: PhD overrides MA
        return np.where(has_phd, phd_mult, np.where(has_ma, ma_mult, 1.0))


# ----------------------------
# Main predictor
# ----------------------------

def nonlinear_predict(
    staff: pd.DataFrame,
    *,
    base_salary: float,
    exp_base_per_year: float,
    exp_half_life_years: float,
    sen_base_per_year: float,
    sen_half_life_years: float,
    ma_pct: float,
    phd_pct: float,
    stack_degrees: bool,
    w_skill: float,
    w_prep: float,
    w_knowledge: float,
    level_adders: dict,          # e.g., {"LS": 0.0, "MS": 0.0, "HS": 0.0}
    aim_multiplier: float,
    # Optional: pass f_non_sen if you want to compute a 'total years' track separately upstream.
    # Here we keep exp based on Years of Exp and sen based on Seniority, matching your prior notebooks.
) -> pd.Series:
    """
    Vectorized nonlinear salary prediction:

      Salary = aim_multiplier * degree_mult * (
                  base_salary
                + exp_cumulative(Years of Exp; exp_base_per_year, exp_half_life_years)
                + exp_cumulative(Seniority;  sen_base_per_year,  sen_half_life_years)
                + linear_adds(Skill, Prep, Knowledge, Level adders)
               )

    Notes:
      - Degree multipliers apply multiplicatively to the whole bracketed sum.
      - If you prefer degree bumps as additive amounts, replace the multiplier with adds.
      - Expects columns:
          'Years of Exp', 'Seniority', 'Education Level',
          'Skill Rating', 'Prep Rating', 'Knowledge Rating', 'Level'
        Missing columns are treated as zeros/empty strings where reasonable.
    """
    # Pull columns (tolerant to missing; default to 0/empty)
    yrs = staff.get("Years of Exp", pd.Series(index=staff.index, dtype="float64"))
    sen = staff.get("Seniority", pd.Series(index=staff.index, dtype="float64"))
    edu = staff.get("Education Level", pd.Series(index=staff.index, dtype="object"))
    skill = staff.get("Skill Rating", pd.Series(index=staff.index, dtype="float64"))
    prep = staff.get("Prep Rating", pd.Series(index=staff.index, dtype="float64"))
    knowledge = staff.get("Knowledge Rating", pd.Series(index=staff.index, dtype="float64"))
    lvl = staff.get("Level", pd.Series(index=staff.index, dtype="object"))

    yrs_arr = _num(yrs, 0.0)
    sen_arr = _num(sen, 0.0)

    # Core contributions
    exp_contrib = exp_cumulative(yrs_arr, exp_base_per_year, exp_half_life_years)
    sen_contrib = exp_cumulative(sen_arr, sen_base_per_year, sen_half_life_years)

    # Linear adds
    lin_adds = (
        float(w_skill) * _num(skill, 0.0) +
        float(w_prep) * _num(prep, 0.0) +
        float(w_knowledge) * _num(knowledge, 0.0)
    )

    # Level adders
    lvl_series = lvl.fillna("").astype(str).str.upper().str.strip()
    add_ls = float(level_adders.get("LS", 0.0))
    add_ms = float(level_adders.get("MS", 0.0))
    add_hs = float(level_adders.get("HS", 0.0))
    lvl_add = (
        np.where(lvl_series.eq("LS"), add_ls, 0.0) +
        np.where(lvl_series.eq("MS"), add_ms, 0.0) +
        np.where(lvl_series.eq("HS"), add_hs, 0.0)
    )

    # Inside-sum
    inside = float(base_salary) + exp_contrib + sen_contrib + lin_adds + lvl_add

    # Degree multiplier on top
    deg_mult = degree_multiplier_series(edu, ma_pct=float(ma_pct), phd_pct=float(phd_pct), stack=bool(stack_degrees))

    # Aim multiplier to scale whole curve
    out = float(aim_multiplier) * deg_mult * inside

    # Return as Series aligned to staff
    return pd.Series(out, index=staff.index, name="Model NL Salary")