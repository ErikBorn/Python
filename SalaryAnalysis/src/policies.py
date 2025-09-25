# src/policies.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _coerce_boolish(s: pd.Series) -> pd.Series:
    """Turn various truthy/falsy values into booleans (1/0, True/False, yes/no strings)."""
    if s is None:
        return pd.Series(False, index=pd.RangeIndex(0))
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    x = s.astype(str).str.strip().str.lower()
    truthy = {"1","true","t","yes","y"}
    falsy  = {"0","false","f","no","n","", "nan", "none"}
    out = x.isin(truthy)
    out = out & (~x.isin(falsy))  # explicitly false stays false
    return out

def apply_outside_exp_cap(
    staff: pd.DataFrame,
    *,
    years_col: str = "Years of Exp",
    seniority_col: str = "Seniority",
    skill_flag_col: str | None = None,   # e.g., "Skill Endorsement" (1 means endorsed)
    cap: float = 10.0
) -> pd.DataFrame:
    """
    Policy:
      - Preserve original years in `tot_exp`.
      - Let outside = max(Years - Seniority, 0).
      - If outside <= cap: credit all outside.
      - If outside > cap: credit only `cap` UNLESS skill_flag==1/True -> then credit all.
      - New Years of Exp = Seniority + credited_outside.

    Returns the modified DataFrame (in-place safe).
    """
    df = staff.copy()

    # Stash original
    df["tot_exp"] = pd.to_numeric(df.get(years_col, 0), errors="coerce").fillna(0.0)

    yrs = pd.to_numeric(df.get(years_col, 0), errors="coerce").fillna(0.0).to_numpy()
    sen = pd.to_numeric(df.get(seniority_col, 0), errors="coerce").fillna(0.0).to_numpy()

    outside = np.maximum(yrs - sen, 0.0)

    # Skill endorsement (optional)
    if skill_flag_col and (skill_flag_col in df.columns):
        endorsed = _coerce_boolish(df[skill_flag_col]).to_numpy()
    else:
        endorsed = np.zeros_like(outside, dtype=bool)

    credited_outside = np.where(
        (outside > cap) & (~endorsed),
        cap,
        outside
    )

    new_years = sen + credited_outside
    df[years_col] = new_years

    return df