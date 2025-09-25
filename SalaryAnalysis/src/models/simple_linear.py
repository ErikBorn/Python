# src/models/simple_linear.py

from __future__ import annotations
import numpy as np
import pandas as pd
import re


# --- internal helpers ---------------------------------------------------------

_MA_RE  = re.compile(r"\b(MA|M\.A\.|MS|M\.S\.|MED|M\.ED)\b", flags=re.IGNORECASE)
_PHD_RE = re.compile(r"\b(PHD|PH\.D\.|EDD|ED\.D\.)\b",       flags=re.IGNORECASE)

def _num(series: pd.Series, default=0.0) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(default).to_numpy(dtype=float)


# --- public API ---------------------------------------------------------------

def simple_linear_predict(
    staff: pd.DataFrame,
    *,
    base: float,
    slope_per_year: float,
    ma_add: float = 0.0,
    phd_add: float = 0.0,
) -> pd.Series:
    """
    Simple linear model:

        salary = base + slope_per_year * (Years of Exp) + degree_adder

    Degree adders:
      - If Education Level indicates PhD (PhD/Ph.D/EdD), uses `phd_add`.
      - Else if indicates Masters (MA/MS/M.Ed), uses `ma_add`.
      - Else (BA/none), uses 0.
      - PhD overrides MA (no stacking).

    Returns:
      pd.Series aligned to `staff.index`, name = "Simple Linear Salary".
    """
    # Coerce years
    yrs = _num(staff.get("Years of Exp", pd.Series(index=staff.index, dtype="float64")), 0.0)

    # Degree tags
    edu = staff.get("Education Level", pd.Series("", index=staff.index)).astype(str)
    has_ma  = edu.str.contains(_MA_RE,  na=False)
    has_phd = edu.str.contains(_PHD_RE, na=False)

    deg_add = np.where(has_phd.to_numpy(), float(phd_add),
               np.where(has_ma.to_numpy(),  float(ma_add), 0.0))

    vals = float(base) + float(slope_per_year) * yrs + deg_add
    return pd.Series(vals, index=staff.index, name="Simple Linear Salary")