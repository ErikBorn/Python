# src/cohort.py

from __future__ import annotations
import re
from typing import List, Tuple, Union
import numpy as np
import pandas as pd


def band_range_closed(label: str) -> Tuple[float, float]:
    """
    Parse cohort band labels into CLOSED numeric spans.

    Accepts forms like:
      "0-5 yrs", "6-10 yrs", "11-15 yrs", "36+ yrs", "41+ Years" (case-insensitive)

    Returns:
      (start, end) as floats with end == +inf for the open-ended band.
      Examples:
        "0-5 yrs"   -> (0.0, 5.0)
        "6-10 yrs"  -> (6.0, 10.0)
        "41+ Years" -> (41.0, +inf)
    """
    s = str(label).strip()

    m_range = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*(?:yrs|years)?\s*$", s, flags=re.IGNORECASE)
    if m_range:
        a, b = int(m_range.group(1)), int(m_range.group(2))
        return float(a), float(b)

    m_plus = re.match(r"^\s*(\d+)\s*\+\s*(?:yrs|years)?\s*$", s, flags=re.IGNORECASE)
    if m_plus:
        a = int(m_plus.group(1))
        return float(a), np.inf

    raise ValueError(f"Unrecognized band label: {label!r}")


def build_band_bins_closed(long: pd.DataFrame) -> Tuple[List[Tuple[float, float, str]], List[str]]:
    """
    From a tidy cohort table `long` (with column 'experience_band'),
    build CLOSED-interval bins ordered by starting year.

    Returns:
      bins:  list of (start, end, label)
      order: list of band labels in sorted order (by start)
    """
    if "experience_band" not in long.columns:
        raise ValueError("long must include column 'experience_band'")

    bands = sorted(
        pd.Series(long["experience_band"].dropna().unique(), dtype="object").tolist(),
        key=lambda b: band_range_closed(b)[0],
    )
    bins: List[Tuple[float, float, str]] = []
    for b in bands:
        start, end = band_range_closed(b)
        bins.append((start, end, b))
    return bins, bands


def assign_band_from_years_closed(y: float, bins: List[Tuple[float, float, str]]) -> Union[str, float]:
    """
    Map a years-of-experience value into a CLOSED band interval.

    Rule: place y where start <= y <= end (end may be +inf).
    Returns the band label, or np.nan if no match / y is NaN.
    """
    if pd.isna(y):
        return np.nan
    yf = float(y)
    for start, end, label in bins:
        if np.isfinite(end):
            if (yf >= start) and (yf <= end):
                return label
        else:
            if yf >= start:
                return label
    return np.nan


def interpolate_salary_strict(long: pd.DataFrame, band: str, target_percentile: float) -> float:
    """
    Interpolate salary within a single band over cohort percentiles.

    - Uses ONLY rows from `long` where experience_band == band.
    - Expects columns: 'percentile' (numeric, e.g., 10/25/50/75/90) and 'salary' (numeric).
    - If target_percentile exactly matches a cohort point, returns that exact salary.
    - Otherwise performs linear interpolation between the nearest cohort points.
    - Clamps target_percentile into [min(xs), max(xs)].

    Returns:
      float salary (np.nan if no data for that band).
    """
    need = {"experience_band", "percentile", "salary"}
    if not need.issubset(long.columns):
        missing = need - set(long.columns)
        raise ValueError(f"`long` is missing required columns: {missing}")

    sub = (
        long.loc[long["experience_band"] == band, ["percentile", "salary"]]
            .dropna(subset=["percentile", "salary"])
            .astype({"percentile": "float64", "salary": "float64"})
            .sort_values("percentile")
    )

    if sub.empty:
        return np.nan

    xs = sub["percentile"].to_numpy()
    ys = sub["salary"].to_numpy()

    # Clamp target into the supported percentile range
    t = float(np.clip(target_percentile, xs.min(), xs.max()))

    # If t exactly hits a known cohort percentile, return the exact table value
    hit = np.isclose(t, xs)
    if hit.any():
        return float(ys[hit.argmax()])

    # Otherwise interpolate
    return float(np.interp(t, xs, ys))