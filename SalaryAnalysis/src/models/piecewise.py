# src/models/piecewise.py

from __future__ import annotations
from typing import List, Dict, Callable, Iterable
import numpy as np
import pandas as pd

from .nonlinear import degree_multiplier_series
# at top
from typing import List, Dict, Callable, Iterable, Optional


def _round_to(x: float, base: int) -> float:
    if base <= 0:
        return float(x)
    return float(base * round(float(x) / base))


def derive_pw_bands_ols(
    nl_baseline_fn: Callable[[np.ndarray], np.ndarray],
    edges: List[int],
    round_base: int = 100,
    round_step: int = 50,
    enforce_continuity: bool = True,
    shrink_to_total: float | None = None,
) -> List[Dict]:
    """
    Derive a BA piecewise-linear schedule that approximates a nonlinear baseline.

    Parameters
    ----------
    nl_baseline_fn
        Function mapping TotalYears -> BA baseline salary (no degree multipliers).
        Must accept a numpy array and return a numpy array of same shape.
    edges
        Sorted list of band boundaries in *Total Years* units, e.g. [0, 5, 10, 15, 20, 25, 30, 35, 40].
        Produces bands [0,5], [5,10], ..., [35,40], and a final open-ended band [40, +inf).
        If you do NOT want an open-ended band, include the last boundary and set `edges_open_end=False`
        by simply post-editing the returned dict (TotalYears_End of last row) if desired.
    round_base
        Round each band's starting BA value to this nearest integer multiple (e.g., $100).
    round_step
        Round each band's yearly step to this nearest integer multiple (e.g., $50).
    enforce_continuity
        If True, adjusts each band's rounded start so the piecewise curve is continuous at band edges.
        (Keeps each band's step; shifts the next band's start to meet the previous band's endpoint.)
    shrink_to_total
        If provided (e.g., 0.98), scales ALL bands (both starts and steps) by a single factor so that
        the average piecewise value across the calibration grid matches `shrink_to_total *` the baseline average.

    Returns
    -------
    List[Dict]
        Each dict has:
          - "TotalYears_Start"
          - "TotalYears_End"  (last band has np.inf)
          - "BA_Start"
          - "BA_Step_per_Year"
    """
    if len(edges) < 2:
        raise ValueError("`edges` must contain at least two boundaries.")

    edges = sorted(int(e) for e in edges)
    # We'll fit up to the last finite edge; beyond that, we still create an open-ended band
    t_min = edges[0]
    t_max = edges[-1]

    # Dense grid for fitting (smooths noise and gives stable OLS)
    # Use 0.1 year resolution across [t_min, t_max]
    t_grid = np.linspace(t_min, t_max, int((t_max - t_min) * 10) + 1)
    y_grid = np.asarray(nl_baseline_fn(t_grid), dtype=float)
    if y_grid.shape != t_grid.shape:
        raise ValueError("`nl_baseline_fn` must return array of same shape as input.")

    bands: List[Dict] = []

    # OLS fit per segment: y ≈ a + b*(t - start), using points within [start, end]
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (t_grid >= start) & (t_grid <= end)
        if not np.any(mask):
            # Fallback: evaluate at endpoints if the grid is too sparse
            mask = np.isin(t_grid, [start, end])

        x = (t_grid[mask] - start)  # predictor centered at band start
        y = y_grid[mask]

        # Design matrix: [1, x]
        X = np.column_stack([np.ones_like(x), x])
        # OLS via normal equations (robust to small x)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        a_hat, b_hat = float(beta[0]), float(beta[1])

        # Round start & step
        a_round = _round_to(a_hat, round_base)
        b_round = _round_to(b_hat, round_step)

        bands.append({
            "TotalYears_Start": float(start),
            "TotalYears_End":   float(end),
            "BA_Start":         float(a_round),
            "BA_Step_per_Year": float(b_round),
        })

    # Add open-ended last band [edges[-1], +inf)
    # Fit its slope from a short tail beyond t_max; extrapolate baseline slightly
    tail_start = float(edges[-1])
    tail_end = tail_start + 5.0  # use a 5-year synthetic tail for slope estimate
    t_tail = np.linspace(tail_start, tail_end, 51)
    y_tail = np.asarray(nl_baseline_fn(t_tail), dtype=float)

    x_tail = (t_tail - tail_start)
    X_tail = np.column_stack([np.ones_like(x_tail), x_tail])
    beta_tail, *_ = np.linalg.lstsq(X_tail, y_tail, rcond=None)
    a_tail, b_tail = float(beta_tail[0]), float(beta_tail[1])

    a_tail_round = _round_to(a_tail, round_base)
    b_tail_round = _round_to(b_tail, round_step)

    bands.append({
        "TotalYears_Start": float(tail_start),
        "TotalYears_End":   float(np.inf),
        "BA_Start":         float(a_tail_round),
        "BA_Step_per_Year": float(b_tail_round),
    })

    # Optionally enforce continuity at the edges *after* rounding:
    if enforce_continuity and len(bands) >= 2:
        for i in range(1, len(bands)):
            prev = bands[i - 1]
            cur  = bands[i]
            # Compute previous endpoint value at its end:
            prev_len = (prev["TotalYears_End"] - prev["TotalYears_Start"]) if np.isfinite(prev["TotalYears_End"]) else 0.0
            prev_end_val = prev["BA_Start"] + prev["BA_Step_per_Year"] * prev_len
            # Force current start to equal previous end value
            cur["BA_Start"] = float(_round_to(prev_end_val, round_base))

    # Optional global shrink/scale to match an overall target factor
    if shrink_to_total is not None:
        # Compare means across the calibration grid (within [t_min, t_max])
        pw_vals = _eval_piecewise_on_grid(t_grid, bands)
        baseline_mean = float(np.nanmean(y_grid))
        pw_mean = float(np.nanmean(pw_vals))
        if pw_mean > 0 and baseline_mean > 0:
            # Scale so that pw_mean == shrink_to_total * baseline_mean
            target_mean = float(shrink_to_total) * baseline_mean
            k = target_mean / pw_mean
            for b in bands:
                b["BA_Start"]         = float(b["BA_Start"] * k)
                b["BA_Step_per_Year"] = float(b["BA_Step_per_Year"] * k)

    return bands


def _eval_piecewise_on_grid(t: np.ndarray, bands: List[Dict]) -> np.ndarray:
    """Evaluate BA piecewise curve (no degree multipliers) on an array of TotalYears."""
    out = np.full_like(t, np.nan, dtype=float)
    for band in bands:
        a = band["TotalYears_Start"]
        b = band["TotalYears_End"]
        base = band["BA_Start"]
        step = band["BA_Step_per_Year"]

        if np.isfinite(b):
            m = (t >= a) & (t <= b)
        else:
            m = (t >= a)

        out[m] = base + (t[m] - a) * step
    return out


def pw_predict(
    total_years: np.ndarray,
    edu: pd.Series,
    pw_bands: List[Dict],
    ma_pct: float,
    phd_pct: float,
    stack_degrees: bool = False,
    *,
    prep_flag: Optional[pd.Series] = None,   # NEW
    prep_bonus: float = 2500.0,              # NEW
) -> np.ndarray:
    """
    Predict salaries from a piecewise BA schedule + degree multipliers (+ optional flat Prep bonus).
    """
    t = np.asarray(total_years, dtype=float)
    out = np.full_like(t, np.nan, dtype=float)

    # Evaluate BA piecewise baseline per person
    for band in pw_bands:
        a = band["TotalYears_Start"]
        b = band["TotalYears_End"]
        base = band["BA_Start"]
        step = band["BA_Step_per_Year"]
        m = (t >= a) if not np.isfinite(b) else ((t >= a) & (t <= b))
        out[m] = base + (t[m] - a) * step

    # Degree multipliers (multiplicative)
    mult = degree_multiplier_series(edu, ma_pct=ma_pct, phd_pct=phd_pct, stack=stack_degrees)
    out = out * mult

    # Flat Prep bonus (additive)
    if prep_flag is not None:
        pf = pd.to_numeric(prep_flag, errors="coerce").fillna(0).to_numpy()
        out = out + np.where(pf == 1, float(prep_bonus), 0.0)

    return out