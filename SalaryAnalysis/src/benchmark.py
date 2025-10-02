# src/benchmark.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from .metrics import achieved_percentiles, band_medians_table, per_band_median_percentiles

__all__ = ["write_benchmark_tables"]

def _money(x):  # tiny helper for pretty CSV
    return "" if pd.isna(x) else f"${int(x):,}"

def _pick_cols_for_label(staff: pd.DataFrame, label: str) -> dict[str, str]:
    """
    Build the columns mapping for the requested cohort label.
    label: 'state' or 'local' (case-insensitive).
    Falls back to the old unprefixed names if needed.
    """
    lbl = str(label).strip().lower()
    if lbl not in {"state", "local"}:
        raise ValueError("label must be 'state' or 'local'")

    # Preferred: new prefixed columns produced by run_pipeline2
    prefix = "CONS_St" if lbl == "state" else "CONS_Lo"
    prefixed = {
        "Real":          "25-26 Salary",
        f"{prefix}_CAP":        f"{prefix}_CAP Salary",
        f"{prefix}_Cap_Real":   f"{prefix}_Cap_Real",
        "COLA 2%":       "All +2% Plan Salary",
    }

    # If the prefixed columns exist, use them.
    if all(c in staff.columns for c in prefixed.values()):
        return prefixed

    # Otherwise, fall back to legacy, single-cohort names
    legacy = {
        "Real":          "25-26 Salary",
        "CONS":          "CONS Salary",
        "CONS_CAP":      "CONS_CAP Salary",
        "CONS_Cap_Real": "CONS_Cap_Real",
        "COLA 2%":       "All +2% Plan Salary",
    }
    if all(c in staff.columns for c in legacy.values()):
        return legacy

    # If we get here, give a helpful error.
    missing_pref = [c for c in prefixed.values() if c not in staff.columns]
    missing_legacy = [c for c in legacy.values() if c not in staff.columns]
    raise ValueError(
        "Missing required columns for benchmarking.\n"
        f"Tried prefixed set ({label}): {missing_pref}\n"
        f"Tried legacy set: {missing_legacy}"
    )

def write_benchmark_tables(
    long_cohort: pd.DataFrame,
    label: str,
    out_dir: str,
    staff: pd.DataFrame,
    # Accept (and currently ignore) extra params from the pipeline so calls don’t break:
    target_percentile: float | None = None,
    inflation: float | None = None,
) -> None:
    """
    Write benchmarking CSVs (per-band median percentiles, achieved percentiles,
    and per-band salary medians) for the given cohort `long_cohort`.

    Files written under {out_dir}/tables:
      - {label}_band_percentile_medians.csv
      - {label}_achieved_percentiles.csv
      - {label}_band_medians.csv
    """
    os.makedirs(f"{out_dir}/tables", exist_ok=True)

    # Choose the correct columns for this label (state/local or legacy)
    cols_dict = _pick_cols_for_label(staff, label)

    # 1) Per-band median percentiles (how each series sits vs the cohort by band)
    band_pct_med = per_band_median_percentiles(
        long_cohort,
        staff,
        cols_dict=cols_dict,
        years_col="Years of Exp",
        decimals=1,
    )
    band_pct_med.to_csv(f"{out_dir}/tables/{label}_band_percentile_medians.csv")

    # 2) Achieved percentiles (overall mean/median %ile)
    pct = achieved_percentiles(
        long_cohort,
        staff,
        salary_cols=list(cols_dict.values()),
        years_col="Years of Exp",
        labels=list(cols_dict.keys()),
    ).round(1)
    pct.to_csv(f"{out_dir}/tables/{label}_achieved_percentiles.csv")

    # 3) Per-band salary medians
    band_tbl = band_medians_table(long_cohort, staff, salary_cols=list(cols_dict.values()))

    # Pretty CSV copy: salaries rounded to $10; headcount stays integer
    band_tbl_num = band_tbl.copy()
    mask = (band_tbl_num.index != "Headcount") if "Headcount" in band_tbl_num.index else slice(None)
    band_tbl_num.loc[mask] = band_tbl_num.loc[mask].round(-1)

    df_save = band_tbl_num.copy().astype(object)
    df_save.loc[mask] = np.vectorize(_money)(df_save.loc[mask].to_numpy(dtype=float))
    if "Headcount" in df_save.index:
        df_save.loc["Headcount"] = band_tbl_num.loc["Headcount"].round(0).astype(int).astype(str)

    df_save.to_csv(f"{out_dir}/tables/{label}_band_medians.csv")