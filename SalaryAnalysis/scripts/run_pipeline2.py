# scripts/run_pipeline.py  (consultant-only, dual-cohort: State vs Local)

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import os
import yaml
import pandas as pd
import numpy as np

from src.io import load_staff, load_cohort
from src.policies import apply_outside_exp_cap
from src.models.consultant import consultant_predict, consultant_predict_capped
from src.planning import plan_salary_column, plan_costs
from src.metrics import achieved_percentiles
from src.viz import (
    plot_scatter_for_cohort,
    plot_scatter_models_interactive,
    cons_cap_overview_html,
)
from src.share import write_cohort_aligned_calculator_html
from src.benchmark import write_benchmark_tables  # <- your new helper

# ---------- small helpers ----------------------------------------------------

def _mean_percentile_against(long_cohort, staff, planned_col) -> float:
    """Mean achieved percentile of `planned_col` against `long_cohort`."""
    df = achieved_percentiles(long_cohort, staff, salary_cols=[planned_col], years_col="Years of Exp")
    s = df.squeeze()
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return float(s.get("Mean %ile", np.nan))


def main(cfg):
    # --- Paths & IO -----------------------------------------------------------
    paths = cfg["paths"]
    out   = paths["out_dir"]
    os.makedirs(out, exist_ok=True)
    os.makedirs(f"{out}/tables", exist_ok=True)
    os.makedirs(f"{out}/figures", exist_ok=True)

    # Staff
    staff = load_staff(paths["staff_csv"])

    # Cohorts: STATE (required) + LOCAL (optional)
    long_st = load_cohort(paths["cohort_csv"])  # state / default cohort
    lo_cfg  = cfg.get("local_cohort", {}) or {}
    use_lo  = bool(lo_cfg.get("enabled", False))
    long_lo = load_cohort(lo_cfg.get("csv")) if use_lo and lo_cfg.get("csv") else None

    # --- Experience policy (credited years) applied EARLY ---------------------
    xp_cfg = cfg.get("experience_policy", {})
    staff  = apply_outside_exp_cap(
        staff,
        years_col="Years of Exp",
        seniority_col="Seniority",
        skill_flag_col=xp_cfg.get("skill_endorsement_col"),  # e.g., "Skill Endorsement"
        cap=float(xp_cfg.get("cap_outside_years", 10)),
    )

    # --- Consultant model knobs ----------------------------------------------
    bump = cfg["planning"]["bump"]   # e.g., 0.02
    tol  = cfg["planning"]["tol"]
    prep_series_name = cfg.get("prep", {}).get("flag_col", "Prep")
    prep_bonus       = cfg.get("prep", {}).get("bonus", 2500.0)

    cns     = cfg["consultant"]
    cap_cfg = cfg.get("consultant_capped", {})

    # Allow Local to use a different target percentile; fall back to State
    tp_st = float(cns["target_percentile"])
    tp_lo = float(lo_cfg.get("target_percentile", tp_st)) if use_lo else None

    # ---------- run consultant against a given cohort -------------------------
    def run_consultant(prefix: str, long_df: pd.DataFrame, target_percentile: float):
        """
        Build prefixed columns using the given cohort & percentile:
          {prefix} Salary
          {prefix}_CAP Salary
          {prefix}_Cap_Real
          {prefix}_CAP Plan Salary  (== real, just named for clarity)
        Returns (base_col, cap_col, real_col, plan_col)
        """
        base_col = f"{prefix} Salary"
        cap_col  = f"{prefix}_CAP Salary"
        real_col = f"{prefix}_Cap_Real"
        plan_col = f"{prefix}_CAP Plan Salary"

        # base (positive-only degree bumps + pre-degree downshift)
        staff[base_col] = consultant_predict(
            staff, long_df,
            target_percentile     = target_percentile,
            inflation             = cns["inflation"],
            base_start            = cns.get("base_start"),
            deg_ma_pct            = cns.get("deg_ma_pct", 0.02),
            deg_phd_pct           = cns.get("deg_phd_pct", 0.04),
            pre_degree_down_pct   = cns.get("pre_degree_down_pct", 0.03),
            auto_downshift        = cns.get("auto_downshift", False),
            prep_col              = prep_series_name,
            prep_bonus            = prep_bonus,
        )

        # capped
        staff[cap_col] = consultant_predict_capped(
            staff, long_df,
            target_percentile     = cap_cfg.get("target_percentile", target_percentile),
            inflation             = cap_cfg.get("inflation", cns["inflation"]),
            base_start            = cap_cfg.get("base_start", cns.get("base_start")),
            deg_ma_pct            = cap_cfg.get("deg_ma_pct", cns.get("deg_ma_pct", 0.02)),
            deg_phd_pct           = cap_cfg.get("deg_phd_pct", cns.get("deg_phd_pct", 0.04)),
            pre_degree_down_pct   = cap_cfg.get("pre_degree_down_pct", cns.get("pre_degree_down_pct", 0.03)),
            auto_downshift        = cap_cfg.get("auto_downshift", cns.get("auto_downshift", False)),
            max_salary            = cap_cfg.get("max_salary", 100_000),
            years_col             = "Years of Exp",
            edu_col               = "Education Level",
            prep_col              = prep_series_name,
            prep_bonus            = prep_bonus,
        )

        # realized version (max of model vs COLA floor)
        staff[real_col] = plan_salary_column(
            staff,
            real_col="25-26 Salary",
            model_col=cap_col,
            out_col=real_col,
            bump=bump,
            tol=tol,
        )

        # keep a plan-named copy (handy for summary wiring)
        staff[plan_col] = staff[real_col]
        return base_col, cap_col, real_col, plan_col

    # Run for State (required)
    st_base, st_cap, st_real, st_plan = run_consultant("CONS_St", long_st, tp_st)

    # Run for Local (optional)
    if use_lo and long_lo is not None:
        lo_base, lo_cap, lo_real, lo_plan = run_consultant("CONS_Lo", long_lo, tp_lo)

    # Simple global “+2%”
    staff["All +2% Plan Salary"] = pd.to_numeric(staff["25-26 Salary"], errors="coerce") * (1.0 + bump)

    # --- BENCHMARK TABLES (write once per cohort) -----------------------------
    # (These replace the old per_band_median_percentiles / band_medians_table blocks.)
    write_benchmark_tables(
        long_st, "state", out, staff,
        target_percentile=tp_st,
        inflation=cns["inflation"]
    )
    if use_lo and long_lo is not None:
        write_benchmark_tables(
            long_lo, "local", out, staff,
            target_percentile=tp_lo,
            inflation=cns["inflation"]
        )

    # --- Interactive scatter (bands + calibrated summaries) -------------------
    real_hc = cfg.get("planning", {}).get("faculty_size")

    def _model_summary(planned_col):
        c = plan_costs(
            staff, real_col="25-26 Salary", planned_col=planned_col,
            bump=bump, real_headcount=real_hc
        )
        mean_st = _mean_percentile_against(long_st, staff, planned_col)
        mean_lo = _mean_percentile_against(long_lo, staff, planned_col) if use_lo and long_lo is not None else None
        return {
            "total_cost": c["Total Cost"],
            "mean_pct_state": mean_st,
            "mean_pct_local": mean_lo,
            "num_raised": c["Num Raised"],
        }

    models_summary = {
        "CONS_St_CAP": _model_summary(st_plan),
    
    }
    if use_lo and long_lo is not None:
        models_summary["CONS_Lo_CAP"] = _model_summary(lo_plan)
    models_summary["COLA 2%"] = _model_summary("All +2% Plan Salary")

    summary_info = {
        "params": {
            "target_percentile_state": tp_st,
            "target_percentile_local": tp_lo if use_lo else None,
            "target_inflation":        cns["inflation"],
            "cola":                    bump,
        },
        "models": models_summary,
    }

    # For the overlay bands we’ll draw STATE; switch to long_lo if you prefer.
    # common point set
    cols_dict_state = {
        "Real":               "25-26 Salary",
        "CONS_St_CAP":        st_cap,
        "CONS_St_Cap_Real":   st_real,
        "COLA 2%":            "All +2% Plan Salary",
    }
    cols_dict_local = {
        "Real":               "25-26 Salary",
        "CONS_Lo_CAP":        lo_cap,
        "CONS_Lo_Cap_Real":   lo_real,
        "COLA 2%":            "All +2% Plan Salary",
    }

    # STATE interactive
    plot_scatter_for_cohort(
        staff, "Years of Exp", cols_dict_state,
        long=long_st,
        cohort_label="State cohort",
        path_html=f"{out}/figures/scatter_interactive_state.html",
        target_percentile=cns.get("target_percentile", tp_st),
        inflation=cns["inflation"],
        plus_minus_pct=cfg["plots"]["plus_minus_pct"],
        summary_info=summary_info,  # if you want the same box
        hover_cols=["Employee","ID","Seniority","Education Level","Level","Category"],
    )

    # LOCAL interactive (only if you have it enabled/loaded)
    if long_lo is not None:
        plot_scatter_for_cohort(
            staff, "Years of Exp", cols_dict_local,
            long=long_lo,
            cohort_label="Local cohort",
            path_html=f"{out}/figures/scatter_interactive_local.html",
            target_percentile=cns.get("target_percentile", tp_lo),
            inflation=cns["inflation"],
            plus_minus_pct=cfg["plots"]["plus_minus_pct"],
            summary_info=summary_info,
            hover_cols=["Employee","ID","Seniority","Education Level","Level","Category"],
        )

    # --- Consultant Cap overview (one-pager; STATE) ---------------------------
    cons_cap_overview_html(
        long_st,
        target_percentile = cap_cfg.get("target_percentile", tp_st),
        inflation         = cap_cfg.get("inflation",         cns["inflation"]),
        deg_ma_pct        = cap_cfg.get("deg_ma_pct",        cns.get("deg_ma_pct", 0.02)),
        deg_phd_pct       = cap_cfg.get("deg_phd_pct",       cns.get("deg_phd_pct", 0.04)),
        prep_bonus        = cfg["planning"].get("prep_bonus", 2500.0),
        max_salary        = cap_cfg.get("max_salary", 100_000),
        cola_floor        = bump,
        outside_cap_years = cfg.get("years_policy", {}).get("outside_cap_years", 10),
        skill_extra_years = cfg.get("years_policy", {}).get("skill_extra_years", 5),
        skill_col_name    = cfg.get("years_policy", {}).get("skill_col_name", "Skill"),
        base_start        = cns.get("base_start"),
        path_html         = f"{out}/figures/cons_cap_overview.html",
        example           = dict(years=18.0, seniority=3.0, degree="MA", prep=1, skill=0, current=70_000),
    )

    # --- Minimal sharing CSV --------------------------------------------------
    CURRENT_COL = "25-26 Salary"
    cols_all = list(staff.columns)
    base_cols = cols_all[: cols_all.index(CURRENT_COL) + 1] if CURRENT_COL in cols_all else cols_all

    keep = [
        st_cap, st_plan, "All +2% Plan Salary",
        "Hire Date", "Eth", "Gender", "Years of Exp", "Seniority",
        "Education Level", "Skill Rating", "Knowledge Rating", "Prep Rating",
        "Level", "Category",
    ]
    if use_lo and long_lo is not None:
        keep = [st_cap, st_plan, lo_cap, lo_plan, "All +2% Plan Salary",
                "Hire Date", "Eth", "Gender", "Years of Exp", "Seniority",
                "Education Level", "Skill Rating", "Knowledge Rating", "Prep Rating",
                "Level", "Category"]

    # Deltas (State)
    staff["CCP Increase (St)"] = pd.to_numeric(staff.get(st_plan), errors="coerce") - pd.to_numeric(staff.get(CURRENT_COL), errors="coerce")
    staff["CCP Increase % (St)"] = np.where(
        pd.to_numeric(staff.get(CURRENT_COL), errors="coerce") > 0,
        staff["CCP Increase (St)"] / pd.to_numeric(staff.get(CURRENT_COL), errors="coerce"),
        np.nan,
    )
    # Deltas (Local) if present
    extra_cols = ["CCP Increase (St)", "CCP Increase % (St)"]
    if use_lo and long_lo is not None:
        staff["CCP Increase (Lo)"] = pd.to_numeric(staff.get(lo_plan), errors="coerce") - pd.to_numeric(staff.get(CURRENT_COL), errors="coerce")
        staff["CCP Increase % (Lo)"] = np.where(
            pd.to_numeric(staff.get(CURRENT_COL), errors="coerce") > 0,
            staff["CCP Increase (Lo)"] / pd.to_numeric(staff.get(CURRENT_COL), errors="coerce"),
            np.nan,
        )
        extra_cols += ["CCP Increase (Lo)", "CCP Increase % (Lo)"]

    export_cols = base_cols + [c for c in keep if c in staff.columns] + extra_cols
    staff.loc[:, export_cols].to_csv(f"{out}/tables/staff_with_models.csv", index=False)

    # --- Calculator HTML (single-cohort; STATE) -------------------------------
    calc_path = f"{out}/figures/cons_cap_calculator.html"
    write_cohort_aligned_calculator_html(
        long_st,
        target_percentile   = tp_st,
        inflation           = cns["inflation"],
        pre_degree_down_pct = cns.get("pre_degree_down_pct", 0.03),
        deg_ma_pct          = cns["deg_ma_pct"],
        deg_phd_pct         = cns["deg_phd_pct"],
        prep_bonus          = cns.get("prep_bonus", 2500.0),
        max_salary          = cap_cfg.get("max_salary", 100_000),
        cola_floor          = bump,
        outside_cap_years   = cfg["experience_policy"]["cap_outside_years"],
        skill_extra_years   = cfg["experience_policy"]["extra_years_per_skill"],
        path_html           = calc_path,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="SalaryAnalysis/scripts/config2.yaml")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)