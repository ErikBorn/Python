# scripts/run_pipeline.py  (consultant-only)

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
from src.planning import plan_salary_column, plan_costs, plan_costs_table, plan_salary_fte
from src.metrics import achieved_percentiles, band_medians_table, per_band_median_percentiles
from src.viz import (
    plot_bands,
    plot_scatter_models,
    plot_scatter_models_interactive,
    cons_cap_overview_html,
)
from src.share import write_cohort_aligned_calculator_html


def main(cfg):
    # --- Paths & IO -----------------------------------------------------------
    paths = cfg["paths"]
    out   = paths["out_dir"]
    os.makedirs(out, exist_ok=True)
    os.makedirs(f"{out}/tables", exist_ok=True)
    os.makedirs(f"{out}/figures", exist_ok=True)

    # Cohort + staff
    long  = load_cohort(paths["cohort_csv"])
    staff = load_staff(paths["staff_csv"])

    FTE_COL          = "Time Value"
    REAL_ACTUAL_COL  = "25-26 Salary (real)"
    REAL_FTE_COL     = "25-26 Salary"          # already FTE-scaled

    # --- Experience policy (if you still want it) -----------------------------
    # If you now supply credited years directly, you can disable this in YAML or
    # set cap very large. Leaving it here for backward compatibility.
    xp_cfg = cfg.get("experience_policy", {})
    staff  = apply_outside_exp_cap(
        staff,
        years_col="Years of Exp",
        seniority_col="Seniority",
        skill_flag_col=xp_cfg.get("skill_endorsement_col"),  # e.g., "Skill Endorsement"
        cap=float(xp_cfg.get("cap_outside_years", 10)),
    )

    # --- Consultant models ----------------------------------------------------
    bump = cfg["planning"]["bump"]   # e.g., 0.02
    tol  = cfg["planning"]["tol"]    # e.g., 0.02

    prep_series_name = cfg.get("prep", {}).get("flag_col", "Prep")
    prep_bonus       = cfg.get("prep", {}).get("bonus", 2500.0)

    cns = cfg["consultant"]

    # Base consultant
    staff["CONS Salary"] = consultant_predict(
        staff, long,
        target_percentile     = cns["target_percentile"],
        inflation             = cns["inflation"],
        base_start            = cns.get("base_start"),
        deg_ma_pct            = cns.get("deg_ma_pct", 0.02),
        deg_phd_pct           = cns.get("deg_phd_pct", 0.04),
        pre_degree_down_pct   = cns.get("pre_degree_down_pct", 0.03),
        auto_downshift        = cns.get("auto_downshift", False),
        years_col             = "Years of Exp",
        edu_col               = "Education Level",
        prep_col              = prep_series_name,
        prep_bonus            = prep_bonus,
        # NEW: flat stipends by flags
        skill_col             = cns.get("skill_col", "Skill Rating"),
        skill_bonus           = cns.get("skill_bonus", 0.0),
        leadership_col        = cns.get("lead_col", "Leadership Rating"),
        leadership_bonus      = cns.get("leadership_bonus", 0.0),
    )

    # Capped consultant
    cap_cfg = cfg.get("consultant_capped", {})
    staff["CONS_CAP Salary"] = consultant_predict_capped(
        staff, long,
        target_percentile     = cap_cfg.get("target_percentile", cns["target_percentile"]),
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
        skill_col             = cns.get("skill_col", "Skill Rating"),
        skill_bonus           = cns.get("skill_bonus", 0.0),
        leadership_col        = cns.get("lead_col", "Leadership Rating"),
        leadership_bonus      = cns.get("leadership_bonus", 0.0),
    )

    # --- COLA 2% (actual + FTE) -------------------------------------------------
    real_actual = _numify_money(staff[REAL_ACTUAL_COL])
    fte         = _numify_fte(staff[FTE_COL])

    staff["All +2% Plan Salary (actual)"] = real_actual * (1.0 + float(cfg["planning"]["bump"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        staff["All +2% Plan Salary"] = np.where(
            fte > 0, staff["All +2% Plan Salary (actual)"] / fte, np.nan
        )

    staff["_REAL_ACTUAL_NUM"] = _numify_money(staff[REAL_ACTUAL_COL])

    # CONS (actual + FTE)
    res = plan_salary_fte(
        staff,
        real_actual_col=REAL_ACTUAL_COL,
        model_fte_col="CONS Salary",
        fte_col=FTE_COL,
        out_actual_col="CONS Plan Salary (actual)",
        out_fte_col="CONS Plan Salary",
        bump=cfg["planning"]["bump"],
        tol=cfg["planning"]["tol"],
    )
    staff["CONS Plan Salary (actual)"] = pd.to_numeric(res["CONS Plan Salary (actual)"], errors="coerce")
    staff["CONS Plan Salary"]          = pd.to_numeric(res["CONS Plan Salary"],          errors="coerce")

    # CONS_CAP (actual + FTE)
    res = plan_salary_fte(
        staff,
        real_actual_col=REAL_ACTUAL_COL,
        model_fte_col="CONS_CAP Salary",
        fte_col=FTE_COL,
        out_actual_col="CONS_CAP Plan Salary (actual)",
        out_fte_col="CONS_CAP Plan Salary",
        bump=cfg["planning"]["bump"],
        tol=cfg["planning"]["tol"],
    )
    staff["CONS_CAP Plan Salary (actual)"] = pd.to_numeric(res["CONS_CAP Plan Salary (actual)"], errors="coerce")
    staff["CONS_CAP Plan Salary"]          = pd.to_numeric(res["CONS_CAP Plan Salary"],          errors="coerce")

    costs_tbl = plan_costs_table(
        staff,
        real_col=REAL_ACTUAL_COL,
        planned_cols=[
            "CONS Plan Salary (actual)",
            "CONS_CAP Plan Salary (actual)",
            "All +2% Plan Salary (actual)",
        ],
        labels=[
            "CONS",
            "CONS_CAP",
            "All +2%",
        ],
        bump_compare=None,
        format_output=True,
    )
    costs_tbl.to_csv(f"{out}/tables/plan_costs.csv")

    # --- Percentiles & band tables (consultant-only) --------------------------
    cols_dict = {
        "Real":     "25-26 Salary",
        "CONS":     "CONS Salary",
        "CONS_CAP": "CONS_CAP Salary",
        "CONS_Cap_Real": "CONS_CAP Plan Salary",   # keep label the same for the overlay
        "COLA 2%":  "All +2% Plan Salary",
    }
    band_pct_med = per_band_median_percentiles(
        long,
        staff,
        cols_dict=cols_dict,
        years_col="Years of Exp",
        decimals=1,
    )
    band_pct_med.to_csv(f"{out}/tables/band_percentile_medians.csv")

    pct = achieved_percentiles(
        long, staff,
        salary_cols=[
            "25-26 Salary",
            "CONS Salary",
            "CONS_CAP Salary",
            "CONS_CAP Plan Salary",   # was "CONS_Cap_Real"
            "All +2% Plan Salary",
        ],
        years_col="Years of Exp",
        labels=["Real", "CONS", "CONS_CAP", "CONS_Cap_Real", "COLA 2%"],
    ).round(1)
    pct.to_csv(f"{out}/tables/achieved_percentiles.csv")

    band_tbl = band_medians_table(
        long, staff,
        salary_cols=[
            "25-26 Salary",
            "CONS Salary",
            "CONS_CAP Salary",
            "CONS_CAP Plan Salary",   # <-- was "CONS_Cap_Real"
            "All +2% Plan Salary",
        ],
    )
    # format a copy for CSV (round salaries to $10; headcount stays integer)
    band_tbl_num = band_tbl.copy()
    mask = (band_tbl_num.index != "Headcount") if "Headcount" in band_tbl_num.index else slice(None)
    band_tbl_num.loc[mask] = band_tbl_num.loc[mask].round(-1)
    df_save = band_tbl_num.copy().astype(object)
    df_save.loc[mask] = np.vectorize(lambda x: "" if pd.isna(x) else f"${int(x):,}")(
        df_save.loc[mask].to_numpy(dtype=float)
    )
    if "Headcount" in df_save.index:
        df_save.loc["Headcount"] = band_tbl_num.loc["Headcount"].round(0).astype(int).astype(str)
    df_save.to_csv(f"{out}/tables/band_medians.csv")

    # --- Static plots ---------------------------------------------------------
    # plot_bands(
    #     long,
    #     cfg["cohort"]["target_percentile"],
    #     cfg["plots"]["plus_minus_pct"],
    #     f"{out}/figures/bands_with_models.png",
    #     staff=staff,
    #     years_col="Years of Exp",
    #     cols_dict={
    #         "Real": "25-26 Salary",
    #         "CONS_Cap_Real": "CONS_Cap_Real",
    #     },
    # )

    # plot_scatter_models(
    #     staff, "Years of Exp",
    #     cols_dict={
    #         "Real": "25-26 Salary",
    #         "CONS_Cap_Real": "CONS_Cap_Real",
    #     },
    #     path_png=f"{out}/figures/scatter_models.png",
    # )

    # --- Interactive scatter (with summary) ----------------------------------
    def _model_summary(planned_actual_col, planned_fte_col_for_mean_pct):
        c = plan_costs(
            staff,
            real_col="_REAL_ACTUAL_NUM",                # actual current
            planned_col=planned_actual_col,          # actual planned
            bump=cfg["planning"]["bump"],           # no headcount scaling anymore
        )

        # Mean %ile for the MODEL — compute against FTE view
        mean_pct = _mean_percentile_against(long, staff, planned_fte_col_for_mean_pct)

        return {
            "total_cost": c["Total Cost"],
            "mean_pct":   mean_pct,
            "num_raised": c["Num Raised"],
        }

    summary_info = {
        "params": {
            "target_percentile": cfg["cohort"]["target_percentile"],
            "target_inflation":  cfg["cohort"]["target_inflation"],
            "cola":              cfg["planning"]["bump"],
            "downshift_pct":     cns.get("pre_degree_down_pct", 0.03),
            "deg_ma_pct":        cns.get("deg_ma_pct", 0.02),
            "deg_phd_pct":       cns.get("deg_phd_pct", 0.04),
            "prep_bonus":        prep_bonus,
            "skill_bonus":       cns.get("skill_bonus", 0.0),
            "leadership_bonus":  cns.get("leadership_bonus", 0.0),
        },
        "models": {
            "CONS":     _model_summary("CONS Plan Salary (actual)",     "CONS Plan Salary"),
            "CONS_CAP": _model_summary("CONS_CAP Plan Salary (actual)", "CONS_CAP Plan Salary"),
            "COLA 2%":  _model_summary("All +2% Plan Salary (actual)",  "All +2% Plan Salary"),  # see 4b below
        },
    }

    plot_scatter_models_interactive(
        staff,
        years_col="Years of Exp",
        cols_dict={
            "Real":          "25-26 Salary",
            "CONS":          "CONS Salary",
            "CONS_CAP":      "CONS_CAP Salary",
            "CONS_Cap_Real": "CONS_CAP Plan Salary",   # legend label unchanged
            "COLA 2%":       "All +2% Plan Salary",
        },
        path_html=f"{out}/figures/scatter_models_interactive.html",
        hover_cols=[c for c in [
            "Employee","ID","Seniority","Education Level","Level","Category",
            "Time Value","25-26 Salary (real)"
        ] if c in staff.columns],
        marker_size=9,
        long=long,
        target_percentile=cfg["cohort"]["target_percentile"],
        inflation=cfg["cohort"]["target_inflation"],
        plus_minus_pct=cfg["plots"]["plus_minus_pct"],
        summary_info=summary_info,
        per_band_table=band_pct_med               # <— new
    )

    # --- Consultant Cap overview ---------------------------------------------
    cons_cap_overview_html(
        long,
        target_percentile = cap_cfg.get("target_percentile", cns["target_percentile"]),
        inflation         = cap_cfg.get("inflation",         cns["inflation"]),
        deg_ma_pct        = cap_cfg.get("deg_ma_pct",        cns.get("deg_ma_pct", 0.02)),
        deg_phd_pct       = cap_cfg.get("deg_phd_pct",       cns.get("deg_phd_pct", 0.04)),
        prep_bonus        = cfg["planning"].get("prep_bonus", 2500.0),
        max_salary        = cap_cfg.get("max_salary", 100_000),
        cola_floor        = cfg["planning"]["bump"],
        outside_cap_years = cfg.get("years_policy", {}).get("outside_cap_years", 10),
        skill_extra_years = cfg.get("years_policy", {}).get("skill_extra_years", 5),
        skill_col_name    = cfg.get("years_policy", {}).get("skill_col_name", "Skill"),
        base_start        = cns.get("base_start"),
        path_html         = f"{out}/figures/cons_cap_overview.html",
        example           = dict(years=18.0, seniority=3.0, degree="MA", prep=1, skill=0, current=70_000),
    )

    # --- Minimal sharing CSV --------------------------------------------------
    CURRENT_COL = "25-26 Salary"
    KEEP_CALCS  = [
        "CONS Salary",
        "CONS_CAP Salary",
        "CONS_CAP Plan Salary",
        "All +2% Plan Salary",
        "Hire Date", "Eth", "Gender", "Years of Exp", "Seniority",
        "Education Level", "Skill Rating", "Leadership Rating", "Prep Rating",
        "Level", "Category",
    ]

    cols_all = list(staff.columns)
    base_cols = cols_all[: cols_all.index(CURRENT_COL) + 1] if CURRENT_COL in cols_all else cols_all

    staff["CCP Increase"] = (
        pd.to_numeric(staff.get("CONS_CAP Plan Salary"), errors="coerce") -
        pd.to_numeric(staff.get(CURRENT_COL),            errors="coerce")
    )
    denom = pd.to_numeric(staff.get(CURRENT_COL), errors="coerce")
    staff["CCP Increase %"] = np.where(denom > 0, staff["CCP Increase"] / denom, np.nan)

    export_cols = base_cols + [c for c in KEEP_CALCS if c in staff.columns] + ["CCP Increase", "CCP Increase %"]
    staff.loc[:, export_cols].to_csv(f"{out}/tables/staff_with_models.csv", index=False)

    # --- Calculator HTML ------------------------------------------------------
    calc_path = f"{out}/figures/cons_cap_calculator.html"
    write_cohort_aligned_calculator_html(
        long,
        target_percentile   = cns["target_percentile"],
        inflation           = cns["inflation"],
        pre_degree_down_pct = cns.get("pre_degree_down_pct", 0.03),
        deg_ma_pct          = cns["deg_ma_pct"],
        deg_phd_pct         = cns["deg_phd_pct"],
        prep_bonus          = cns.get("prep_bonus", 2500.0),
        skill_bonus         = cns.get("skill_bonus", 0.0),
        leadership_bonus    = cns.get("leadership_bonus", 0.0),
        max_salary          = cap_cfg.get("max_salary", 100_000),
        cola_floor          = cfg["planning"]["bump"],
        path_html           = calc_path,
    )

def _mean_percentile_against(long_cohort, staff, planned_col) -> float:
    """Mean achieved percentile of `planned_col` against `long_cohort`."""
    df = achieved_percentiles(long_cohort, staff, salary_cols=[planned_col], years_col="Years of Exp")
    s = df.squeeze()
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return float(s.get("Mean %ile", np.nan))

def _numify_money(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.replace(r"[,\$]", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")

def _numify_fte(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        st = s.astype(str).str.strip()
        pct_mask = st.str.endswith("%", na=False)
        s_pct = pd.to_numeric(st.str.rstrip("%"), errors="coerce") / 100.0
        s_num = pd.to_numeric(st, errors="coerce")
        s = np.where(pct_mask, s_pct, s_num)
        s = pd.Series(s, index=st.index)
    else:
        s = pd.to_numeric(s, errors="coerce")
    big = s > 1.5         # e.g., 80 -> 0.8
    s.loc[big] = s.loc[big] / 100.0
    return s.fillna(1.0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="SalaryAnalysis/scripts/config.yaml")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)