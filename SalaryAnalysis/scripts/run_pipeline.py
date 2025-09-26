import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import os
import yaml
import pandas as pd
import numpy as np

from src.io import load_staff, load_cohort, write_csv
from src.clean import standardize_columns
from src.cohort import build_band_bins_closed, interpolate_salary_strict
from src.rpb import compute_rpb, validate_rpb
from src.models.nonlinear import nonlinear_predict, total_years
from src.models.piecewise import derive_pw_bands_ols, pw_predict
from src.metrics import achieved_percentiles, band_medians_table, per_band_median_percentiles
from src.planning import plan_salary_column, plan_costs
from src.viz import cons_cap_overview_html, plot_bands, plot_scatter_models, plot_scatter_models_interactive
from src.models.consultant import consultant_predict, consultant_predict_capped
from src.policies import apply_outside_exp_cap
from src.share import write_cohort_aligned_calculator_html  

def make_nl_baseline_fn(nl_params: dict):
    """
    Returns f(T): BA baseline from the nonlinear model evaluated at total-years T.
    Degree multipliers neutralized; seniority=0 so Years of Exp == T.
    """
    p = dict(nl_params)
    p.update({
        "ma_pct": 0.0,
        "phd_pct": 0.0,
        "stack_degrees": False,
        "f_non_sen": None,   # use years directly
    })

    def f(t):
        t = np.asarray(t, dtype=float)
        n = t.size
        tmp = pd.DataFrame({
            "Years of Exp":      t,
            "Seniority":         np.zeros(n),
            "Education Level":   [""] * n,
            "Skill Rating":      np.zeros(n),
            "Prep Rating":       np.zeros(n),
            "Knowledge Rating":  np.zeros(n),
            "Level":             [""] * n,
        })
        return nonlinear_predict(tmp, **p).to_numpy()

    return f

def main(cfg):
    paths = cfg["paths"]; out = paths["out_dir"]; os.makedirs(out, exist_ok=True)
    # 1) Load
    long  = load_cohort(paths["cohort_csv"])        # tidy long
    staff = load_staff(paths["staff_csv"])          # raw staff -> cleaned

    # 1b) Apply outside-experience policy EARLY
    xp_cfg = cfg.get("experience_policy", {})
    staff = apply_outside_exp_cap(
        staff,
        years_col="Years of Exp",
        seniority_col="Seniority",
        skill_flag_col=xp_cfg.get("skill_endorsement_col"),  # e.g. "Skill Endorsement"
        cap=float(xp_cfg.get("cap_outside_years", 10))
    )
    # 2) RPB
    staff["RPB Salary"] = compute_rpb(
        long, staff,
        target_percentile=cfg["cohort"]["target_percentile"],
        target_inflation=cfg["cohort"]["target_inflation"]
    )
    
    # 3) Nonlinear model
    nl = cfg["nonlinear"]
    staff["Model NL Salary"] = nonlinear_predict(staff, **nl)

    # 4) Piecewise from NLM baseline (OLS)
    # Build BA baseline callable over Total Years
    T = total_years(staff["Years of Exp"], staff["Seniority"], nl["f_non_sen"])

    pw_cfg = cfg["piecewise"]
    nl_baseline_fn = make_nl_baseline_fn(nl)
    pw_bands = derive_pw_bands_ols(
        nl_baseline_fn,
        edges=pw_cfg["edges"],
        round_base=pw_cfg["round_base"],
        round_step=pw_cfg["round_step"],
        enforce_continuity=True,
        shrink_to_total=pw_cfg.get("shrink_to_total")
    )
    # predict PW with degree multipliers
    staff["PW"] = pw_predict(
        total_years=total_years(staff["Years of Exp"], staff["Seniority"], nl["f_non_sen"]),
        edu=staff.get("Education Level", ""),
        pw_bands=pw_bands,
        ma_pct=nl["ma_pct"], phd_pct=nl["phd_pct"], stack_degrees=nl["stack_degrees"],
        prep_flag=staff.get("Prep", staff.get("Prep Rating")),  # uses "Prep" or falls back to "Prep Rating"
        prep_bonus=cfg["piecewise"]["prep_bonus"]  # or cfg["piecewise"]["prep_bonus"]
    )

    cola = cfg["planning"]["bump"]         # e.g. 0.02
    tol  = cfg["planning"]["tol"]          # e.g. 0.02

    staff["PWR Salary"] = plan_salary_column(
        staff,
        real_col="25-26 Salary",
        model_col="PW",                     # the piecewise target
        out_col="PWR Salary",
        bump=cola,
        tol=tol
    )
    # 4b) Consultant model
    # cns = cfg["consultant"]
    # cns_cap_cfg = cfg["consultant_capped"]
    prep_series_name = cfg.get("prep", {}).get("flag_col", "Prep")
    prep_bonus = cfg.get("prep", {}).get("bonus", 2500.0)

    # --- Consultant model (positive-only bumps + pre-downshift) ---
    cns = cfg["consultant"]
    staff["CONS Salary"] = consultant_predict(
        staff, long,
        target_percentile = cns["target_percentile"],
        inflation         = cns["inflation"],
        base_start        = cns.get("base_start"),
        # positive-only bumps
        deg_ma_pct        = cns.get("deg_ma_pct", 0.02),
        deg_phd_pct       = cns.get("deg_phd_pct", 0.04),
        # NEW: aim the base curve down before degree bumps
        pre_degree_down_pct = cns.get("pre_degree_down_pct", 0.03),
        auto_downshift      = cns.get("auto_downshift", False),
        # flat bonus
        prep_col          = prep_series_name,
        prep_bonus        = prep_bonus,
    )

    # --- Capped consultant model ---
    cns_cap_cfg = cfg.get("consultant_capped", {})
    staff["CONS_CAP Salary"] = consultant_predict_capped(
        staff, long,
        target_percentile   = cns_cap_cfg.get("target_percentile", cns["target_percentile"]),
        inflation           = cns_cap_cfg.get("inflation", cns["inflation"]),
        base_start          = cns_cap_cfg.get("base_start", cns.get("base_start")),
        deg_ma_pct          = cns_cap_cfg.get("deg_ma_pct", cns.get("deg_ma_pct", 0.02)),
        deg_phd_pct         = cns_cap_cfg.get("deg_phd_pct", cns.get("deg_phd_pct", 0.04)),
        pre_degree_down_pct = cns_cap_cfg.get("pre_degree_down_pct", cns.get("pre_degree_down_pct", 0.03)),
        auto_downshift      = cns_cap_cfg.get("auto_downshift", cns.get("auto_downshift", False)),
        max_salary          = cns_cap_cfg.get("max_salary", 100_000),
        prep_col            = prep_series_name,
        prep_bonus          = prep_bonus,
    )

    # Realized (raise-to-model-or-2% floor) version of capped consultant model
    staff["CONS_Cap_Real"] = plan_salary_column(
        staff,
        real_col="25-26 Salary",
        model_col="CONS_CAP Salary",
        out_col="CONS_Cap_Real",
        bump=cfg["planning"]["bump"],   # your COLA floor, e.g. 0.02
        tol=cfg["planning"]["tol"]      # your tolerance (e.g. 0.02)
    )

    # 5) Plans & costs (raise-to-model-or-2% bump)
    bump, tol = cfg["planning"]["bump"], cfg["planning"]["tol"]

    # Create the plan salary columns first
    for label, col in [("PW","PW"), ("NLM","Model NL Salary"), ("RPB","RPB Salary"), ("CONS","CONS Salary"), ("CONS_CAP","CONS_CAP Salary"),]:
        out_col = f"{label} Plan Salary"
        staff[out_col] = plan_salary_column(staff, "25-26 Salary", col, out_col,
                                            bump=cfg["planning"]["bump"],
                                            tol=cfg["planning"]["tol"])

    # Now compute costs
    plans = ["PW", "NLM", "RPB"]
    costs = {
        label: plan_costs(staff, "25-26 Salary", f"{label} Plan Salary", bump=bump)
        for label in plans
    }

    # If you created a global 2% row, e.g.:
    staff["All +2% Plan Salary"] = pd.to_numeric(staff["25-26 Salary"], errors="coerce") * (1.0 + bump)
    # add it explicitly:
    costs["All +2%"] = plan_costs(staff, "25-26 Salary", "All +2% Plan Salary", bump=bump)

    # NEW: pretty, rounded costs table + comparison row
    from src.planning import plan_costs_table  # (uses format_money_10 internally)

    costs_tbl = plan_costs_table(
        staff,
        real_col="25-26 Salary",
        planned_cols=[
            # "PWR Salary",          # realized PW result
            "PW Plan Salary",      # if you still keep the raw PW-plan column
            "NLM Plan Salary",
            "RPB Plan Salary",
            "CONS Plan Salary",
            "CONS_CAP Plan Salary",          # NEW
        ],
        labels=[
            # "PWR",
            "PW",
            "NLM",
            f"RPB ({cfg['cohort']['target_percentile']}%)",
            "CONS",
            "CONS_CAP",                      # NEW
        ],
        bump_compare=0.02,     # keeps the “All +2%” comparison row
        format_output=True
    )
    costs_tbl.to_csv(f"{out}/tables/plan_costs.csv")

    from src.metrics import per_band_median_percentiles

    # rows=models, cols=bands; values = median achieved %ile per band
    band_pct_med = per_band_median_percentiles(
        long,
        staff,
        cols_dict={
            "Real": "25-26 Salary",
            # "RPB":  "RPB Salary",
            # "NLM":  "Model NL Salary",
            # # "PW":   "PW",
            # "CONS":"CONS Salary",
            # "CONS_CAP":"CONS_CAP Salary",
            "CONS_Cap_Real":"CONS_Cap_Real",
            "COLA 2%":"All +2% Plan Salary",
            # "PWR":  "PWR Plan Salary",   # include if you’ve computed PWR
        },
        years_col="Years of Exp",
        decimals=1,
    )
    
    os.makedirs(f"{out}/tables", exist_ok=True)
    band_pct_med.to_csv(f"{out}/tables/band_percentile_medians.csv")

    # 6) Metrics / percentiles & tables
    # First calculate raw and plan separately (same as before)
    pct_models = achieved_percentiles(
        long, staff,
        salary_cols=[
            "25-26 Salary", "RPB Salary", "Model NL Salary", "PW", "CONS Salary"
        ],
        years_col="Years of Exp",
        labels=["Real", "RPB", "NLM", "PW", "CONS"]
    ).round(1)

    pct_plans = achieved_percentiles(
        long, staff,
        salary_cols=[
            "RPB Plan Salary", "NLM Plan Salary", "PW Plan Salary", "CONS Plan Salary", "CONS_CAP Plan Salary"
        ],
        years_col="Years of Exp",
        labels=["RPB", "NLM", "PW", "CONS", "CONS_Cap"]   # match to models for easier merge
    ).round(1)

    # --- Reshape ---
    # Both DataFrames have index ["Mean %ile", "Median %ile"], columns=labels
    # We want rows=Model, columns=MultiIndex (Metric, Version)

    # Transpose so models are rows
    raw_T  = pct_models.T
    plan_T = pct_plans.T

    # Add a 'Version' level
    raw_T["Version"]  = "Raw"
    plan_T["Version"] = "Plan"

    # Combine
    combined = pd.concat([raw_T, plan_T], axis=0)

    # Set index to (Model, Version)
    combined = combined.set_index("Version", append=True)

    # Reorder to [Raw, Plan] for each model
    combined = combined.reorder_levels([1, 0]).sort_index(level=1)

    # Final shape: rows=(Model,Version), cols=["Mean %ile","Median %ile"]
    # If you want Model as rows and Raw/Plan as columns side by side:
    final = combined.unstack(level=0)

    # Cleanup column names
    final.columns = [f"{metric} ({version})" for metric, version in final.columns]

    # Write out
    pct = achieved_percentiles(
            long, staff,
            salary_cols=["25-26 Salary", "RPB Salary", "Model NL Salary", "PW", "CONS Salary", "CONS_CAP Salary"]
        ).round(1)
    pct.to_csv(f"{out}/tables/achieved_percentiles.csv")

    band_tbl = band_medians_table(
        long, staff,
        salary_cols=["25-26 Salary", "RPB Salary", "Model NL Salary", "PW", "CONS Salary", "CONS_CAP Salary"]
    )
    # (your existing rounding/formatting block follows)

    # -------- Option A: keep numeric; format only when saving --------
    # Round salaries (not headcount) to nearest $10 for output
    band_tbl_num = band_tbl.copy()
    mask_rows = (band_tbl_num.index != "Headcount") if "Headcount" in band_tbl_num.index else slice(None)
    band_tbl_num.loc[mask_rows] = band_tbl_num.loc[mask_rows].round(-1)

    # Build a *separate* formatted copy for CSV without touching numerics
    df_save = band_tbl_num.copy().astype(object)

    # Money formatter for salary rows
    def _money_str(x):
        return "" if pd.isna(x) else f"${int(x):,}"

    # Apply formatting to salary rows
    df_save.loc[mask_rows] = np.vectorize(_money_str)(
        df_save.loc[mask_rows].to_numpy(dtype=float)
    )

    # Headcount row stays integer (no $)
    if "Headcount" in df_save.index:
        df_save.loc["Headcount"] = band_tbl_num.loc["Headcount"].round(0).astype(int).astype(str)

    df_save.to_csv(f"{out}/tables/band_medians.csv")
    # ---------------------------------------------------------------

    # 7) Plots
    #Use this for banded plot
    cols_dict = {
        "Real": "25-26 Salary",
        # "RPB": "RPB Salary",
        # "NLM": "Model NL Salary",
        # "PW": "PW",
        # "PWR": "PWR Salary",   # cyan
        # "CONS":"CONS Salary",
        # "CONS_CAP": "CONS_CAP Salary",
        "CONS_Cap_Real": "CONS_Cap_Real",   # NEW
    }

    plot_bands(
        long,
        cfg["cohort"]["target_percentile"],
        cfg["plots"]["plus_minus_pct"],
        f"{out}/figures/bands_with_models.png",
        staff=staff,
        years_col="Years of Exp",
        cols_dict=cols_dict
    )
    #Use this for scatter plot
    cols_dict = {
        "Real": "25-26 Salary",
        # "RPB": "RPB Salary",
        # "NLM": "Model NL Salary",
        # "PW": "PW",
        # "CONS":"CONS Salary",
        # "CONS_CAP": "CONS_CAP Salary",
        "CONS_Cap_Real": "CONS_Cap_Real",   # NEW
        # "COLA 2%": "All +2% Plan Salary",
        # "PWR": "PWR Salary",   # cyan
    }

    plot_scatter_models(staff, "Years of Exp", cols_dict,
                        f"{out}/figures/scatter_models_with_pwr.png")

    # ... you already computed these planned columns earlier:
    #   "PW Plan Salary", "NLM Plan Salary", "RPB Plan Salary", maybe "All +2% Plan Salary"

    # Build per-model summary for the panel
    def _model_summary(label, planned_col):
        c = plan_costs(staff, "25-26 Salary", planned_col, bump=cfg["planning"]["bump"])

        # overall %ile (mean) for the planned salaries
        pct_df = achieved_percentiles(
            long, staff, salary_cols=[planned_col], years_col="Years of Exp"
        )

        # Works whether pct_df is a Series or a 1-col DataFrame
        s = pct_df.squeeze()
        if isinstance(s, pd.DataFrame):  # extremely defensive
            s = s.iloc[:, 0]
        mean_pct = float(s.get("Mean %ile", np.nan))

        return {
            "total_cost": c["Total Cost"],
            "mean_pct": mean_pct,
            "num_raised": c["Num Raised"],
        }

    summary_info = {
        "params": {
            "target_percentile": cfg["cohort"]["target_percentile"],
            "target_inflation":  cfg["cohort"]["target_inflation"],
            "cola":              cfg["planning"]["bump"],
        },
        "models": {
            # "PW":  _model_summary("PW",  "PW Plan Salary"),
            # "NLM": _model_summary("NLM", "NLM Plan Salary"),
            # "RPB": _model_summary("RPB", "RPB Plan Salary"),
            # include consultant / PWR variants if you want:
            "CONS": _model_summary("CONS", "CONS Plan Salary"),
            "CONS_CAP": _model_summary("CONS_CAP", "CONS_CAP Plan Salary"),
            "COLA 2%": _model_summary("COLA 2%", "All +2% Plan Salary"),
        }
    }

    # What to show as points on the interactive scatter
    cols_dict = {
        "Real": "25-26 Salary",
        # "RPB":  "RPB Salary",
        # "NLM":  "Model NL Salary",
        # "PW":   "PW",
        "CONS": "CONS Salary",
        "CONS_CAP": "CONS_CAP Salary",
        "CONS_Cap_Real": "CONS_Cap_Real",   # NEW
        "COLA 2%": "All +2% Plan Salary",
        # "PWR":  "PW Plan Salary",   # optional to show the plan outcome as a series
    }

    # Export interactive HTML with bands + summary box
    plot_scatter_models_interactive(
        staff,
        years_col="Years of Exp",
        cols_dict=cols_dict,
        path_html=f"{out}/figures/scatter_models_interactive.html",
        hover_cols=["Employee","ID","Seniority","Education Level","Level","Category"],
        marker_size=9,
        long=long,
        target_percentile=cfg["cohort"]["target_percentile"],
        inflation=0.02,
        plus_minus_pct=cfg["plots"]["plus_minus_pct"],
        summary_info=summary_info
    )

    cns_cap = cfg.get("consultant_capped", {})
    cons_cap_overview_html(
        long,
        target_percentile = cns_cap.get("target_percentile", cfg["consultant"]["target_percentile"]),
        inflation         = cns_cap.get("inflation", cfg["consultant"]["inflation"]),
        deg_ma_pct        = cns_cap.get("deg_ma_pct", cfg["consultant"].get("deg_ma_pct", 0.02)),
        deg_phd_pct       = cns_cap.get("deg_phd_pct", cfg["consultant"].get("deg_phd_pct", 0.04)),
        prep_bonus        = cfg["planning"].get("prep_bonus", 2500.0),
        max_salary        = cns_cap.get("max_salary", 100_000),
        cola_floor        = cfg["planning"]["bump"],
        outside_cap_years = cfg.get("years_policy", {}).get("outside_cap_years", 10),
        skill_extra_years = cfg.get("years_policy", {}).get("skill_extra_years", 5),
        skill_col_name    = cfg.get("years_policy", {}).get("skill_col_name", "Skill"),
        base_start        = cfg["consultant"].get("base_start"),
        path_html         = f"{cfg['paths']['out_dir']}/figures/cons_cap_overview.html",
        # optional: pick your own example person
        example = dict(years=18.0, seniority=3.0, degree="MA", prep=1, skill=0, current=70_000),
    )

    # 8) Save staff w/ predictions
    # Drop redundant column if it exists
    # --- tidy salary columns for export ---
    # 1) Drop redundant column if present
    if "PWR Salary" in staff.columns:
        staff = staff.drop(columns=["PWR Salary"])

    # 2) Desired order for the calculated/model columns
    calc_cols_desired = [
            # "RPB Salary",
            # "Model NL Salary",
            # "PW",
            "CONS Salary",
            "CONS_CAP Salary",      # NEW
            # "PW Plan Salary",
            # "NLM Plan Salary",
            # "RPB Plan Salary",
            "CONS Plan Salary",
            "CONS_CAP Plan Salary", # NEW
            "All +2% Plan Salary",
        ]
    calc_cols_present = [c for c in calc_cols_desired if c in staff.columns]

    # 3) Build final column order:
    orig = list(staff.columns)
    if "25-26 Salary" in orig:
        idx_real = orig.index("25-26 Salary")
    else:
        # Fallback: put calcs at the end if the real column isn't present
        idx_real = len(orig) - 1

    prefix = orig[:idx_real + 1]            # everything up to and including the real salary
    final_cols: list[str] = []

    def add_unique(cols):
        for c in cols:
            if c in staff.columns and c not in final_cols:
                final_cols.append(c)

    # keep original order, then inject the calcs, then everything else
    add_unique(prefix)
    add_unique(calc_cols_present)
    add_unique(orig)

    # 4) Reindex and save
    staff = staff.reindex(columns=final_cols)
    staff.to_csv(f"{out}/tables/staff_with_models.csv", index=False)

    # 9) Write Consultant Cap calculator HTML
    calc_path = f"{out}/figures/cons_cap_calculator.html"
    write_cohort_aligned_calculator_html(
        long,
        target_percentile=cfg["consultant"]["target_percentile"],
        inflation=cfg["consultant"]["inflation"],
        pre_degree_down_pct=cfg["consultant"].get("pre_degree_down_pct", 0.03),
        deg_ma_pct=cfg["consultant"]["deg_ma_pct"],
        deg_phd_pct=cfg["consultant"]["deg_phd_pct"],
        prep_bonus=cfg["consultant"].get("prep_bonus", 2500.0),
        max_salary=cfg.get("consultant_capped", {}).get("max_salary", 100_000),
        cola_floor=cfg["planning"]["bump"],
        outside_cap_years=cfg["experience_policy"]["cap_outside_years"],
        skill_extra_years=cfg["experience_policy"]["extra_years_per_skill"],
        path_html=calc_path,
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="SalaryAnalysis/scripts/config.yaml")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)