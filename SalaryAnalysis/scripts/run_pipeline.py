import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse, os, yaml, pandas as pd
from src.io import load_staff, load_cohort, write_csv
from src.clean import standardize_columns
from src.cohort import build_band_bins_closed, interpolate_salary_strict
from src.rpb import compute_rpb, validate_rpb
from src.models.nonlinear import nonlinear_predict, total_years
from src.models.piecewise import derive_pw_bands_ols, pw_predict
from src.metrics import achieved_percentiles, band_medians_table
from src.planning import plan_salary_column, plan_costs
from src.viz import plot_bands, plot_scatter_models

def main(cfg):
    paths = cfg["paths"]; out = paths["out_dir"]; os.makedirs(out, exist_ok=True)
    # 1) Load
    long  = load_cohort(paths["cohort_csv"])        # tidy long
    staff = load_staff(paths["staff_csv"])          # raw staff -> cleaned

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
    # quick callable: use staff grid to fit; for better fidelity, supply your own function
    import numpy as np
    def nl_baseline_fn(t):
        # predict NLM with BA-only multiplier (ma_pct=phd_pct=0) via scaling:
        tmp = staff.copy()
        tmp["Years of Exp"] = t if isinstance(t, np.ndarray) else np.asarray(t)
        return nonlinear_predict(
            tmp, **{**nl, "ma_pct":0.0, "phd_pct":0.0}
        )

    pw_cfg = cfg["piecewise"]
    pw_bands = derive_pw_bands_ols(
        nl_baseline_fn,
        edges=pw_cfg["edges"],
        round_base=pw_cfg["round_base"],
        round_step=pw_cfg["round_step"],
        enforce_continuity=pw_cfg["enforce_continuity"],
        shrink_to_total=None if pw_cfg["shrink_to_total"] in (None, "null") else pw_cfg["shrink_to_total"]
    )
    # predict PW with degree multipliers
    staff["PW"] = pw_predict(
        total_years=total_years(staff["Years of Exp"], staff["Seniority"], nl["f_non_sen"]),
        edu=staff.get("Education Level",""),
        pw_bands=pw_bands,
        ma_pct=nl["ma_pct"], phd_pct=nl["phd_pct"], stack_degrees=nl["stack_degrees"]
    )

    # 5) Plans & costs (raise-to-model-or-2% bump)
    bump, tol = cfg["planning"]["bump"], cfg["planning"]["tol"]
    for label, col in [("PW","PW"), ("NLM","Model NL Salary"), ("RPB","RPB Salary")]:
        out_col = f"{label} Plan Salary"
        staff[out_col] = plan_salary_column(staff, "25-26 Salary", col, out_col, bump=bump, tol=tol)

    costs = {
        label: plan_costs(staff, "25-26 Salary", f"{label} Plan Salary")
        for label in ["PW","NLM","RPB"]
    }
    pd.DataFrame.from_dict(costs, orient="index").to_csv(f"{out}/tables/plan_costs.csv")

    # 6) Metrics / percentiles & tables
    pct = achieved_percentiles(long, staff,
        salary_cols=["25-26 Salary", "Model NL Salary", "RPB Salary", "PW"])
    pct.to_csv(f"{out}/tables/achieved_percentiles.csv")

    band_tbl = band_medians_table(long, staff,
        salary_cols=["25-26 Salary", "Model NL Salary", "RPB Salary", "PW"])
    band_tbl.to_csv(f"{out}/tables/band_medians.csv")

    # 7) Plots
    plot_bands(long, cfg["cohort"]["target_percentile"],
               cfg["plots"]["plus_minus_pct"], f"{out}/figures/bands_p{cfg['cohort']['target_percentile']}.png")

    cols_dict = {"Real":"25-26 Salary","NLM":"Model NL Salary","RPB":"RPB Salary","PW":"PW"}
    plot_scatter_models(staff, "Years of Exp", cols_dict, f"{out}/figures/scatter_models.png")

    # 8) Save staff w/ predictions
    staff.to_csv(f"{out}/tables/staff_with_models.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="config.yaml")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)