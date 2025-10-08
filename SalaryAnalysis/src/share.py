from __future__ import annotations
from typing import Iterable, Sequence, Optional, Mapping
import os
import json
import numpy as np
import pandas as pd
from .cohort import build_band_bins_closed, interpolate_salary_strict

def _band_from_years_label(
    years: float,
    *,
    band_starts: Sequence[int] = (0, 6, 11, 16, 21, 26, 31, 36, 41),
    custom_labels: Optional[Sequence[str]] = None,
) -> str | float:
    """Return text band like '0-5', '6-10', …, '41+' for a numeric years value."""
    if years is None or not np.isfinite(years):  # NaN -> keep blank in export
        return np.nan

    y = float(years)
    if custom_labels is None:
        ends = list(band_starts[1:]) + [None]
        labels = [f"{s}-{e-1}" if e is not None else f"{s}+"
                  for s, e in zip(band_starts, ends)]
    else:
        labels = list(custom_labels)
        if len(labels) != len(band_starts):
            raise ValueError("custom_labels must match band_starts length.")

    for i, s in enumerate(band_starts):
        if i == len(band_starts) - 1:
            if y >= s: return labels[i]
        else:
            e = band_starts[i + 1]
            if s <= y < e: return labels[i]
    return np.nan


def export_with_band_and_headers(
    staff: pd.DataFrame,
    *,
    years_source_col: str = "Years of Exp",   # where "years" lives in staff
    years_output_name: str = "Years",         # column name the sheet expects
    headers: Sequence[str],                   # exact header list (order preserved)
    path_csv: str,
    # band options
    band_col_name: str = "Band",
    band_starts: Sequence[int] = (0, 6, 11, 16, 21, 26, 31, 36, 41),
    custom_band_labels: Optional[Sequence[str]] = None,
    # optional simple renaming for mismatches (e.g., staff col -> header name)
    rename_map: Optional[Mapping[str, str]] = None,
    # optional sort after building columns
    sort_by: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Create a CSV exactly matching `headers`, computing a text Band and a 'Years' column.

    - Reads years from `years_source_col` in `staff` and writes it to `years_output_name`.
    - Computes `band_col_name` from years using the standard buckets.
    - For any header not present after renaming/computation, inserts a blank column.
    - Returns the final DataFrame and writes it to `path_csv`.
    """
    df = staff.copy()

    # ---- Normalize / rename incoming columns ----
    if rename_map:
        df = df.rename(columns=dict(rename_map))

    # ---- Years: source -> output name ----
    if years_source_col not in df.columns:
        raise ValueError(f"Missing years_source_col {years_source_col!r} in staff.")
    df[years_output_name] = pd.to_numeric(df[years_source_col], errors="coerce")

    # ---- Band: compute text bucket from Years ----
    df[band_col_name] = df[years_output_name].apply(
        lambda v: _band_from_years_label(v, band_starts=band_starts, custom_labels=custom_band_labels)
    )

    # ---- Build output frame in the exact header order ----
    out_cols = []
    for h in headers:
        if h in df.columns:
            out_cols.append(h)
        else:
            # create a blank column for any header we don't have in staff
            df[h] = np.nan
            out_cols.append(h)

    out = df.loc[:, out_cols].copy()

    # Optional sort (e.g., by Band then Years)
    if sort_by:
        out = out.sort_values(list(sort_by), kind="mergesort")

    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    out.to_csv(path_csv, index=False)
    return out


def export_for_format_sheet(
    staff: pd.DataFrame,
    *,
    years_col: str = "Years of Exp",
    include_cols: Iterable[str],
    path_csv: str,
    add_band: bool = True,
    band_col_name: str = "Band",
    band_starts: Sequence[int] = (0, 6, 11, 16, 21, 26, 31, 36, 41),
    custom_band_labels: Optional[Sequence[str]] = None,
    put_band_after: str | None = None,     # insert Band after this column if present
    sort_by: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Build a CSV that mirrors your formatting sheet.

    - `include_cols` is the exact column order you want in the output (excluding Band).
    - If `add_band=True`, a text 'Band' column is calculated from `years_col` using the
      standard buckets (0-5, 6-10, ..., 41+), then inserted either:
        - right after `put_band_after` (if that column exists), or
        - at the end (fallback).
    - Returns the final DataFrame and writes it to `path_csv`.

    NOTE: values are written raw (no currency symbols) to preserve target-sheet formatting.
    """
    # Defensive copy and type cleaning for years
    df = staff.copy()
    if years_col not in df.columns:
        raise ValueError(f"'{years_col}' not found in staff columns.")

    df[years_col] = pd.to_numeric(df[years_col], errors="coerce")

    # Calculate Band if requested
    if add_band:
        df[band_col_name] = df[years_col].apply(
            lambda v: _band_from_years_label(v, band_starts=band_starts, custom_labels=custom_band_labels)
        )

    # Build the column order
    final_cols = list(include_cols)
    if add_band:
        if put_band_after and put_band_after in final_cols:
            # insert after the specified column
            idx = final_cols.index(put_band_after) + 1
            final_cols.insert(idx, band_col_name)
        else:
            final_cols.append(band_col_name)

    # Keep only the columns that actually exist; warn if any are missing
    missing = [c for c in final_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in staff for export: {missing}")

    out = df.loc[:, final_cols].copy()

    # Optional sort (commonly on Band then Years/Employee)
    if sort_by:
        # robust numeric sort on years if included
        out = out.sort_values(list(sort_by), kind="mergesort")

    # Ensure dir exists and write
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    out.to_csv(path_csv, index=False)

    return out

def write_cohort_aligned_calculator_html(
    long: pd.DataFrame,
    *,
    # model knobs
    target_percentile: float = 70.0,
    inflation: float = 0.02,
    pre_degree_down_pct: float = 0.03,
    deg_ma_pct: float = 0.02,
    deg_phd_pct: float = 0.04,
    # flat stipends
    prep_bonus: float = 2500.0,
    skill_bonus: float = 0.0,
    leadership_bonus: float = 0.0,
    # fixed policy values
    max_salary: float = 100_000.0,
    cola_floor: float = 0.02,
    # where to write
    path_html: str = "outputs/figures/cohort_aligned_calculator.html",
):
    """
    Self-contained calculator for the Cohort-aligned (CONS_CAP) model.
    Uses a single 'Credited Years' input supplied by you.
    """

    # Build band-midpoint anchors (@ target percentile, inflated)
    bins, _ = build_band_bins_closed(long)
    xp, fp = [], []
    for start, end, label in bins:
        mid = 0.5 * (float(start) + (float(end) if np.isfinite(end) else float(start) + 5.0))
        s = interpolate_salary_strict(long, label, float(target_percentile))
        if not np.isnan(s):
            xp.append(float(mid))
            fp.append(float(s) * (1.0 + float(inflation)))

    cfg = {
        "xp": xp,
        "fp": fp,
        "preDown": float(pre_degree_down_pct),
        "maPct": float(deg_ma_pct),
        "phdPct": float(deg_phd_pct),
        "prepBonus": float(prep_bonus),
        "skillBonus": float(skill_bonus),
        "leadBonus": float(leadership_bonus),
        "maxSalary": float(max_salary),
        "colaFloor": float(cola_floor),
        "pTitle": int(round(target_percentile)),
        "inflPct": int(round(inflation * 100)),
    }

    def _money(x): return f"${x:,.0f}"

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Cohort-aligned Salary Calculator</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  :root {{ --bg:#f7f9fc; --card:#fff; --text:#1f2d3d; --muted:#6b7280; --accent:#0f766e; --border:#e5e7eb; }}
  html,body {{ margin:0; padding:0; background:var(--bg); color:var(--text);
               font:16px/1.45 system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial; }}
  .wrap {{ max-width:1000px; margin:24px auto 48px; padding:0 16px; }}
  h1 {{ font-size:1.6rem; margin:0 0 12px; }}
  .subtle {{ color:var(--muted); }}
  .grid {{ display:grid; grid-template-columns: 1.05fr 0.95fr; gap:16px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:14px; box-shadow:0 1px 2px rgba(0,0,0,.04); }}
  .row {{ display:grid; grid-template-columns: 1fr 1fr; gap:12px; margin:8px 0; }}
  label {{ font-weight:600; font-size:.95rem; }}
  input,select {{ width:80%; padding:10px 12px; border:1px solid var(--border); border-radius:10px; background:#fff; }}
  .hint {{ font-size:.85rem; color:var(--muted); margin-top:4px; }}
  .btn {{ display:inline-block; padding:10px 14px; background:var(--accent); color:#fff; border-radius:10px; border:none; cursor:pointer; font-weight:600; }}
  .mono {{ font-variant-numeric: tabular-nums; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .note {{ font-size:.9rem; color:var(--muted); }}
  table {{ width:100%; border-collapse:separate; border-spacing:0; margin-top:6px; }}
  th, td {{ padding:10px 12px; border-bottom:1px solid var(--border); text-align:left; white-space:normal; word-wrap:break-word; }}
  th {{ background:#f3f4f6; }}
  .big {{ font-size:1.2rem; font-weight:700; }}
  @media (max-width: 800px) {{ .grid {{ grid-template-columns: 1fr; }} .row {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="wrap">
  <h1>SMA Cohort-aligned Salary Calculator<br>
    <span class="subtle">Uses your provided <b>Credited Years</b> directly.</span>
  </h1>

  <div class="grid">
    <div class="card">
      <div class="row">
        <div>
          <label>Credited Years</label>
          <input id="yrs" type="number" step="1" value="10">
          <div class="hint">These are the credited external years plus SMA tenured years.</div>
        </div>
        <div>
          <label>Current salary</label>
          <input id="curr" type="number" step="1" value="50000">
          <div class="hint">Used only for COLA comparison</div>
        </div>
      </div>

      <div class="row">
        <div>
          <label>Degree Level</label>
          <select id="deg">
            <option>BA</option><option>MA</option><option>PhD</option>
          </select>
          <div class="hint">MA +{int(cfg['maPct']*100)}%, PhD +{int(cfg['phdPct']*100)}%</div>
        </div>
        <div>
          <label>AP/Prep stipend</label>
          <select id="prep"><option value="0" selected>No</option><option value="1">Yes (+{_money(cfg['prepBonus'])})</option></select>
        </div>
      </div>

      <div class="row">
        <div>
          <label>Skill stipend</label>
          <select id="skillStip"><option value="0" selected>No</option><option value="1">Yes{f' (+{_money(cfg["skillBonus"])})' if cfg['skillBonus'] else ''}</option></select>
        </div>
        <div>
          <label>Leadership stipend</label>
          <select id="leadStip"><option value="0" selected>No</option><option value="1">Yes{f' (+{_money(cfg["leadBonus"])})' if cfg['leadBonus'] else ''}</option></select>
        </div>
      </div>

      <button class="btn" onclick="compute()">Compute</button>
    </div>

    <div class="card">
      <table>
        <thead>
          <tr><th>Step</th><th>Added</th><th>Running total</th></tr>
        </thead>
        <tbody>
          <tr><td>Baseline at credited years</td><td class="mono" id="add0"></td><td class="mono" id="down"></td></tr>
          <tr><td>After degree multiplier</td><td class="mono" id="addDeg"></td><td class="mono" id="degOut"></td></tr>
          <tr><td>+ Prep stipend</td><td class="mono" id="addPrep"></td><td class="mono" id="prepOut"></td></tr>
          <tr><td>+ Skill stipend</td><td class="mono" id="addSkill"></td><td class="mono" id="skillOut"></td></tr>
          <tr><td>+ Leadership stipend</td><td class="mono" id="addLead"></td><td class="mono" id="leadOut"></td></tr>
          <tr><td><b>Cohort-aligned Salary (after cap)</b></td><td class="mono" id="addCap"></td><td class="mono big" id="consCap"></td></tr>
        </tbody>
      </table>
      <hr/>
      <table>
        <tbody>
          <tr><td>COLA salary</td><td class="mono" id="colaOut"></td></tr>
          <tr><td><b>Paid (max of Cohort-aligned or COLA)</b></td><td class="mono big" id="paid"></td></tr>
        </tbody>
      </table>
      <div class="note" style="margin-top:8px">
        Anchors use band midpoints at P{cfg['pTitle']} with +{cfg['inflPct']}% inflation.
      </div>
    </div>
  </div>
</div>

<script>
const CFG = {json.dumps(cfg)};

function money(x) {{
  return (isFinite(x)) ? "$" + Math.round(x).toLocaleString() : "";
}}

function interpWithTail(x, xp, fp) {{
  if (xp.length < 2) return Array(x.length).fill(fp[0]);
  const x0 = xp[0], x1 = xp[xp.length-1];
  const y = new Array(x.length);
  for (let i=0;i<x.length;i++) {{
    const xi = x[i];
    if (xi <= x0) {{
      const mL = (fp[1]-fp[0])/(xp[1]-xp[0]);
      y[i] = fp[0] + mL*(xi - x0);
    }} else if (xi >= x1) {{
      const mR = (fp[fp.length-1]-fp[fp.length-2])/(xp[xp.length-1]-xp[xp.length-2]);
      y[i] = fp[fp.length-1] + mR*(xi - x1);
    }} else {{
      let j=1; while (xp[j] < xi) j++;
      const xL = xp[j-1], xR = xp[j], yL = fp[j-1], yR = fp[j];
      const t = (xi - xL)/(xR - xL);
      y[i] = yL + t*(yR - yL);
    }}
  }}
  return y;
}}

function degreeMultiplier(deg, maPct, phdPct) {{
  const d = String(deg || "").toLowerCase();
  if (d.includes("phd")) return 1 + phdPct;
  if (d.includes("ma"))  return 1 + maPct;
  return 1.0; // BA
}}

function compute() {{
  const yrs  = parseFloat(document.getElementById("yrs").value || "0");
  const deg  = document.getElementById("deg").value;
  const prep = parseInt(document.getElementById("prep").value || "0");
  const skillStip = parseInt(document.getElementById("skillStip").value || "0");
  const leadStip  = parseInt(document.getElementById("leadStip").value  || "0");
  const curr = parseFloat(document.getElementById("curr").value || "0");

  // Baseline @ credited years, then downshift
  const base = interpWithTail([yrs], CFG.xp, CFG.fp)[0];
  const down = base * (1 - CFG.preDown);
  document.getElementById("add0").textContent = "";        // nothing "added" here
  document.getElementById("down").textContent = money(down);

  // Degree multiplier
  const mult = degreeMultiplier(deg, CFG.maPct, CFG.phdPct);
  const afterDeg = down * mult;
  document.getElementById("addDeg").textContent = money(afterDeg - down);
  document.getElementById("degOut").textContent = money(afterDeg);

  // Stipends (raw + running total shown)
  const addPrep = prep ? CFG.prepBonus : 0;
  const afterPrep = afterDeg + addPrep;
  document.getElementById("addPrep").textContent = money(addPrep);
  document.getElementById("prepOut").textContent = money(afterPrep);

  const addSkill = skillStip ? CFG.skillBonus : 0;
  const afterSkill = afterPrep + addSkill;
  document.getElementById("addSkill").textContent = money(addSkill);
  document.getElementById("skillOut").textContent = money(afterSkill);

  const addLead = leadStip ? CFG.leadBonus : 0;
  const afterLead = afterSkill + addLead;
  document.getElementById("addLead").textContent = money(addLead);
  document.getElementById("leadOut").textContent = money(afterLead);

  // Cap
  const capped = Math.min(afterLead, CFG.maxSalary);
  document.getElementById("addCap").textContent = (capped < afterLead) ? ("−" + money(afterLead - capped)) : "";
  document.getElementById("consCap").textContent = money(capped);

  // COLA compare
  const colaSal = curr * (1 + CFG.colaFloor);
  document.getElementById("colaOut").textContent = money(colaSal);

  const paid = Math.max(colaSal, capped);
  document.getElementById("paid").textContent = money(paid);
}}

compute();
</script>
</body>
</html>
"""
    import os
    os.makedirs(os.path.dirname(path_html), exist_ok=True)
    with open(path_html, "w", encoding="utf-8") as f:
        f.write(html)