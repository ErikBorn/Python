# --- Calculator HTML writer (standalone, no server) ---------------------------
from __future__ import annotations
from typing import Optional, Dict
import json
import numpy as np
import pandas as pd
from .cohort import build_band_bins_closed, interpolate_salary_strict

def write_cohort_aligned_calculator_html(
    long: pd.DataFrame,
    *,
    # model knobs (use your config values when calling)
    target_percentile: float = 70.0,
    inflation: float = 0.02,
    pre_degree_down_pct: float = 0.03,   # baseline downshift before degree bumps
    deg_ma_pct: float = 0.02,
    deg_phd_pct: float = 0.04,
    prep_bonus: float = 2500.0,
    # fixed policy values (not shown as inputs)
    max_salary: float = 100_000.0,
    cola_floor: float = 0.02,
    # credited years policy
    outside_cap_years: int = 10,
    skill_extra_years: int = 5,
    # where to write
    path_html: str = "outputs/figures/cohort_aligned_calculator.html",
):
    """
    Write a self-contained HTML calculator for the Cohort-aligned (CONS_CAP) model.
    UI shows only: years, seniority, degree, prep, skill, current salary.
    COLA floor and cap are applied internally; not editable in the UI.
    """

    # Build midpoint anchors (same as consultant model; no cap here)
    bins, _ = build_band_bins_closed(long)
    xp, fp = [], []
    for start, end, label in bins:
        # midpoint of band; treat open-ended as start+2.5
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
        "maxSalary": float(max_salary),
        "colaFloor": float(cola_floor),
        "outsideCap": int(outside_cap_years),
        "skillExtra": int(skill_extra_years),
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
  :root {{
    --bg:#f7f9fc; --card:#ffffff; --text:#1f2d3d; --muted:#6b7280; --accent:#0f766e;
    --border:#e5e7eb;
  }}
  html,body {{ margin:0; padding:0; background:var(--bg); color:var(--text);
               font:16px/1.45 system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial; }}
  .wrap {{ max-width:1000px; margin:24px auto 48px; padding:0 16px; }}
  h1 {{ font-size:1.6rem; margin:0 0 12px; }}
  .subtle {{ color:var(--muted); }}
  .grid {{ display:grid; grid-template-columns: 1.05fr 0.95fr; gap:16px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px;
           padding:14px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
  .row {{ display:grid; grid-template-columns: 1fr 1fr; gap:12px; margin:8px 0; }}
  label {{ font-weight:600; font-size:.95rem; }}
  input,select {{ width:80%; padding:10px 12px; border:1px solid var(--border);
                  border-radius:10px; background:#fff; }}
  .hint {{ font-size:.85rem; color:var(--muted); margin-top:4px; }}
  .btn {{ display:inline-block; padding:10px 14px; background:var(--accent); color:#fff;
          border-radius:10px; border:none; cursor:pointer; font-weight:600; }}
  .out {{ font:600 1.05rem/1.3 system-ui; }}
  .big {{ font-size:1.25rem; }}
  table {{ width:100%; border-collapse:separate; border-spacing:0; margin-top:6px; }}
  th, td {{ padding:10px 12px; border-bottom:1px solid var(--border); text-align:left;
           white-space:normal; word-wrap:break-word; }}
  th {{ background:#f3f4f6; }}
  .mono {{ font-variant-numeric: tabular-nums; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .note {{ font-size:.9rem; color:var(--muted); }}
  @media (max-width: 800px) {{
    .grid {{ grid-template-columns: 1fr; }}
    .row {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <h1>SMA Cohort-aligned Salary Calculator for 2026-27 Teaching Faculty<br>
    <span class="subtle">A calculator that applies SMA's 26-27 salary adjustment logic.</span>
  </h1>

  <div class="grid">
    <div class="card">
      <div class="row">
        <div>
          <label>Total years in profession</label>
          <input id="yrs" type="number" step="1" value="10">
          <div class="hint">e.g., 11</div>
        </div>
        <div>
          <label>SMA seniority (years)</label>
          <input id="sen" type="number" step="1" value="2">
          <div class="hint">years at SMA</div>
        </div>
      </div>

      <div class="row">
        <div>
          <label>Degree Level</label>
          <select id="deg">
            <option>BA</option>
            <option>MA</option>
            <option>PhD</option>
          </select>
          <div class="hint">MA adds {int(cfg['maPct']*100)}%, PhD adds {int(cfg['phdPct']*100)}%</div>
        </div>
        <div>
          <label>AP/Prep course (flat bonus)</label>
          <select id="prep">
            <option value="0" selected>No</option>
            <option value="1">Yes (+{_money(cfg['prepBonus'])})</option>
          </select>
        </div>
      </div>

      <div class="row">
        <div>
          <label>Skill endorsement</label>
          <select id="skill">
            <option value="0" selected>No</option>
            <option value="1">Yes (+{cfg['skillExtra']} yrs credit)</option>
          </select>
          <div class="hint">Credits max {cfg['outsideCap']} yrs outside exp,<br> (+{cfg['skillExtra']} if endorsed)</div>
        </div>
        <div>
          <label>Current salary</label>
          <input id="curr" type="number" step="1" value="50000">
          <div class="hint">used for COLA comparison</div>
        </div>
      </div>

      <button class="btn" onclick="compute()">Compute</button>
    </div>

    <div class="card">
      <div class="out">Credited years used by model: <span id="oy" class="mono"></span></div>
      <table>
        <tbody>
          <!-- Collapsed baseline: already downshifted internally -->
          <tr><td>Baseline (interpolated at credited years)</td><td class="mono" id="down"></td></tr>
          <tr><td>After degree bump</td><td class="mono" id="degOut"></td></tr>
          <tr><td>+ Prep bonus</td><td class="mono" id="prepOut"></td></tr>
          <tr><td><b>Cohort-aligned Salary</b></td><td class="mono big" id="consCap"></td></tr>
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
        Baseline is adjusted before degree bumps.
      </div>
    </div>
  </div>
</div>

<script>
const CFG = {json.dumps(cfg)};

function money(x) {{
  if (!isFinite(x)) return "";
  return "$" + Math.round(x).toLocaleString();
}}

// linear interp with tails: right uses last-segment slope; left uses first-segment slope
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

function creditedYears(totalYears, seniority, outsideCap, skillExtra) {{
  const outside = Math.max(totalYears - seniority, 0);
  const cap = outsideCap + (skillExtra > 0 ? skillExtra : 0);
  const creditedOutside = Math.min(outside, cap);
  return creditedOutside + seniority;
}}

function degreeMultiplier(deg, maPct, phdPct) {{
  const d = String(deg || "").toLowerCase();
  if (d.includes("phd")) return 1 + phdPct;
  if (d.includes("ma"))  return 1 + maPct;
  return 1.0; // BA
}}

function compute() {{
  const yrs  = parseFloat(document.getElementById("yrs").value || "0");
  const sen  = parseFloat(document.getElementById("sen").value || "0");
  const deg  = document.getElementById("deg").value;
  const prep = parseInt(document.getElementById("prep").value || "0");
  const skill= parseInt(document.getElementById("skill").value || "0");
  const curr = parseFloat(document.getElementById("curr").value || "0");

  const oy = creditedYears(yrs, sen, CFG.outsideCap, skill ? CFG.skillExtra : 0);
  document.getElementById("oy").textContent = oy.toFixed(1);

  // Baseline @ credited years, then downshift (shown as “Baseline”)
  const base = interpWithTail([oy], CFG.xp, CFG.fp)[0];
  const down = base * (1 - CFG.preDown);
  document.getElementById("down").textContent = money(down);

  // Degree bumps (positive-only)
  const mult = degreeMultiplier(deg, CFG.maPct, CFG.phdPct);
  const afterDeg = down * mult;
  document.getElementById("degOut").textContent = money(afterDeg);

  // Prep bonus
  const afterPrep = afterDeg + (prep ? CFG.prepBonus : 0);
  document.getElementById("prepOut").textContent = money(afterPrep);

  // Apply cap (fixed internally)
  const cohortAligned = Math.min(afterPrep, CFG.maxSalary);
  document.getElementById("consCap").textContent = money(cohortAligned);

  // COLA compare (fixed internally)
  const colaSal = curr * (1 + CFG.colaFloor);
  document.getElementById("colaOut").textContent = money(colaSal);

  const paid = Math.max(colaSal, cohortAligned);
  document.getElementById("paid").textContent = money(paid);
}}
compute(); // initial render
</script>
</body>
</html>
"""

    # write file
    import os
    os.makedirs(os.path.dirname(path_html), exist_ok=True)
    with open(path_html, "w", encoding="utf-8") as f:
        f.write(html)