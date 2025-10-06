# src/viz.py

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import textwrap

from .cohort import build_band_bins_closed, interpolate_salary_strict
from src.models.consultant import consultant_predict_capped

# --- add near the top of src/viz.py ---
def _fmt_money(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return ""

def _money_100(x: float) -> str:
    # nearest $100, no decimals
    try:
        return f"${int(round(float(x)/100.0)*100):,}"
    except Exception:
        return ""


# ----------------------------
# 1) Plot cohort bands at a target percentile
# ----------------------------

# src/viz.py (in _DEFAULT_COLORS)
_DEFAULT_COLORS = {
    "Real": "#1f77b4",
    "RPB":  "#d62728",
    "NLM":  "#2ca02c",
    "PW":   "#9467bd",
    "PWR":  "#17becf",
    "CONS": "#8c564b",
    "CONS_CAP": "#9467bd",   # NEW: magenta/pink
    "CONS_Cap_Real": "#e377c2",   # NEW
}

def _wrap_html(text: str, width: int = 150) -> str:
    """
    Insert <br> line breaks so Plotly annotation text wraps.
    Respects existing <br> by wrapping each line separately.
    """
    lines = text.split("<br>")
    wrapped = ["<br>".join(textwrap.fill(l, width=width).splitlines()) for l in lines]
    return "<br>".join(wrapped)

def plot_bands(
    long: pd.DataFrame,
    target_percentile: float,
    plus_minus_pct: float,
    path_png: Optional[str] = None,
    title_suffix: str = "Cohort",
    staff: Optional[pd.DataFrame] = None,             # NEW: allow overlay scatters
    years_col: str = "Years of Exp",
    cols_dict: Optional[Dict[str, str]] = None,
    colors: Optional[Dict[str, str]] = None,
):
    """
    Draw cohort benchmark bands at a target percentile.
    Optionally overlay scatterplots for staff models (like plot_scatter_models).
    """
    bins, bands_order = build_band_bins_closed(long)

    # Compute per-band target salaries
    segs = []
    for start, end, label in bins:
        y = interpolate_salary_strict(long, label, float(target_percentile))
        if np.isnan(y):
            continue
        segs.append((start, end, label, float(y)))

    if not segs:
        raise RuntimeError("No bands to plot (check `long` and target_percentile).")

    # Determine plot bounds
    # --- Determine plot bounds (SAFE to open-ended bands) ---
    # 1) take only finite band ends
    finite_ends = [e for _, e, _, _ in segs if np.isfinite(e)]
    bands_max = max(finite_ends) if finite_ends else 40

    # 2) if staff provided, include their max years so scatters aren't clipped
    staff_max = 0
    if staff is not None and years_col in staff.columns:
        staff_years = pd.to_numeric(staff[years_col], errors="coerce")
        if staff_years.notna().any():
            staff_max = float(staff_years.max())

    # 3) final x upper bound (cap at least 40)
    max_x = int(max(40.0, bands_max, staff_max))
    min_y = float(np.nanmin([s for *_, s in segs]))
    max_y = float(np.nanmax([s for *_, s in segs]))
    y_pad = 0.08 * (max_y - min_y if max_y > min_y else max_y)

    fig, ax = plt.subplots(figsize=(12, 6))

    # --- draw bands ---
    for start, end, label, val in segs:
        x0, x1 = start, (end if np.isfinite(end) else start + 5)
        ax.hlines(val, x0, x1, linewidth=4, color="C0")
        y_lo = val * (1 - plus_minus_pct)
        y_hi = val * (1 + plus_minus_pct)
        ax.hlines(y_lo, x0, x1, linewidth=1.5, linestyles="dashed", color="C0")
        ax.hlines(y_hi, x0, x1, linewidth=1.5, linestyles="dashed", color="C0")
        ax.vlines([x0, x1], ymin=y_lo, ymax=y_hi, colors="gray", linewidth=0.6, alpha=0.35)
        mid_x = (x0 + x1) / 2.0
        ax.text(mid_x, val, label, ha="center", va="bottom", fontsize=9)

    # --- overlay scatters if staff provided ---
    if staff is not None and cols_dict is not None:
        colors = {**_DEFAULT_COLORS, **(colors or {})}
        x = pd.to_numeric(staff[years_col], errors="coerce").to_numpy()
        jitter = np.linspace(-0.15, 0.15, num=max(3, len(cols_dict)))

        for i, (label, col) in enumerate(cols_dict.items()):
            if col not in staff.columns:
                continue
            y = pd.to_numeric(staff[col], errors="coerce").to_numpy()
            m = np.isfinite(x) & np.isfinite(y)
            if not np.any(m):
                continue
            ax.scatter(
                x[m] + (jitter[i] if i < len(jitter) else 0.0),
                y[m],
                s=24,
                alpha=0.85,
                color=colors.get(label, None),
                label=label,
            )

        ax.legend(loc="lower right")

    # cosmetics
    ax.set_xlim(0, max_x)
    ax.set_ylim(min_y - y_pad, max_y + y_pad)
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary ($)")
    ax.set_title(
        f"Benchmark Salary Bands @ P{int(round(target_percentile))} ({title_suffix})\n"
        f"Dashed lines at ±{plus_minus_pct*100:.0f}%"
    )
    ax.grid(axis="y", alpha=0.2)
    ax.set_xticks(np.arange(0, max_x + 1, 5))

    fig.tight_layout()

    if path_png:
        fig.savefig(path_png, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig, ax


# ----------------------------
# 2) Scatter: Real and model columns vs years
# ----------------------------

def plot_scatter_models(
    staff: pd.DataFrame,
    years_col: str,
    cols_dict: Dict[str, str],
    path_png: Optional[str] = None,
    title: str = "Salaries vs Years of Experience",
    colors: Optional[Dict[str, str]] = None,
    alpha: float = 0.85,
    size: float = 24,
):
    """
    Scatter-plot multiple salary series vs. years.

    Args:
        staff: dataframe with at least years_col and the provided salary columns
        years_col: e.g., "Years of Exp"
        cols_dict: mapping {display_label: column_name_in_staff}
                   e.g., {"Real":"25-26 Salary", "NLM":"Model NL Salary", "RPB":"RPB Salary"}
        path_png: optional save path
        colors: optional mapping {display_label: hex/rgb}; defaults applied if absent
        alpha, size: scatter style
    """
    if years_col not in staff.columns:
        raise ValueError(f"Missing years column: {years_col!r}")

    colors = {**_DEFAULT_COLORS, **(colors or {})}

    # Build plot df (drop rows missing years)
    df = staff[[years_col] + list(cols_dict.values())].copy()
    df[years_col] = pd.to_numeric(df[years_col], errors="coerce")
    df = df.dropna(subset=[years_col])

    if df.empty:
        raise RuntimeError("No rows to plot after cleaning years.")

    x = df[years_col].to_numpy()
    fig, ax = plt.subplots(figsize=(13, 7))

    # Slight horizontal jitter to reduce overlap of identical x
    jitter = np.linspace(-0.15, 0.15, num=max(3, len(cols_dict)))

    for i, (label, col) in enumerate(cols_dict.items()):
        if col not in df.columns:
            continue
        y = pd.to_numeric(df[col], errors="coerce").to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        if not np.any(m):
            continue
        ax.scatter(x[m] + (jitter[i] if i < len(jitter) else 0.0),
                   y[m], s=size, alpha=alpha,
                   color=colors.get(label, None),
                   label=label)

    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary ($)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)

    max_years = max(40, int(np.nanmax(x)))
    ax.set_xticks(np.arange(0, max_years + 1, 5))

    # y-lims with headroom
    all_y = []
    for col in cols_dict.values():
        if col in df.columns:
            all_y.append(pd.to_numeric(df[col], errors="coerce").to_numpy())
    if all_y:
        ycat = np.concatenate(all_y)
        ymin, ymax = np.nanmin(ycat), np.nanmax(ycat)
        pad = 0.06 * (ymax - ymin if ymax > ymin else max(ymax, 1))
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.legend(loc="lower right")
    fig.tight_layout()

    if path_png:
        fig.savefig(path_png, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig, ax


# ----------------------------
# 3) Percentiles by band (bar chart)
# ----------------------------

def plot_percentiles_bar(
    per_band_pct_df: pd.DataFrame,
    include: List[str],
    colors: Dict[str, str],
    path_png: Optional[str] = None,
    title: str = "Achieved Cohort Percentiles (by Band)",
    ylim: tuple[float, float] = (0, 100),
):
    """
    Bar chart of achieved percentiles by band for selected models.

    Args:
        per_band_pct_df: DataFrame index = band labels, columns = model names (e.g., "Real","NLM","RPB"),
                         or the transpose thereof. This function expects
                         rows = bands, cols = models. If your table is models x bands,
                         pass per_band_pct_df.T.
        include: list of column/model names to include (order = legend order).
        colors: mapping {model_name: color}
        path_png: optional save path.
        title: title string
        ylim: y-axis bounds, default (0, 100)
    """
    if per_band_pct_df.empty:
        raise RuntimeError("Empty per_band_pct_df.")

    # If data looks transposed (models x bands), flip it so rows=bands, cols=models
    # Heuristic: if include are the index values, transpose.
    if all(k in per_band_pct_df.index for k in include) and not all(k in per_band_pct_df.columns for k in include):
        data = per_band_pct_df.T.copy()
    else:
        data = per_band_pct_df.copy()

    # Restrict to requested models and drop all-NaN columns
    cols = [c for c in include if c in data.columns]
    if not cols:
        raise ValueError("None of the requested models in `include` are present in per_band_pct_df.")

    data = data[cols].copy()
    data = data.dropna(how="all")  # drop bands that are all NaN

    bands = list(data.index)
    n_bands = len(bands)
    n_models = len(cols)

    x = np.arange(n_bands, dtype=float)
    width = min(0.8 / max(n_models, 1), 0.22)

    fig, ax = plt.subplots(figsize=(max(12, n_bands * 1.2), 6))

    for i, model in enumerate(cols):
        vals = pd.to_numeric(data[model], errors="coerce").to_numpy(dtype=float)
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, vals, width=width,
               label=model, color=colors.get(model, None), alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(bands, rotation=0)
    ax.set_ylim(*ylim)
    ax.set_ylabel("Cohort Percentile")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower right")

    fig.tight_layout()

    if path_png:
        fig.savefig(path_png, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig, ax


def plot_scatter_models_interactive(
    staff: pd.DataFrame,
    years_col: str,
    cols_dict: Dict[str, str],
    path_html: str,
    title: str = "Salaries vs Years of Experience (interactive)",
    colors: Optional[Dict[str, str]] = None,
    hover_cols: Optional[List[str]] = None,
    marker_size: int = 9,
    *,
    long: Optional[pd.DataFrame] = None,
    target_percentile: Optional[float] = None,
    plus_minus_pct: float = 0.10,
    inflation: float = 0.0,
    band_color: str = "rgba(44,160,44,0.85)",
    summary_info: Optional[dict] = None,
    per_band_table: Optional[pd.DataFrame] = None,  # <— the DataFrame you already compute
):
    if years_col not in staff.columns:
        raise ValueError(f"Missing years column: {years_col!r}")

    # --- build hover fields ---------------------------------------------------
    # start with caller list or sensible defaults
    if hover_cols is None:
        hover_cols = [c for c in
                      ["Employee", "ID", "Seniority", "Education Level", "Level", "Category"]
                      if c in staff.columns]

    # ensure stipend/ratings are shown when available (don’t duplicate)
    def _maybe_add(col_names: List[str]):
        for c in col_names:
            if c in staff.columns and c not in hover_cols:
                hover_cols.append(c)
                break

    _maybe_add(["Prep", "Prep Rating"])
    _maybe_add(["Skill Rating"])
    _maybe_add(["Leadership Rating", "Knowledge Rating"])

    pal = {**_DEFAULT_COLORS, **(colors or {})}
    fig = go.Figure()
    x_all = pd.to_numeric(staff[years_col], errors="coerce").to_numpy()

    # --- model scatters -------------------------------------------------------
    ymins, ymaxs = [], []
    jitter = np.linspace(-0.15, 0.15, num=max(3, len(cols_dict)))

    # small random jitter controls
    x_jitter_years = 0.12          # ±0.12 years
    y_jitter_pct   = 0.003         # ±0.3% of salary
    y_jitter_abs   = 150.0         # or at least ±$150
    rng = np.random.default_rng(12345)  # fixed seed for repeatability

    # by default only these traces visible (others start hidden in legend)
    _default_visible = {"Real", "CONS_Cap_Real"}

    for i, (label, col) in enumerate(cols_dict.items()):
        if col not in staff.columns:
            continue
        y = pd.to_numeric(staff[col], errors="coerce").to_numpy()
        m = np.isfinite(x_all) & np.isfinite(y)
        if not np.any(m):
            continue

        # --- build hover text from true (unjittered) values ---
        hdf = staff.loc[m, hover_cols].copy() if hover_cols else pd.DataFrame(index=staff.index[m])
        hdf["Years"] = x_all[m]
        hdf["Salary"] = y[m]
        cd = hdf.to_numpy()
        yrs_idx = len(hover_cols)
        sal_idx = len(hover_cols) + 1

        text = []
        for row in cd:
            lines = [
                f"<b>{label}</b>",
                f"Salary: {row[sal_idx]:,.0f}",
                f"Years: {row[yrs_idx]:.1f}",
            ]
            for k, c in enumerate(hover_cols):
                lines.append(f"{c}: {row[k]}")
            text.append("<br>".join(lines))

        # --- random jitter (applied to plotted positions only) ---
        n = int(m.sum())
        # center-of-trace horizontal offset you already had
        base_x = x_all[m] + (jitter[i] if i < len(jitter) else 0.0)
        # add small random ± jitter
        x_rand = (rng.random(n) * 2.0 - 1.0) * x_jitter_years
        y_span = np.maximum(y[m] * y_jitter_pct, y_jitter_abs)  # scale jitter by salary
        y_rand = (rng.random(n) * 2.0 - 1.0) * y_span

        x_plot = base_x + x_rand
        y_plot = y[m] + y_rand

        fig.add_trace(go.Scatter(
            x=x_plot,
            y=y_plot,
            mode="markers",
            name=label,
            marker=dict(size=marker_size, color=pal.get(label)),
            text=text,
            hovertemplate="%{text}<extra></extra>",
            visible=True if label in _default_visible else "legendonly",
        ))
        ymins.append(np.nanmin(y[m])); ymaxs.append(np.nanmax(y[m]))

    # --- determine x-range (use staff; add bands later) ----------------------
    x_right = max(40, int(np.nanmax(x_all))) if np.isfinite(x_all).any() else 40

    # --- optional flat band overlay ------------------------------------------
    band_ymins, band_ymaxs = [], []
    if long is not None and target_percentile is not None:
        bins, _ = build_band_bins_closed(long)
        finite_bins = [(s, e, lab) for (s, e, lab) in bins if np.isfinite(e)]
        if finite_bins:
            x_right = max(x_right, int(max(e for (_, e, _) in finite_bins)))

        first_drawn = True
        for start, end, label in finite_bins:
            x0, x1 = float(start), float(end)
            y = interpolate_salary_strict(long, label, float(target_percentile))
            if pd.isna(y):
                continue
            y = float(y) * (1.0 + float(inflation))
            y_lo = y * (1.0 - plus_minus_pct)
            y_hi = y * (1.0 + plus_minus_pct)

            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y, y],
                mode="lines",
                line=dict(width=3, color=band_color),
                name=f"Bands @P{int(round(target_percentile))}",
                legendgroup="bands",
                showlegend=first_drawn,
                hoverinfo="skip",
                visible=True,  # bands visible by default
            ))
            for yd in (y_lo, y_hi):
                fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[yd, yd],
                    mode="lines",
                    line=dict(width=1.5, color=band_color, dash="dash"),
                    name="± band",
                    legendgroup="bands",
                    showlegend=False,
                    hoverinfo="skip",
                    visible=True,
                ))
            band_ymins.extend([y_lo, y, y_hi])
            band_ymaxs.extend([y_lo, y, y_hi])
            first_drawn = False

    # --- layout / axes --------------------------------------------------------
    fig.update_layout(
        title=title,
        xaxis_title="Years of Experience",
        yaxis_title="Salary ($)",
        template="plotly_white",
        legend=dict(bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)", borderwidth=1),
        hovermode="closest",
        hoverlabel=dict(namelength=-1),
    )
        # --- optional small per-band table overlay (bottom-right) ---
    if per_band_table is not None and not per_band_table.empty:
        # Keep it small: choose just a few columns you want to show
        # Example expects columns like ["Real","CONS","CONS_CAP","CONS_Cap_Real","COLA 2%"]
        # Rename to short labels for the display
        rename_cols = {
            "Real": "Real",
            "CONS": "Cons",
            "CONS_CAP": "Cap",
            "CONS_Cap_Real": "CCR",
            "COLA 2%": "COLA",
        }
        # Try to preserve only columns that exist
        show_cols = [c for c in ["Real","CONS","CONS_CAP","CONS_Cap_Real","COLA 2%"] if c in per_band_table.columns]
        tbl = per_band_table[show_cols].copy()
        tbl = tbl.rename(columns=rename_cols)

        # Truncate: no decimals; coerce safely
        tbl = tbl.apply(lambda s: pd.to_numeric(s, errors="coerce").round(0).astype("Int64"))

        # Build HTML table
        header_cells = "".join([f"<th style='padding:4px 8px;text-align:right;'>{c}</th>" for c in tbl.columns])
        body_rows = []
        for idx, row in tbl.iterrows():
            # band label might be the index; use str(idx)
            cells = "".join([f"<td style='padding:2px 8px;text-align:right;'>{'' if pd.isna(v) else int(v)}</td>"
                             for v in row.to_list()])
            body_rows.append(
                f"<tr><td style='padding:2px 8px;text-align:left;'>{idx}</td>{cells}</tr>"
            )

        # small_html = (
        #     "<div style='font-family:ui-monospace,monospace;font-size:11px;"
        #     "background:rgba(255,255,255,0.9);padding:6px 8px;border:1px solid rgba(0,0,0,.2);"
        #     "border-radius:6px;'>"
        #     "<div style='font-weight:700;margin-bottom:4px;'>Median achieved %ile by band</div>"
        #     "<table style='border-collapse:collapse;'>"
        #     "<tr><th style='padding:4px 8px;text-align:left;'>Band</th>"
        #     f"{header_cells}</tr>"
        #     f"{''.join(body_rows)}"
        #     "</table></div>"
        # )

    Ymins = ymins + band_ymins
    Ymaxs = ymaxs + band_ymaxs
    if Ymins and Ymaxs:
        ymin, ymax = float(np.nanmin(Ymins)), float(np.nanmax(Ymaxs))
        pad = 0.06 * (ymax - ymin if ymax > ymin else max(abs(ymax), 1.0))
        fig.update_yaxes(range=[ymin - pad, ymax + pad])

    fig.update_xaxes(dtick=5, range=[0, x_right + 1])

    # --- summary box (your enhanced version stays as-is) ----------------------
    if summary_info:
        params = summary_info.get("params", {}) or {}
        models = summary_info.get("models", {}) or {}
        lines = []

        # Accept several possible keys for target, then fall back to function arg
        tgt  = (params.get("target_percentile")
                or params.get("target_percentile_state")
                or params.get("target_percentile_local")
                or target_percentile)
        inf  = params.get("target_inflation")
        cola = params.get("cola")

        # NEW: extra knobs
        down = params.get("downshift_pct")        # fraction, e.g. 0.03
        ma   = params.get("deg_ma_pct")           # fraction
        phd  = params.get("deg_phd_pct")          # fraction
        prep = params.get("prep_bonus")           # dollars
        skbo = params.get("skill_bonus")          # dollars
        lebo = params.get("leadership_bonus")     # dollars

        def _pct(x):
            try:
                return f"{float(x)*100:.1f}%"
            except Exception:
                return None

        if any(v is not None for v in (tgt, inf, cola, down, ma, phd, prep, skbo, lebo)):
            lines.append("<b>Parameters</b>")
            if tgt  is not None: lines.append(f"Target: P{int(round(float(tgt)))}")
            if inf  is not None: lines.append(f"Inflation: {_pct(inf)}")
            if cola is not None: lines.append(f"COLA floor: {_pct(cola)}")

            knob_bits = []
            if down is not None:
                knob_bits.append(f"Baseline adj: −{_pct(down)}")
            if ma is not None or phd is not None:
                bump = []
                if ma  is not None:  bump.append(f"MA +{_pct(ma)}")
                if phd is not None: bump.append(f"PhD +{_pct(phd)}")
                knob_bits.append(", ".join(bump))
            if prep is not None:
                knob_bits.append(f"Prep stipend: {_fmt_money(float(prep))}")
            if skbo is not None and float(skbo) != 0:
                knob_bits.append(f"Skill stipend: {_fmt_money(float(skbo))}")
            if lebo is not None and float(lebo) != 0:
                knob_bits.append(f"Leadership stipend: {_fmt_money(float(lebo))}")

            # break into rows of 3 items each
            if knob_bits:
                for i in range(0, len(knob_bits), 3):
                    lines.append(" · ".join(knob_bits[i:i+3]))
            lines.append("")

        if models:
            lines.append("<b>Models</b>")
            for label, d in models.items():
                parts = [f"{label}:"]
                if "total_cost" in d: parts.append(f"Cost {_fmt_money(float(d['total_cost']))}")
                if "mean_pct"   in d: parts.append(f"Mean %ile {float(d['mean_pct']):.1f}")
                if "num_raised" in d: parts.append(f"Raised {int(d['num_raised'])}")
                # If you also pass mean_pct_state/local, show them too (optional):
                if "mean_pct_state" in d: parts.append(f"State {float(d['mean_pct_state']):.1f}")
                if "mean_pct_local" in d and d["mean_pct_local"] is not None:
                    parts.append(f"Local {float(d['mean_pct_local']):.1f}")
                lines.append("  " + " | ".join(parts))

        text = "<br>".join(lines)

        fig.add_annotation(
            x=0.02, y=0.98, xref="paper", yref="paper" ,
            xanchor="left", yanchor="top",
            align="left", showarrow=False,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.25)", borderwidth=1,
            text=text, font=dict(size=12)
        )

    # --- optional small per-band table overlay (bottom-right) ---
    if per_band_table is not None and not per_band_table.empty:
        # Keep only the columns you want and rename
        rename_cols = {
            "Real": "Real",
            "CONS": "Cons",
            "CONS_CAP": "Cap",
            "CONS_Cap_Real": "CCR",
            "COLA 2%": "COLA",
        }
        wanted = [c for c in ["Real","CONS","CONS_CAP","CONS_Cap_Real","COLA 2%"]
                if c in per_band_table.columns]
        if not wanted:
            wanted = list(per_band_table.columns)[:8]

        tbl = per_band_table[wanted].copy().rename(columns=rename_cols)

        # Truncate to whole numbers
        def _to_int_series(s: pd.Series) -> pd.Series:
            return pd.to_numeric(s, errors="coerce").round(0).astype("Int64")
        for c in tbl.columns:
            tbl[c] = _to_int_series(tbl[c])

        # Keep it compact
        tbl = tbl.iloc[:10]

        # Build lists for Table: first column = band labels (index)
        band_labels = tbl.index.astype(str).tolist()
        value_cols  = [band_labels] + [tbl[c].astype(str).replace("<NA>", "").tolist()
                                    for c in tbl.columns]

        fig.add_trace(go.Table(
            header=dict(
                values=["Band"] + list(tbl.columns),
                align="center",
                fill_color="rgba(242,242,242,1)",
                font=dict(size=12, color="#333"),
                height=24,
            ),
            cells=dict(
                values=value_cols,
                align=["right"] + ["center"] * len(tbl.columns),
                height=22,
            ),
            domain=dict(x=[0.66, 0.98], y=[0.02, 0.30]),  # bottom-right
            name="Band medians",
            hoverinfo="skip",   # optional
            visible=True
        ))
    
    import os
    os.makedirs(os.path.dirname(path_html), exist_ok=True)
    fig.write_html(path_html, include_plotlyjs="cdn", full_html=True)
    return fig

def cons_cap_overview_html(
    long: pd.DataFrame,
    *,
    # policy / model knobs
    target_percentile: float = 50.0,
    inflation: float = 0.04,
    deg_ma_pct: float = 0.02,
    deg_phd_pct: float = 0.04,
    prep_bonus: float = 2500.0,
    max_salary: float = 100_000.0,
    cola_floor: float = 0.02,
    # years policy
    outside_cap_years: int = 10,
    skill_extra_years: int = 5,      # additional credit if Skill==1
    skill_col_name: str = "Skill",   # your 0/1 skill flag column
    # example person (optional; if None, a default is used)
    example: Optional[Dict] = None,  # dict with years, seniority, degree, prep, skill
    # where to write
    path_html: str = "outputs/figures/cons_cap_overview.html",
    # optional 0-year anchor for consultant method (if you use it)
    base_start: float | None = None,
    pre_degree_down_pct: float = 0.03,
    auto_downshift: bool = False,   # if you compute k from mix elsewhere, keep False here
):
    """
    Render a concise HTML explainer for the Consultant (capped) plan, with:
      • policy text (years logic, model logic, raise rule)
      • cohort target table (inflation-adjusted, rounded, CAPPED)
      • worked example using cohort data
    """
    # ---------------- 1) Cohort table (inflation-adjusted & stop at cap) ----------------
    def _to_plus_label(start: float) -> str:
        return f"{int(round(start))}+ yrs"

    rows: list[tuple[str, float]] = []
    bins, _ = build_band_bins_closed(long)

    cap = float(max_salary)
    hit_cap = False

    for start, end, label in bins:
        s = interpolate_salary_strict(long, label, float(target_percentile))
        if np.isnan(s):
            continue
        v = float(s) * (1.0 + float(inflation))  # inflation-adjusted

        if v >= cap and not hit_cap:
            # first time we hit the cap: collapse this and all later bands into one "start+ yrs" row
            rows.append((_to_plus_label(start), cap))
            hit_cap = True
            break
        elif not hit_cap:
            rows.append((label, v))
        else:
            break

    col_hdr = f"P{int(round(target_percentile))} target (+{inflation*100:.0f}%)"
    coh = pd.DataFrame(rows, columns=["Experience Band", col_hdr])
    coh[col_hdr] = coh[col_hdr].apply(_money_100)  # nearest $100 for display

    # --- compute the same baseline factor the model uses ---
    if auto_downshift:
        # If you want an auto mix-based factor here, pass it in or compute consistently.
        # For now keep it fixed, same as your model default:
        k = max(0.0, 1.0 - float(pre_degree_down_pct))
    else:
        k = max(0.0, 1.0 - float(pre_degree_down_pct))

    rows = []
    bins, _ = build_band_bins_closed(long)

    def _to_plus_label(start: float) -> str:
        return f"{int(round(start))}+ yrs"

    cap = float(max_salary)
    hit_cap = False

    for start, end, label in bins:
        s = interpolate_salary_strict(long, label, float(target_percentile))
        if np.isnan(s):
            continue

        # raw cohort target -> inflate -> apply model baseline downshift
        v = float(s) * (1.0 + float(inflation))
        v = v * k

        # cap & collapse when we first hit the cap
        if v >= cap and not hit_cap:
            label = _to_plus_label(start)
            v = cap
            rows.append((label, v))
            hit_cap = True
            break
        elif not hit_cap:
            rows.append((label, v))
        else:
            break

    # make a clear header that these are model-adjusted targets
    # adj_txt = f" (−{pre_degree_down_pct*100:.0f}% baseline adj)" if pre_degree_down_pct else ""
    col_hdr = f"P{int(round(target_percentile))} model baseline (+{inflation*100:.0f}%)"#{adj_txt}"

    coh = pd.DataFrame(rows, columns=["Experience Band", col_hdr])
    coh[col_hdr] = coh[col_hdr].apply(_money_100)

    table = go.Table(
        header=dict(
            values=["Experience Band", col_hdr],
            fill_color="#f2f2f2",
            align="left",
            font=dict(size=12, color="#333"),
            height=28,
        ),
        cells=dict(
            values=[coh["Experience Band"], coh[col_hdr]],
            align="left",
            height=26,
        ),
        # bottom, nearly full width
        domain=dict(x=[0.06, 0.94], y=[0.05, 0.35]),
    )

    # ---------------- 2) Policy text blocks (wrapped, friendly language) ---------------
    years_text = (
        "<b>How we count experience</b><br>"
        "We start with each person’s total years in the profession, then subtract the time they’ve already "
        "worked in our district. We give credit for up to "
        f"<b>{outside_cap_years}</b> years of outside experience. If someone has a recognized skill endorsement, "
        f"we’ll allow up to <b>{outside_cap_years + skill_extra_years}</b> years. We then add their St. Mary's Academy "
        "seniority. This adjusted total is the ‘credited years’ we use for salaries."
    )

    cons_text = (
        "<b>How the cohort-aligned salary is set</b><br>"
        "We look at peer schools and set target salaries for each experience band, adjust those targets for inflation, "
        "and connect them to make a smooth salary curve. Advanced degrees add to the base: "
        f"<b>MA +{deg_ma_pct*100:.0f}%</b>, <b>PhD +{deg_phd_pct*100:.0f}%</b> (BA does not change the base). "
        f"If someone teaches an AP/Prep course, they get an additional flat bonus of {_fmt_money(prep_bonus)}. "
        f"No one’s salary can exceed <b>{_fmt_money(max_salary)}</b> — that’s the maximum under this model."
    )

    raise_text = (
        "<b>How raises are applied</b><br>"
        "We compare each person’s cohort-aligned salary to their current salary with a cost-of-living adjustment. "
        f"The cost-of-living adjustment is {cola_floor*100:.0f}%. Each person receives whichever is higher: the "
        "cohort-aligned salary or the COLA raise. This ensures no one is left behind while aligning most salaries with the model."
    )

    # ---------------- 3) Worked example (calculated via live function) -----------------
    if example is None:
        example = dict(years=18.0, seniority=2.0, degree="MA", prep=1, skill=0, current=73_000)

    ex_df = pd.DataFrame([{
        "Years of Exp": float(example.get("years")),
        "Seniority": float(example.get("seniority")),
        "Education Level": str(example.get("degree")),
        "Prep": int(example.get("prep")),
        skill_col_name: int(example.get("skill")),
        "Current": float(example.get("current")),
    }])

    # --- compute credited years for the example ---
    total_exp = float(ex_df.loc[0, "Years of Exp"])
    sen       = float(ex_df.loc[0, "Seniority"])
    skill     = int(ex_df.get(skill_col_name, 0).iloc[0]) if skill_col_name in ex_df.columns else 0

    outside_exp = max(0.0, total_exp - sen)
    max_outside = outside_cap_years + (skill_extra_years if skill else 0)
    credited_outside = min(outside_exp, max_outside)
    credited_years = sen + credited_outside

    # IMPORTANT: use credited years in the model call
    ex_df.loc[0, "Years of Exp"] = credited_years

    ex_cons_cap = consultant_predict_capped(
        ex_df, long,
        target_percentile=target_percentile,
        inflation=inflation,
        base_start=base_start,
        deg_ma_pct=deg_ma_pct,
        deg_phd_pct=deg_phd_pct,
        max_salary=max_salary,
        prep_col="Prep",
        prep_bonus=prep_bonus,
    ).iloc[0]

    ex_cola = ex_df.loc[0, "Current"] * (1.0 + float(cola_floor))
    ex_paid = max(ex_cons_cap, ex_cola)

    # Pull values once (avoids KeyError on a hard-coded column name)
    yrs   = float(ex_df.at[0, "Years of Exp"])
    sen   = float(ex_df.at[0, "Seniority"])
    deg   = str(ex_df.at[0, "Education Level"])
    prep  = int(ex_df.at[0, "Prep"])
    skill = int(ex_df.at[0, skill_col_name])  # <-- use the configured column
    curr  = float(ex_df.at[0, "Current"])

    # --- Compute credited years for the example person ---
    total_exp = yrs
    outside_exp = max(0.0, total_exp - sen)

    # Apply outside cap and skill bump
    max_outside = outside_cap_years + (skill_extra_years if skill else 0)
    credited_outside = min(outside_exp, max_outside)

    credited_years = sen + credited_outside

    ex_lines = [
        "<b>Worked example</b>",
        f"• Features: Years={yrs:.1f}, Seniority={sen:.1f}, Degree={deg}, Prep={prep}, Skill Endorsement={skill}, ",
        f"• Credited Years (used in model) = {credited_years:.1f}",
        f"• 25-26 Salary = {_fmt_money(curr)}",
        "",
        f"• Cohort-aligned Salary = { _fmt_money(ex_cons_cap) }",
        f"• {int(cola_floor*100)}% COLA Salary = { _fmt_money(ex_cola) }",
        "",
        f"• <b>26-27 Salary </b> (max of the two) = <b>{ _fmt_money(ex_paid) }</b>",
    ]
    example_text = "<br>".join(ex_lines)

    # Wrap for better readability in the annotation box (narrower width for shorter lines)
    years_text_wrapped   = _wrap_html(years_text)
    cons_text_wrapped    = _wrap_html(cons_text)
    raise_text_wrapped   = _wrap_html(raise_text)
    example_text_wrapped = _wrap_html(example_text)

    narrative_html = "<br><br>".join([
        years_text_wrapped,
        cons_text_wrapped,
        raise_text_wrapped,
        example_text_wrapped
    ])

    # ---------------- 4) Compose layout (narrative middle/top, table bottom) ----------
    fig = go.Figure()
    fig.add_trace(table)

    # Table header above the table
    fig.add_annotation(
        x=0.06, y=0.37, xref="paper", yref="paper",
        text="<b>Cohort targets</b> (inflation-adjusted & capped)",
        showarrow=False, align="left", font=dict(size=14)
    )

    # Narrative box near the top, left-aligned, auto-wrapped
    fig.add_annotation(
        x=0.06, y=.95, xref="paper", yref="paper",
        text=narrative_html, showarrow=False, align="left",
        xanchor="left", yanchor="top",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="rgba(0,0,0,0.15)", borderwidth=1,
        font=dict(size=12)
    )

    # Title & margins
    fig.update_layout(
        template="plotly_white",
        width=1100, height=800,                 # a bit taller to give the table room
        margin=dict(l=30, r=30, t=60, b=28),
        title=dict(
            text=(
                "Cohort-aligned plan overview<br>"
                f"Target=P{int(round(target_percentile))} · Inflation={inflation*100:.0f}% · "
                f"MA=+{deg_ma_pct*100:.0f}% · PhD=+{deg_phd_pct*100:.0f}% · "
                f"Prep bonus={_fmt_money(prep_bonus)} · Cap={_fmt_money(max_salary)} · "
                f"COLA floor={cola_floor*100:.0f}%"
            ),
            x=0.0, xanchor="left"
        ),
    )

    # Write file
    import os
    os.makedirs(os.path.dirname(path_html), exist_ok=True)
    fig.write_html(path_html, include_plotlyjs="cdn", full_html=True)
    return fig

def plot_scatter_for_cohort(
    staff: pd.DataFrame,
    years_col: str,
    cols_dict: Dict[str, str],
    *,
    long: pd.DataFrame,
    cohort_label: str,
    path_html: str,
    target_percentile: float,
    inflation: float,
    plus_minus_pct: float = 0.10,
    band_color: str = "rgba(44,160,44,0.85)",
    summary_info: Optional[dict] = None,
    marker_size: int =  9,  # matches your default
    hover_cols: Optional[List[str]] = None,
):
    title = f"Salaries vs Years of Experience — {cohort_label}"
    return plot_scatter_models_interactive(
        staff,
        years_col=years_col,
        cols_dict=cols_dict,
        path_html=path_html,
        title=title,
        long=long,
        target_percentile=target_percentile,
        inflation=inflation,
        plus_minus_pct=plus_minus_pct,
        band_color=band_color,
        summary_info=summary_info,
        marker_size=marker_size,
        hover_cols=hover_cols,
    )