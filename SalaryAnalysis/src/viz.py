# src/viz.py

from __future__ import annotations
from typing import Dict, List, Iterable, Optional
from .cohort import build_band_bins_closed, band_range_closed, interpolate_salary_strict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- add near the top of src/viz.py ---
def _fmt_money(x: float) -> str:
    try:
        return f"${x:,.0f}"
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
    "CONS_CAP": "#e377c2",   # NEW: magenta/pink
}

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
    summary_info: Optional[dict] = None,   # textbox only; no bands
):
    if years_col not in staff.columns:
        raise ValueError(f"Missing years column: {years_col!r}")

    if hover_cols is None:
        hover_cols = [c for c in
                      ["Employee","ID","Seniority","Education Level","Level","Category"]
                      if c in staff.columns]

    pal = {**_DEFAULT_COLORS, **(colors or {})}
    fig = go.Figure()
    x_all = pd.to_numeric(staff[years_col], errors="coerce").to_numpy()

    # --- model scatters only (safe hover text) ---
    ymins, ymaxs = [], []
    jitter = np.linspace(-0.15, 0.15, num=max(3, len(cols_dict)))

    for i, (label, col) in enumerate(cols_dict.items()):
        if col not in staff.columns:
            continue

        y = pd.to_numeric(staff[col], errors="coerce").to_numpy()
        m = np.isfinite(x_all) & np.isfinite(y)
        if not np.any(m):
            continue

        # Build customdata (for your own reference if you ever need it)
        hdf = staff.loc[m, hover_cols].copy() if hover_cols else pd.DataFrame(index=staff.index[m])
        hdf["Years"] = x_all[m]
        hdf["Salary"] = y[m]
        cd = hdf.to_numpy()

        # Indices (no negatives!)
        yrs_idx = len(hover_cols)
        sal_idx = len(hover_cols) + 1

        # Pre-rendered per-point hover text (HTML)
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

        fig.add_trace(go.Scatter(
            x=x_all[m] + (jitter[i] if i < len(jitter) else 0.0),
            y=y[m],
            mode="markers",
            name=label,
            marker=dict(size=marker_size, color=pal.get(label)),
            text=text,                                # <- use text
            hovertemplate="%{text}<extra></extra>",   # <- render text verbatim
            # customdata=cd,  # keep if you want for future use
        ))

        ymins.append(np.nanmin(y[m]))
        ymaxs.append(np.nanmax(y[m]))

    # Layout (no band overlay)
    fig.update_layout(
        title=title,
        xaxis_title="Years of Experience",
        yaxis_title="Salary ($)",
        template="plotly_white",
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1),
        hovermode="closest",   # <- keep per-point hover crisp
        hoverlabel=dict(namelength=-1)
    )

    if ymins and ymaxs:
        ymin, ymax = float(np.nanmin(ymins)), float(np.nanmax(ymaxs))
        pad = 0.06 * (ymax - ymin if ymax > ymin else max(ymax, 1))
        fig.update_yaxes(range=[ymin - pad, ymax + pad])

    if np.isfinite(x_all).any():
        max_years = max(40, int(np.nanmax(x_all)))
        fig.update_xaxes(dtick=5, range=[0, max_years])

    # Summary textbox (optional; doesn’t block hover)
    if summary_info:
        params = summary_info.get("params", {})
        models = summary_info.get("models", {})
        lines = []
        tgt = params.get("target_percentile")
        inf = params.get("target_inflation")
        cola = params.get("cola")
        if tgt is not None or inf is not None or cola is not None:
            lines += ["<b>Parameters</b>"]
            if tgt  is not None: lines.append(f"Target: P{int(round(float(tgt)))}")
            if inf  is not None: lines.append(f"Inflation: {float(inf)*100:.1f}%")
            if cola is not None: lines.append(f"COLA floor: {float(cola)*100:.1f}%")
            lines.append("")
        if models:
            lines.append("<b>Models</b>")
            for label, d in models.items():
                parts = [f"{label}:"]
                if "total_cost" in d: parts.append(f"Cost {_fmt_money(float(d['total_cost']))}")
                if "mean_pct"   in d: parts.append(f"Mean %ile {float(d['mean_pct']):.1f}")
                if "num_raised" in d: parts.append(f"Raised {int(d['num_raised'])}")
                lines.append("  " + " | ".join(parts))
        text = "<br>".join(lines)

        fig.add_annotation(
            x=0.02, y=0.98, xref="paper", yref="paper",  # upper-left
            xanchor="left", yanchor="top",
            align="left", showarrow=False,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.25)", borderwidth=1,
            text=text, font=dict(size=12)
        )

    import os
    os.makedirs(os.path.dirname(path_html), exist_ok=True)
    fig.write_html(path_html, include_plotlyjs="cdn", full_html=True)
    return fig