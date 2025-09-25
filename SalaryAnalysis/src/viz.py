# src/viz.py

from __future__ import annotations
from typing import Dict, List, Iterable, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .cohort import build_band_bins_closed, band_range_closed, interpolate_salary_strict


# ----------------------------
# 1) Plot cohort bands at a target percentile
# ----------------------------

def plot_bands(
    long: pd.DataFrame,
    target_percentile: float,
    plus_minus_pct: float,
    path_png: Optional[str] = None,
    title_suffix: str = "Cohort",
    max_years_override: Optional[int] = None,   # NEW: allow manual cap
):
    bins, bands_order = build_band_bins_closed(long)

    # Compute per-band target salaries
    segs = []
    for start, end, label in bins:
        y = interpolate_salary_strict(long, label, float(target_percentile))
        if np.isnan(y):
            continue
        segs.append((float(start), float(end), str(label), float(y)))

    if not segs:
        raise RuntimeError("No bands to plot (check `long` and target_percentile).")

    # ---- Safe bounds (avoid inf) ----
    # Finite ends (e.g., 5, 10, ..., 40)
    finite_ends = [e for _, e, _, _ in segs if np.isfinite(e)]
    if finite_ends:
        finite_cap = max(finite_ends)
    else:
        finite_cap = 0.0

    # If there’s an open-ended band, draw it as 5-year width from its start
    open_band_draw_caps = [
        s + 5.0 for s, e, _, _ in segs if not np.isfinite(e)
    ]

    # Choose the max x from finite ends, open-band draw caps, and a floor of 40
    computed_max_x = max([40.0, finite_cap] + open_band_draw_caps)  # all finite
    max_x = int(max_years_override if max_years_override is not None else computed_max_x)

    # Y bounds
    vals = [v for *_, v in segs]
    min_y = float(np.nanmin(vals))
    max_y = float(np.nanmax(vals))
    y_pad = 0.08 * (max_y - min_y if max_y > min_y else max_y)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(12, 6))

    for start, end, label, val in segs:
        x0 = start
        x1 = end if np.isfinite(end) else (start + 5.0)  # draw open band as 5-year segment
        ax.hlines(val, x0, x1, linewidth=4, color="C0")

        y_lo = val * (1 - plus_minus_pct)
        y_hi = val * (1 + plus_minus_pct)
        ax.hlines(y_lo, x0, x1, linewidth=1.5, linestyles="dashed", color="C0")
        ax.hlines(y_hi, x0, x1, linewidth=1.5, linestyles="dashed", color="C0")
        ax.vlines([x0, x1], ymin=y_lo, ymax=y_hi, colors="gray", linewidth=0.6, alpha=0.35)

        mid_x = (x0 + x1) / 2.0
        ax.text(mid_x, val, label, ha="center", va="bottom", fontsize=9)

    ax.set_xlim(0, max_x)
    ax.set_ylim(min_y - y_pad, max_y + y_pad)
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary ($)")
    ax.set_title(
        f"Benchmark Salary Bands @ P{int(round(target_percentile))}  ({title_suffix})\n"
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

_DEFAULT_COLORS = {
    "Real": "#1f77b4",   # blue
    "EGB_LM": "#7f7f7f",  # gray
    "EGB_NLM": "#2ca02c", # green
    "SLM": "#ff7f0e",     # orange
    "RPB": "#d62728",     # red
    "NLM": "#2ca02c",     # alias
    "PW": "#9467bd",      # purple
}

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

    ax.legend(loc="best")
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
    ax.legend(loc="best")

    fig.tight_layout()

    if path_png:
        fig.savefig(path_png, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig, ax