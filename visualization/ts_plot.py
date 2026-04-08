from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.graphics.tsaplots import plot_acf


def _set_cjk_font_fallback() -> None:
    """Force CJK-capable fallback fonts to avoid glyph-missing warnings."""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


_set_cjk_font_fallback()

def plot_acf_by_window(
    ts: pd.DataFrame,
    *,
    window: int,
    lags: int | None = None,
    date_col: str | None = None,
    value_col: str | None = None,
    save_path: str | None = None,
    title: str | None = None,
    show_title: bool = True,
    mode: str = "stem",
    grid: bool = True,
    style: str = "seaborn-v0_8-whitegrid",
    color: str = "#2a6f97",
    ci_color: str = "#a9cce3",
) -> plt.Figure:
    """
    Plot ACF on the latest ``window`` rows of a 2-column time series DataFrame.

    Args:
        ts: DataFrame where column 1 is date and column 2 is value by default.
        window: Number of latest observations to use for ACF.
        lags: Number of lags for ACF. Defaults to ``min(window // 2, 40)``.
        date_col: Optional date column name. Defaults to first column.
        value_col: Optional value column name. Defaults to second column.
        save_path: Optional path to save image.
        title: Optional chart title.
        show_title: If False, do not render chart title.
        mode: "stem" (classic ACF bars) or "line" (ACF curve).
        grid: Whether to draw grid.
        style: Matplotlib style name.
        color: Main line/stem color.
        ci_color: Confidence-band color for ``mode="line"``.

    Returns:
        Matplotlib figure object.
    """
    if ts.shape[1] < 2:
        raise ValueError("ts must have at least two columns: [date, value].")
    if window <= 1:
        raise ValueError("window must be greater than 1.")

    d_col = date_col or ts.columns[0]
    v_col = value_col or ts.columns[1]
    if d_col not in ts.columns or v_col not in ts.columns:
        raise KeyError(f"Missing columns: date_col={d_col}, value_col={v_col}")

    data = ts[[d_col, v_col]].copy()
    data[d_col] = pd.to_datetime(data[d_col], errors="coerce")
    data[v_col] = pd.to_numeric(data[v_col], errors="coerce")
    data = data.dropna(subset=[d_col, v_col]).sort_values(d_col).tail(window)
    if len(data) < 3:
        raise ValueError("Not enough valid rows after cleaning to plot ACF.")

    acf_lags = lags if lags is not None else min(len(data) // 2, 40)
    acf_lags = max(1, min(acf_lags, len(data) - 1))

    try:
        plt.style.use(style)
    except OSError:
        plt.style.use("default")
    _set_cjk_font_fallback()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if mode == "stem":
        plot_acf(data[v_col], lags=acf_lags, ax=ax, zero=False)
    elif mode == "line":
        # line mode: clearer when lag count is large.
        vals, conf = sm_acf(data[v_col].to_numpy(), nlags=acf_lags, alpha=0.05, fft=True)
        lag_idx = np.arange(1, len(vals))
        acf_vals = vals[1:]
        low = conf[1:, 0] - vals[1:]
        high = conf[1:, 1] - vals[1:]
        ax.plot(lag_idx, acf_vals, color=color, linewidth=2.0, label="ACF")
        ax.fill_between(lag_idx, low, high, color=ci_color, alpha=0.35, label="95% CI")
        ax.axhline(0.0, color="#6c757d", linewidth=1.1, linestyle="--")
        ax.set_xlim(1, acf_lags)
        ax.legend(frameon=True, fontsize=9)
    else:
        raise ValueError("mode must be either 'stem' or 'line'.")
    if show_title:
        ax.set_title(
            title
            or f"ACF of {v_col} (window={len(data)}, "
            f"{data[d_col].iloc[0].date()} to {data[d_col].iloc[-1].date()})"
        )
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.tick_params(axis="both", labelsize=10)
    if grid:
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    else:
        ax.grid(False)
    fig.tight_layout()

    if save_path:
        out = Path(save_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig



def plot_multi_series(
    df: pd.DataFrame,
    date_col: str,
    value_cols,
    save_path: str | None = None,
    *,
    titles: list[str] | None = None,
    suptitle: str | None = None,
    show_subplot_titles: bool = True,
    show_suptitle: bool = True,
    style: str = "seaborn-v0_8-whitegrid",
    grid: bool = True,
    figsize_per_row: float = 3.6,
) -> plt.Figure:
    """
    Plot one or multiple grouped time series with optional twin y-axis.

    ``value_cols`` supports:
    - 1D: ``["A", "B", "C"]`` -> one subplot, all on primary y-axis.
    - 2D: ``[["A", "B"], ["C"]]`` -> 2x1 subplots.
    - 2D with twinx split marker ``";"``:
      ``[["A", "B", ";", "C"], ["D"]]`` -> first subplot: A/B on left, C on right.

    Title rules:
    - ``titles`` (subplot titles) must match subplot count when subplot count > 1.
    - When subplot count == 1, ``titles`` is ignored.
    - ``suptitle`` is applied only when subplot count > 2.
    - Set ``show_subplot_titles=False`` / ``show_suptitle=False`` to hide titles.
    """
    if date_col not in df.columns:
        raise KeyError(f"date_col not in DataFrame: {date_col}")

    if not isinstance(value_cols, (list, tuple)) or len(value_cols) == 0:
        raise ValueError("value_cols must be a non-empty list/tuple.")

    # Normalize into groups (each group = one subplot).
    if isinstance(value_cols[0], (list, tuple)):
        groups = [list(g) for g in value_cols]
    else:
        groups = [list(value_cols)]

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col]).sort_values(date_col)

    # Validate columns (ignore separator token ';').
    requested_cols: list[str] = []
    for g in groups:
        for c in g:
            if c == ";":
                continue
            requested_cols.append(c)
    missing = [c for c in requested_cols if c not in data.columns]
    if missing:
        raise KeyError(f"Columns not in DataFrame: {missing}")

    try:
        plt.style.use(style)
    except OSError:
        plt.style.use("default")
    _set_cjk_font_fallback()

    nrows = len(groups)
    if nrows > 1 and titles is not None and len(titles) != nrows:
        raise ValueError(
            f"titles length must equal subplot count ({nrows}), got {len(titles)}."
        )
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(12, max(3.2, figsize_per_row * nrows)),
        sharex=True,
    )
    if nrows == 1:
        axes = [axes]

    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for i, (ax, group) in enumerate(zip(axes, groups)):
        if len(group) == 0:
            continue

        if ";" in group:
            sep = group.index(";")
            left_cols = [c for c in group[:sep] if c != ";"]
            right_cols = [c for c in group[sep + 1 :] if c != ";"]
        else:
            left_cols = [c for c in group if c != ";"]
            right_cols = []

        if not left_cols and not right_cols:
            raise ValueError("Each subplot group must contain at least one column.")

        # Primary y-axis
        handles = []
        labels = []
        for j, col in enumerate(left_cols):
            color = palette[j % len(palette)] if palette else None
            h = ax.plot(
                data[date_col],
                pd.to_numeric(data[col], errors="coerce"),
                label=col,
                linewidth=1.7,
                color=color,
            )[0]
            handles.append(h)
            labels.append(col)

        ax.set_ylabel(", ".join(left_cols) if left_cols else "value")
        if grid:
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        else:
            ax.grid(False)

        # Secondary y-axis (if needed)
        if right_cols:
            ax_r = ax.twinx()
            for j, col in enumerate(right_cols):
                color = palette[(j + len(left_cols)) % len(palette)] if palette else None
                h = ax_r.plot(
                    data[date_col],
                    pd.to_numeric(data[col], errors="coerce"),
                    label=col,
                    linewidth=1.7,
                    linestyle="-.",
                    color=color,
                    alpha=0.9,
                )[0]
                handles.append(h)
                labels.append(f"{col} (right)")
            ax_r.set_ylabel(", ".join(right_cols))

        if show_subplot_titles:
            if nrows == 1:
                pass
            elif titles is not None:
                ax.set_title(titles[i])
            else:
                ax.set_title(f"Series group {i + 1}")
        ax.legend(handles, labels, loc="upper left", frameon=True, fontsize=9)

    axes[-1].set_xlabel(date_col)
    if nrows > 2 and suptitle and show_suptitle:
        fig.suptitle(suptitle, fontsize=13, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
    else:
        fig.tight_layout()

    if save_path:
        out = Path(save_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


