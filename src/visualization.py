"""Shared chart style and the plots used in more than one place."""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from . import config as cfg

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = {"aggregate": BLUE, "wm": ORANGE, "pred": AQUA}


def apply_style() -> None:
    """Thin marks, recessive grid and axes, one sans-serif face."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlesize": 11.5, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "lines.linewidth": 1.4, "legend.frameon": False, "legend.fontsize": 9,
    })


def save(fig, name: str) -> None:
    """Write a figure to ``outputs/figures`` and keep it visible in the notebook."""
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.FIGURES / f"{name}.png")


def format_time_axis(ax, fmt: str = "%d %b\n%H:%M") -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))


def before_after(raw: pd.Series, cleaned: pd.Series, title: str, name: str, zoom_day: pd.Timestamp):
    """Raw vs cleaned: full study by hourly maximum, plus the day of the worst raw reading."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [2, 1]})
    day_end = zoom_day + pd.Timedelta(days=1)
    panels = [(raw, "Before cleaning", BLUE, 0), (cleaned, "After cleaning", ORANGE, 1)]
    for series, label, color, row in panels:
        series = series.sort_index()
        full, zoom = axes[row, 0], axes[row, 1]
        full.plot(series.resample("1h").max(), color=color, linewidth=0.7, label=f"{label}, hourly maximum")
        full.set_title(f"{label}: hourly maximum, full study")
        full.set_ylabel("Power (W)")
        full.set_xlabel("Date (UK local time)")
        full.legend(loc="upper left")
        zoom.plot(series[(series.index >= zoom_day) & (series.index < day_end)], color=color, linewidth=0.9, label=label)
        zoom.set_title(f"{label}: {zoom_day:%d %b %Y}")
        zoom.set_xlabel("Time (UK local)")
        zoom.legend(loc="upper left")
        zoom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        zoom.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        full.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.suptitle(title, x=0.01, ha="left", fontweight="semibold", fontsize=13)
    fig.tight_layout()
    save(fig, name)
    return fig
