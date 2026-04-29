from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.ticker import PercentFormatter


STYLE_PATH = Path(__file__).with_name('neurips.mplstyle')

COLORS = {
    'ink': '#1F2933',
    'muted_ink': '#52606D',
    'grid': '#D8E1E8',
    'blue': '#35608D',
    'orange': '#D97706',
    'green': '#4C8C4A',
    'rose': '#C0566F',
    'gray': '#98A2B3',
    'light_gray': '#E8EEF3',
    'sand': '#E7D8B1',
}

FIGURE_SIZES = {
    'single': (3.35, 2.45),
    'single_tall': (3.35, 2.95),
    'double': (6.9, 2.8),
    'double_tall': (6.9, 3.2),
    'double_wide': (7.2, 3.35),
}


def apply_style() -> None:
    plt.style.use(str(STYLE_PATH))
    mpl.rcParams.update({
        'axes.prop_cycle': plt.cycler(color=[COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['rose']]),
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def percent_string(value: float, decimals: int = 1) -> str:
    return f'{100 * value:.{decimals}f}%'


def add_confidence_interval(
    ax: plt.Axes,
    x: float,
    point: float,
    low: float,
    high: float,
    *,
    color: str = '#222222',
) -> None:
    ax.errorbar(
        x,
        point,
        yerr=[[point - low], [high - point]],
        fmt='none',
        ecolor=color,
        elinewidth=1.15,
        capsize=3,
        capthick=1.15,
        zorder=5,
    )


def style_axes(
    ax: plt.Axes,
    *,
    ylabel: str | None = None,
    xlabel: str | None = None,
    ylim: tuple[float, float] | None = None,
    y_as_percent: bool = False,
    grid_axis: str = 'y',
) -> None:
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if y_as_percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis=grid_axis)
    ax.set_axisbelow(True)


def style_panel_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc='left', pad=8)


def add_figure_note(fig: plt.Figure, text: str, *, y: float = 0.01) -> None:
    fig.text(
        0.5,
        y,
        text,
        ha='center',
        va='bottom',
        fontsize=8.2,
        color=COLORS['muted_ink'],
    )

def annotate_bars(
    ax: plt.Axes,
    bars,
    *,
    values: list[float] | np.ndarray | None = None,
    decimals: int = 1,
    fontsize: float = 6.4,
    zero_floor: float = 0.015,
    inside_margin_fraction: float = 0.04,
    small_bar_threshold: float = 0.12,
) -> None:
    if values is None:
        values = [bar.get_height() for bar in bars]

    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    inside_margin = inside_margin_fraction * y_range
    small_cutoff = small_bar_threshold * y_range

    for bar, value in zip(bars, values):
        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        label = percent_string(float(value), decimals=decimals)
        facecolor = bar.get_facecolor()
        text_color = '#FFFFFF' if _relative_luminance(facecolor) < 0.55 else COLORS['ink']
        bbox = None

        if height <= 0:
            y = ylim[0] + zero_floor * y_range
            va = 'bottom'
            text_color = COLORS['ink']
        elif height < small_cutoff:
            y = max(ylim[0] + height * 0.55, ylim[0] + zero_floor * y_range)
            va = 'center'
            text_color = COLORS['ink']
            bbox = {
                'boxstyle': 'round,pad=0.14',
                'facecolor': 'white',
                'edgecolor': 'none',
                'alpha': 0.92,
            }
        else:
            y = height - inside_margin
            va = 'top'

        ax.text(
            x,
            y,
            label,
            ha='center',
            va=va,
            fontsize=fontsize,
            fontweight='semibold',
            color=text_color,
            bbox=bbox,
            zorder=6,
            clip_on=False,
        )

def _relative_luminance(color) -> float:
    r, g, b = to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b
