"""Step 4 charts for the notebook and docs/04_research_findings.md (saved as PNG images).

Static images (matplotlib), not Plotly: GitHub shows a PNG inside a Markdown
page or a notebook, but not an interactive Plotly chart.

Design rules used throughout: one colour per data series, thin lines, light
gridlines, and several small charts side by side (small multiples) instead of
one chart with seven colours that are hard to tell apart.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window (also works in the notebook)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834"]  # blue, orange: validated as a colourblind-safe pair

ITEM_NAMES = {"1A": "Risk Factors", "7": "MD&A"}
HORIZON_NAMES = {"abnormal_3m": "3 months", "abnormal_6m": "6 months", "abnormal_12m": "12 months"}
MEASURE_NAMES = {
    "rf_change_cosine": "Risk Factors: word counts", "rf_change_jaccard": "Risk Factors: word sets",
    "rf_change_embedding": "Risk Factors: meaning", "mdna_change_cosine": "MD&A: word counts",
    "mdna_change_jaccard": "MD&A: word sets", "mdna_change_embedding": "MD&A: meaning",
}


def _style() -> None:
    # Matplotlib warns once per run when the fallback font has no semibold; the main font (Segoe UI) has one.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
        "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "legend.frameon": False,
    })


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def topic_trends_chart(trends: pd.DataFrame, path: Path) -> plt.Figure:
    """Small multiples: mentions per 10,000 words by fiscal year, one panel per topic."""
    _style()
    topics = trends.groupby("topic")["per_10k_words"].max().sort_values(ascending=False).index
    cols = 4
    rows = -(-len(topics) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.6 * rows), sharex=True)
    for ax, topic in zip(axes.flat, topics):
        d = trends[trends["topic"] == topic].sort_values("fiscal_year")
        ax.plot(d["fiscal_year"], d["per_10k_words"], color=SERIES[0], marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        last = d.iloc[-1]
        ax.annotate(f"{last['per_10k_words']:.1f}", (last["fiscal_year"], last["per_10k_words"]),
                    xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=8)
        ax.set_title(topic)
        ax.set_ylim(bottom=0)
        ax.set_xticks(d["fiscal_year"])
        ax.set_xticklabels([f"'{y % 100}" for y in d["fiscal_year"]])
        ax.set_xlim(d["fiscal_year"].min() - 0.3, d["fiscal_year"].max() + 0.6)
    for ax in list(axes.flat)[len(topics):]:
        ax.set_visible(False)
    fig.suptitle("Mentions per 10,000 words of the annual report, by fiscal year (each panel has its own scale)",
                 x=0.01, ha="left", color=INK_2, fontsize=9)
    fig.tight_layout()
    _save(fig, path)
    return fig


def change_by_year_chart(text_change: pd.DataFrame, measure: str, path: Path) -> plt.Figure:
    """Median change score per fiscal year, Risk Factors and MD&A, with the middle 50% as a band."""
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for color, (item, name) in zip(SERIES, ITEM_NAMES.items()):
        d = text_change[text_change["item"] == item].groupby("fiscal_year")[measure]
        q = pd.DataFrame({"low": d.quantile(0.25), "mid": d.median(), "high": d.quantile(0.75)}) * 100
        ax.fill_between(q.index, q["low"], q["high"], color=color, alpha=0.10, linewidth=0)
        ax.plot(q.index, q["mid"], color=color, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=name)
        ax.annotate(name, (q.index[-1], q["mid"].iloc[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=8)
        ax.set_xticks(q.index)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Change vs previous year (%)")
    ax.set_title("How much reports change from one year to the next")
    ax.legend(loc="upper right", fontsize=8)
    ax.text(0, -0.2, "Line: median company. Band: middle half of companies. "
            "Change = 1 - cosine similarity of word counts.", transform=ax.transAxes, color=MUTED, fontsize=7.5)
    fig.tight_layout()
    _save(fig, path)
    return fig


def group_returns_chart(groups: dict[str, pd.DataFrame], primary: str, path: Path) -> plt.Figure:
    """One panel per horizon: average abnormal return of each change quintile, with 95% intervals."""
    _style()
    fig, axes = plt.subplots(1, len(groups), figsize=(11, 3.4), sharey=True)
    for ax, (horizon, g) in zip(axes, groups.items()):
        ax.bar(g["group"], g["mean"] * 100, width=0.45, color=SERIES[0], zorder=2)
        ax.errorbar(g["group"], g["mean"] * 100, yerr=g["ci95"] * 100, fmt="none", ecolor=INK_2,
                    elinewidth=1, capsize=3, zorder=3)
        ax.axhline(0, color=AXIS, linewidth=1, zorder=1)
        ax.set_xticks(g["group"])
        ax.set_xticklabels(["1\nleast" if q == 1 else f"{q}\nmost" if q == g["group"].max() else str(q)
                            for q in g["group"]])
        title = HORIZON_NAMES.get(horizon, horizon) + ("  (main test)" if horizon == primary else "")
        ax.set_title(title)
        ax.set_xlabel("Risk Factors change, fifths within each year")
    axes[0].set_ylabel("Average return minus S&P 500 (%)")
    fig.suptitle("Return after the report, by how much its Risk Factors changed. Lines: 95% confidence intervals.",
                 x=0.01, ha="left", color=INK_2, fontsize=9)
    fig.tight_layout()
    _save(fig, path)
    return fig


def scatter_chart(df: pd.DataFrame, x: str, y: str, path: Path, x_label: str, y_label: str) -> plt.Figure:
    """Each report as a dot, with a straight-line fit."""
    _style()
    d = df[[x, y]].dropna()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(d[x] * 100, d[y] * 100, s=14, color=SERIES[0], alpha=0.55, edgecolors=SURFACE, linewidths=0.6)
    slope, intercept = np.polyfit(d[x] * 100, d[y] * 100, deg=1)  # least-squares straight line
    xs = np.array([d[x].min(), d[x].max()]) * 100
    ax.plot(xs, intercept + slope * xs, color=INK_2, linewidth=1.5)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(axis="x")
    fig.tight_layout()
    _save(fig, path)
    return fig


def robustness_chart(grid: pd.DataFrame, path: Path) -> plt.Figure:
    """Every change measure at every horizon: regression coefficient with its 95% interval.

    The main test is drawn in colour and the rest in grey: the reader should see
    one pre-chosen test, and the others as context for how much results vary.
    """
    _style()
    measures, horizons = list(dict.fromkeys(grid["x"])), list(dict.fromkeys(grid["y"]))
    ys = np.arange(len(measures))[::-1]  # first measure at the top
    fig, axes = plt.subplots(1, len(horizons), figsize=(11, 0.42 * len(measures) + 1.6), sharex=True, sharey=True)
    for ax, horizon in zip(np.atleast_1d(axes), horizons):
        d = grid[grid["y"] == horizon].set_index("x").loc[measures]
        colors = [SERIES[0] if primary else MUTED for primary in d["primary"]]
        ax.hlines(ys, d["ci_low"], d["ci_high"], colors=colors, linewidth=2, zorder=2)
        ax.scatter(d["coef_pp"], ys, s=40, color=colors, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.axvline(0, color=INK_2, linewidth=0.9, zorder=1)  # darker than the grid: "no effect" is the reference
        ax.set_yticks(ys)
        ax.set_yticklabels([MEASURE_NAMES.get(m, m) for m in measures])
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_title(HORIZON_NAMES.get(horizon, horizon) + ("  (main test in blue)" if d["primary"].any() else ""))
        ax.set_xlabel("Return per 1 SD more change (pp)")
    fig.suptitle("Regression coefficient of each change measure, with year and sector effects and controls. "
                 "Lines: 95% confidence intervals; crossing 0 = no clear effect.",
                 x=0.01, ha="left", color=INK_2, fontsize=9)
    fig.tight_layout()
    _save(fig, path)
    return fig
