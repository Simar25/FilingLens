"""Step 3.6 - draw a line chart when a SQL result shows a trend.

Whether to draw a chart is decided by simple rules on the result table, not
by the LLM. Rules are free, instant and predictable:
  - The table needs a time column (fiscal_year, year, date or month) with at
    least 3 different values. Two points aren't a trend; the table is enough.
  - One line per company when there's a ticker column, else one line per
    value column.
  - Never two units on one axis. Revenue in $bn and a margin in % on one axis
    would squash one of them flat, and a second y-axis invites false
    comparisons. Only columns with the same unit (the _bn / _pct / _usd ending
    that text_to_sql asks for) share a chart.
  - At most 8 lines, each with its own colour from a fixed palette that was
    checked for colour-blind readers. More lines than that can't be told apart.

Charts are Plotly figures. The `ask` command saves them as HTML files you
can hover over; the web app (Step 6) will show the same figures.
"""

import json
import textwrap
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

TIME_COLUMNS = ("fiscal_year", "year", "date", "month")
SERIES_COLUMNS = ("ticker", "company", "name")
UNITS = {  # column ending -> (axis title, tick prefix, tick suffix, hover format)
    "bn": ("Billions of US dollars", "$", "B", "$%{y:,.2f}B"),
    "pct": ("Percent", "", "%", "%{y:,.1f}%"),
    "usd": ("US dollars", "$", "", "$%{y:,.2f}"),
}
MIN_POINTS = 3
MAX_LINES = 8
MAX_MARKER_POINTS = 30  # dots on every point of a daily price line would be clutter
LABEL_LINES = 4         # up to this many lines also get a name label at their end

# Light and dark versions of the same 8 hues, in a fixed order that keeps neighbours
# distinguishable for colour-blind readers (checked with a palette validator).
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "text": "#52514e", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "text": "#c3c2b7", "grid": "#2c2c2a", "axis": "#383835",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]},
}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def unit_of(column: str) -> str | None:
    ending = column.rsplit("_", 1)[-1].lower() if "_" in column else ""
    return ending if ending in UNITS else None


def readable(column: str) -> str:
    """'gross_margin_pct' -> 'gross margin'."""
    words = column.split("_")
    if unit_of(column):
        words = words[:-1]
    return " ".join(words)


def plan_lines(table: pd.DataFrame) -> tuple[str, list[tuple[str, pd.DataFrame, str]]] | None:
    """(time column, [(line name, rows, value column), ...]), or None when the table isn't a trend."""
    by_name = {c.lower(): c for c in table.columns}
    x = next((by_name[c] for c in TIME_COLUMNS if c in by_name), None)
    if x is None or table[x].nunique() < MIN_POINTS:
        return None
    series = next((by_name[c] for c in SERIES_COLUMNS if c in by_name), None)
    values = [c for c in table.columns
              if c.lower() not in TIME_COLUMNS + SERIES_COLUMNS + ("cik",)
              and pd.api.types.is_numeric_dtype(table[c]) and table[c].notna().any()]
    if not values:
        return None

    first = values[0]
    if series and table[series].nunique() > 1:
        lines = [(str(name), rows, first) for name, rows in table.groupby(series, sort=True)]
    else:
        unit = unit_of(first)
        same_unit = [c for c in values if unit and unit_of(c) == unit] or [first]
        lines = [(readable(c), table, c) for c in same_unit]
    # A company with no values (e.g. a bank's gross margin is never reported) gets no line.
    lines = [(name, rows, value) for name, rows, value in lines if rows[value].notna().any()]
    return (x, lines) if 0 < len(lines) <= MAX_LINES else None


def make_chart(table: pd.DataFrame, title: str) -> go.Figure | None:
    """A line chart of the table, or None if it isn't a trend (see the rules above)."""
    plan = plan_lines(table)
    if plan is None:
        return None
    x, lines = plan
    theme = THEMES["light"]
    column = lines[0][2]
    axis_title, prefix, suffix, hover = UNITS.get(unit_of(column), (readable(column), "", "", "%{y:,}"))
    is_year = not pd.api.types.is_datetime64_any_dtype(table[x]) and pd.api.types.is_numeric_dtype(table[x])

    fig = go.Figure()
    for i, (name, rows, value) in enumerate(lines):
        rows = rows.dropna(subset=[value]).sort_values(x)
        color = theme["series"][i]
        fig.add_trace(go.Scatter(
            x=rows[x], y=rows[value], name=name,
            mode="lines+markers" if len(rows) <= MAX_MARKER_POINTS else "lines",
            line={"color": color, "width": 2},
            marker={"size": 8, "color": color, "line": {"color": theme["surface"], "width": 2}},
            hovertemplate=f"{hover}<extra>{name}</extra>",
        ))

    many = len(lines) > 1
    if many and len(lines) <= LABEL_LINES and not _end_labels_collide(fig):
        for i, trace in enumerate(fig.data):
            fig.add_annotation(x=trace.x[-1], y=trace.y[-1], text=trace.name, showarrow=False,
                               xanchor="left", xshift=10, font={"color": theme["text"], "size": 12})

    # When every line shows the same column (one per company), say which; otherwise the legend names them.
    measure = readable(column) if all(value == column for _, _, value in lines) else ""
    subtitle = " · ".join(p for p in [measure, "by fiscal year" if x.lower() == "fiscal_year" else ""] if p)
    wrapped = "<br>".join(textwrap.wrap(title, 70))
    fig.update_layout(
        title={"text": wrapped + (f"<br><sup>{subtitle}</sup>" if subtitle else ""), "x": 0.01, "xanchor": "left",
               "font": {"color": theme["ink"], "size": 17}},
        font={"family": FONT, "color": theme["text"], "size": 13},
        paper_bgcolor=theme["surface"], plot_bgcolor=theme["surface"],
        hovermode="x unified",  # a vertical line follows the mouse and one tooltip lists every line
        showlegend=many,
        legend={"orientation": "h", "x": 0, "xanchor": "left", "y": 1.02, "yanchor": "bottom", "title": None},
        margin={"l": 70, "r": 90 if many else 30, "t": 110 if many else 90, "b": 60},
        xaxis={"title": "Fiscal year" if x.lower() == "fiscal_year" else readable(x).capitalize(),
               "showgrid": False, "linecolor": theme["axis"], "ticks": "outside", "tickcolor": theme["axis"],
               **({"dtick": 1, "tickformat": "d"} if is_year else {})},
        yaxis={"title": axis_title, "gridcolor": theme["grid"], "gridwidth": 1, "tickprefix": prefix,
               "ticksuffix": suffix, "tickformat": ",", "zeroline": True, "zerolinecolor": theme["axis"]},
    )
    return fig


def _end_labels_collide(fig: go.Figure) -> bool:
    """True when two lines end too close together for their labels to be read (the legend still names them)."""
    ends = sorted(float(t.y[-1]) for t in fig.data if len(t.y))
    values = [float(v) for t in fig.data for v in t.y]
    span = (max(values) - min(values)) or 1.0
    return any(b - a < 0.05 * span for a, b in zip(ends, ends[1:]))


def save_chart(fig: go.Figure, path: Path) -> Path:
    """Write the chart as an HTML file that works offline and follows the computer's dark mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # "directory": one shared copy of Plotly's JavaScript (~4.7 MB) next to the charts,
    # instead of a copy inside every file or a download from the internet every time.
    fig.write_html(path, include_plotlyjs="directory", post_script=_dark_mode_script(fig),
                   config={"displaylogo": False})
    return path


def _dark_mode_script(fig: go.Figure) -> str:
    """JavaScript that recolours the chart with the dark palette when the computer is in dark mode."""
    dark = THEMES["dark"]
    layout = {
        "paper_bgcolor": dark["surface"], "plot_bgcolor": dark["surface"], "font.color": dark["text"],
        "title.font.color": dark["ink"], "xaxis.linecolor": dark["axis"], "xaxis.tickcolor": dark["axis"],
        "yaxis.gridcolor": dark["grid"], "yaxis.zerolinecolor": dark["axis"],
        **{f"annotations[{i}].font.color": dark["text"] for i in range(len(fig.layout.annotations))},
    }
    colors = dark["series"][:len(fig.data)]
    traces = {"line.color": colors, "marker.color": colors, "marker.line.color": dark["surface"]}
    return (
        "document.body.style.margin = '0';"  # the browser's default margin adds a scrollbar
        f"document.body.style.background = '{THEMES['light']['surface']}';"  # no white frame around the chart
        "if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {"
        "  var gd = document.getElementById('{plot_id}');"
        f"  document.body.style.background = '{dark['surface']}';"
        f"  Plotly.relayout(gd, {json.dumps(layout)});"
        f"  Plotly.restyle(gd, {json.dumps(traces)});"
        "}"
    )
