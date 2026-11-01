"""Step 6 - the web app: ask a question, explore the numbers, read the research.

Run:  python -m streamlit run app.py --server.address localhost     then open http://localhost:8501
(--server.address keeps it on this computer: on a network, anyone could spend the API tokens.)

Everything clever lives in filinglens.assistant, the same code the `ask` command uses. This file
only shows the results, so the app and the command can't disagree. The look is a trading terminal
with a lens: .streamlit/config.toml has the colours and font, app.css the shapes, and app.js the
magnifying lens over the market map and the function keys. Online it runs on filinglens/online.py.
"""

import html
import threading

import pandas as pd
import streamlit as st

from filinglens import llm
from filinglens.assistant import Assistant, Response, SetupError
from filinglens.charts import THEMES, make_chart
from filinglens.config import PROJECT_ROOT
from filinglens.online import ONLINE, OnlineAssistant
from filinglens.rag.answer import CITATION

EXAMPLES = ["Compare NVIDIA's and AMD's gross margins since 2021",
            "What supply chain risks did Apple describe in 2024?",
            "How did Intel's revenue change in fiscal 2023, and why?"]
# Label -> (SQL expression, chart unit). Earnings per share is left out on purpose: for companies that
# split their stock, the database mixes pre-split and post-split years (docs/06_web_app.md).
METRICS = {"Revenue": ("revenue / 1e9", "bn"), "Net income": ("net_income / 1e9", "bn"),
           "Operating income": ("operating_income / 1e9", "bn"), "Free cash flow": ("free_cash_flow / 1e9", "bn"),
           "R&D expense": ("rnd_expense / 1e9", "bn"), "Gross margin": ("gross_margin * 100", "pct"),
           "Operating margin": ("operating_margin * 100", "pct"), "Net margin": ("net_margin * 100", "pct"),
           "Revenue growth": ("revenue_growth * 100", "pct")}
MAX_COMPANIES = 8  # the chart's limit: more lines can't be told apart
SOURCE_OF_ROUTE = {"numbers": "the database", "text": "the reports", "both": "the database and the reports"}
MEASURED = pd.DataFrame(
    [("Right passage in the top 5 results", "94%", "50 search questions"),
     ("Every expected figure in the answer", "38 of 40", "40 number questions"),
     ("Right figure when asked why as well", "20 of 20", "20 mixed questions")],
    columns=["What was measured", "Result", "Test set"])
SECTOR_NAMES = {"Communication Services": "MEDIA", "Consumer Cyclical": "RETAIL", "Consumer Defensive": "STAPLES",
                "Financial Services": "FINANCE", "Basic Materials": "MATERIALS", "Technology": "TECH",
                "Healthcare": "HEALTH", "Industrials": "INDUSTRY", "Real Estate": "PROPERTY", "Energy": "ENERGY",
                "Utilities": "UTILITIES"}
GROWTH_BANDS = [(0.20, "up3", "Grew 20% or more"), (0.08, "up2", "Grew 8 to 20%"), (0.0, "up1", "Grew up to 8%"),
                (-0.08, "down1", "Fell up to 8%")]
RESEARCH_CHARTS = [("04_change_by_year.png", "Companies barely rewrite their reports: about 1% of Risk Factors a year."),
                   ("04_topic_trends.png", "AI went from a quarter of reports to 98% in fiscal 2025.")]


def growth_band(growth: float | None) -> tuple[str, str]:
    """(CSS class, legend label) of a revenue growth rate. Whole bands, as on a printed map, not a smooth scale."""
    if growth is None or pd.isna(growth):
        return "none", "Not reported"
    for floor, name, label in GROWTH_BANDS:
        if growth >= floor:
            return name, label
    return "down2", "Fell more than 8%"


def board(assistant: Assistant) -> str:
    """All the companies at a glance, by sector: 99 squares, each a link that starts a question about that company.
    Built from database values only, and names are escaped, so it's safe to show as HTML."""
    rows = assistant.con.execute("""
        SELECT i.sector, i.ticker, w.revenue_growth
        FROM company_info i
        LEFT JOIN financials_wide w ON w.ticker = i.ticker
            AND w.fiscal_year = (SELECT MAX(fiscal_year) FROM financials_wide)
        ORDER BY i.sector, i.market_cap DESC NULLS LAST, i.ticker
    """).df()
    columns = []
    for sector, companies in rows.groupby("sector", sort=False):
        squares = []
        for row in companies.itertuples():
            band, _ = growth_band(row.revenue_growth)
            # One decimal under 1%: a whole number would show a small fall as "-0%".
            figure = "" if band == "none" else f"{row.revenue_growth:+.{abs(row.revenue_growth) < 0.01:d}%}"
            name = html.escape(assistant.companies.get(row.ticker, row.ticker).split(" /")[0], quote=True)
            squares.append(f'<a class="square {band}" href="?company={row.ticker}" target="_self" title="{name}">'
                           f'<b>{row.ticker}</b><span>{figure}</span></a>')
        columns.append(f'<div class="sector"><h4>{SECTOR_NAMES.get(sector, html.escape(sector))}</h4>{"".join(squares)}</div>')
    legend = "".join(f'<span><i class="{name}"></i>{label}</span>' for name, label in
                     [growth_band(g) for g in (0.3, 0.1, 0.01, -0.01, -0.3, None)])
    return f'<div class="board">{"".join(columns)}</div><div class="legend">{legend}</div>'


def tape(assistant: Assistant) -> str:
    """The price tape: last close and last day's move of the largest companies. Drawn twice so it can loop."""
    rows = assistant.con.execute("""
        WITH recent AS (
            SELECT ticker, close, LAG(close) OVER (PARTITION BY ticker ORDER BY date) AS previous,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS age
            FROM prices
        )
        SELECT r.ticker, r.close, r.close / r.previous - 1
        FROM recent r LEFT JOIN company_info i USING (ticker)
        WHERE r.age = 1
        ORDER BY i.market_cap DESC NULLS FIRST
        LIMIT 31
    """).fetchall()
    items = []
    for ticker, close, change in rows:
        move = "" if change is None else f' <i class="{"rise" if change >= 0 else "fall"}">{change:+.1%}</i>'
        items.append(f"<span><b>{html.escape(ticker)}</b> {close:,.2f}{move}</span>")
    return f'<div class="tape" aria-label="Latest closing prices"><div>{"".join(items) * 2}</div></div>'


def masthead(assistant: Assistant) -> str:
    reports, passages, last_day = assistant.con.execute(
        "SELECT (SELECT COUNT(*) FROM filings), (SELECT COUNT(*) FROM chunks), (SELECT MAX(date) FROM prices)").fetchone()
    return (f'<div class="mast"><h1>FILING<b>LENS</b></h1><span>{len(assistant.companies)} companies, {reports} annual '
            f"reports, {passages:,} passages. Prices to {last_day}.</span></div>")


@st.cache_resource
def load() -> tuple[Assistant, threading.Lock, str]:
    """One Assistant for all visitors, so the models load once. The lock is there because its database
    connection can only do one thing at a time: other visitors wait their turn. The header and map are
    built once too; they only change when the data does."""
    assistant = OnlineAssistant() if ONLINE else Assistant()
    return assistant, threading.Lock(), masthead(assistant) + tape(assistant), board(assistant)


def company_labels(assistant: Assistant) -> dict[str, str]:
    """'Apple Inc. (AAPL)' -> 'AAPL'. SEC names can end in a state, like 'AMERICAN TOWER CORP /MA/'."""
    return {f"{name.split(' /')[0]} ({ticker})": ticker for ticker, name in sorted(assistant.companies.items())}


def plain(text: str) -> str:
    """Report text and LLM answers in Streamlit's Markdown: $...$ would become a formula, ~...~ a strikethrough."""
    return text.replace("$", r"\$").replace("~", r"\~")


def ticked(text: str) -> str:
    """Citations as ticks: [DB] and [3] mean this statement was traced to that source."""
    ticks = CITATION.sub(lambda m: f":orange[✓ {m[1]}]", plain(text)).replace("[DB]", ":orange[✓ DB]")
    return ticks.replace("]:orange[", "] :orange[")


def show_chart(chart, tab: str) -> None:
    """The charts in filinglens/charts.py are drawn for a light page; here they get their dark colours."""
    dark = THEMES["dark"]
    chart.update_layout(font_family="IBM Plex Mono", font_color=dark["text"], title_font_family="IBM Plex Mono",
                        title_font_color="#FFA028", paper_bgcolor="#000000", plot_bgcolor="#000000", margin_t=95)
    chart.update_xaxes(linecolor=dark["axis"], tickcolor=dark["axis"])
    chart.update_yaxes(gridcolor=dark["grid"], zerolinecolor=dark["axis"])
    for trace, color in zip(chart.data, dark["series"]):
        trace.line.color = trace.marker.color = color
        trace.marker.line.color = "#000000"
    for note in chart.layout.annotations:
        note.font.color = dark["text"]
    with st.container(key=f"chart_{tab}"):
        st.plotly_chart(chart, theme=None, config={"displayModeBar": False})


def show_answer(response: Response) -> None:
    with st.container(key="answer"):
        with st.container(key="focus"):
            st.markdown(f"IN FOCUS: {plain(response.question)}")
        note, margin = st.columns([5, 2], gap="large")
        with note.container(key="note"):
            st.markdown(ticked(response.text))
        if response.invalid_citations:
            note.warning(f"The answer cites sources that don't exist: {response.invalid_citations}")
        with margin.container(key="sources"):
            st.markdown("**TRACED TO**")
            sql = response.sql
            if sql and sql.sql:
                with st.expander(":orange[✓ DB] Database result"):
                    if sql.ok:
                        st.dataframe(sql.table, hide_index=True)
                    st.code(sql.sql, language="sql")
            for number in response.cited:
                source = response.sources[number - 1]
                with st.expander(f":orange[✓ {number}] {source.label}"):
                    st.markdown(plain(source.text))
            if not (sql and sql.sql) and not response.cited:
                st.caption("This answer cites no source.")
    if response.chart is not None:
        show_chart(response.chart, "ask")
    replies = [reply for _, reply in response.calls]
    seconds = sum(reply.seconds for reply in replies) + response.search_seconds
    tokens = sum(reply.input_tokens + reply.output_tokens for reply in replies)
    st.caption(f"Answered from {SOURCE_OF_ROUTE[response.route.route]} in {seconds:.0f} s, using {tokens:,} tokens.")


def ask_tab(assistant: Assistant, lock: threading.Lock) -> None:
    question_area = st.container()  # drawn above the examples, but run after them: they fill the question in
    picked = False
    with st.container(key="examples"):
        for example in EXAMPLES:
            if st.button(example, type="tertiary"):
                st.session_state.question, picked = example, True
    ticker = st.query_params.get("company")
    if ticker in assistant.companies and st.session_state.get("from_board") != ticker:  # a square of the map was clicked
        name = assistant.companies[ticker].split(" /")[0]
        st.session_state.question, st.session_state.from_board = f"What drove {name}'s revenue in its latest report?", ticker
    with question_area, st.form("ask", border=False):
        field, button = st.columns([11, 1], vertical_alignment="bottom")
        question = field.text_input("Your question", key="question", label_visibility="collapsed",
                                    placeholder="Ask about a figure, the wording of a report, or why something changed")
        asked = button.form_submit_button("GO", type="primary", width="stretch")
    if ONLINE:
        question_area.caption("Name a company in your question: the online version searches that company's reports.")
    if (asked or picked) and question.strip():
        with st.spinner("Reading the reports and the numbers..."), lock:
            try:
                st.session_state.response = assistant.ask(question.strip())
                st.rerun()  # draws the page again with the map folded away, so the answer is near the top
            except (llm.LLMError, SetupError) as exc:
                st.session_state.pop("response", None)
                st.error(str(exc))
    if "response" in st.session_state:  # kept here so moving to another tab doesn't lose the answer
        show_answer(st.session_state.response)


def ledger(table: pd.DataFrame) -> pd.DataFrame:
    """Figures the way a financial statement prints them: losses in brackets, a dash where nothing was reported."""
    def cell(value: float) -> str:
        if pd.isna(value):
            return "—"
        return f"({abs(value):,.2f})" if value < 0 else f"{value:,.2f}"

    return table.apply(lambda column: column.map(cell)).rename_axis(index=None, columns=None)


def explore_tab(assistant: Assistant, lock: threading.Lock) -> None:
    names = company_labels(assistant)
    left, right = st.columns([3, 1])
    chosen = left.multiselect("Companies", list(names), max_selections=MAX_COMPANIES,
                              default=[n for n in names if names[n] in ("NVDA", "AMD", "INTC")])
    label = right.selectbox("Figure", list(METRICS))
    if not chosen:
        st.info("Pick at least one company.")
        return
    expression, unit = METRICS[label]
    column = f"{label.lower().replace(' ', '_').replace('&', 'n')}_{unit}"
    with lock:
        table = assistant.con.execute(
            f"SELECT ticker, fiscal_year, ROUND({expression}, 2) AS {column} FROM financials_wide "
            "WHERE ticker IN (SELECT UNNEST(?)) ORDER BY ticker, fiscal_year", [[names[c] for c in chosen]]).df()
    chart = make_chart(table, f"{label} by company")
    if chart is None:
        st.info("The database has no data for this choice. Banks, for example, report no gross margin.")
        return
    show_chart(chart, "explore")
    st.table(ledger(table.pivot(index="ticker", columns="fiscal_year", values=column)))
    st.caption("US$ billions." if unit == "bn" else "Percent.")


def research_tab() -> None:
    st.subheader("HOW WELL IT WORKS")
    st.write("These results come from test questions with known answers, checked against the database and the reports.")
    st.table(MEASURED.set_index("What was measured").rename_axis(index=None))
    if ONLINE:
        st.caption("Measured on the full version. This online version uses keyword search only (90% in the first row) "
                   "and a different AI model, so its answers may differ.")

    st.subheader("DOES A CHANGE IN AN ANNUAL REPORT PREDICT THE STOCK?")
    st.write("No detectable effect. The finance paper Lazy Prices found that companies whose reports change a lot do "
             "worse afterwards. Across 469 reports from 94 large US companies, a bigger change in Risk Factors didn't "
             "predict the next 6 months' return (rank correlation +0.02, p = 0.64), and none of 18 variations was "
             "significant. This rules out an effect as large as the paper's, but not a small one.")
    st.image(str(PROJECT_ROOT / "docs" / "images" / "04_returns_by_change_group.png"), width="stretch",
             caption="If report changes predicted weak returns, the bars would fall from left to right. They don't.")
    for column, (image, caption) in zip(st.columns(2), RESEARCH_CHARTS):
        column.image(str(PROJECT_ROOT / "docs" / "images" / image), caption=caption, width="stretch")


def main() -> None:
    st.set_page_config(page_title="FilingLens", page_icon=":material/search:", layout="wide")
    st.html(PROJECT_ROOT / "app.css")
    script = (PROJECT_ROOT / "app.js").read_text(encoding="utf-8")
    st.html(f"<script>{script}</script>", unsafe_allow_javascript=True)
    try:
        assistant, lock, header, market_map = load()
    except Exception as exc:  # e.g. the database hasn't been built yet: say so instead of showing a stack trace
        st.error(f"The app couldn't start: {exc}")
        st.stop()
    st.markdown(header, unsafe_allow_html=True)
    with st.container(key="map"), st.expander(f"MARKET MAP: {len(assistant.companies)} companies by sector, by revenue "
                                              "growth in the latest fiscal year",
                                              expanded="response" not in st.session_state):
        st.markdown(market_map, unsafe_allow_html=True)
    ask, explore, research = st.tabs(["F1 ASK", "F2 EXPLORE", "F3 RESEARCH"])
    with ask:
        ask_tab(assistant, lock)
    with explore:
        explore_tab(assistant, lock)
    with research:
        research_tab()


main()
