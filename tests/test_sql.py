import pandas as pd
import pytest

from filinglens import database
from filinglens.assistant import NO_DATA_ANSWER, Assistant
from filinglens.charts import make_chart, save_chart
from filinglens.combine import build_prompt
from filinglens.llm import LLMReply
from filinglens.rag.vector_store import SearchResult
from filinglens.router import parse_route, route_question
from filinglens.sql.safety import QueryError, UnsafeQueryError, check_sql, connect_readonly, run_query
from filinglens.sql.schema import describe_schema
from filinglens.sql.text_to_sql import SqlResult, extract_sql, has_data, text_to_sql


# ---------- a small test database ----------

@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.duckdb"
    con = database.connect(path)
    database.replace_rows(con, "companies", pd.DataFrame(
        {"ticker": ["NVDA", "AMD"], "cik": [1045810, 2488], "name": ["NVIDIA CORP", "ADVANCED MICRO DEVICES INC"]}))
    rows = []
    for ticker, revenues in {"NVDA": [10.0, 16.0, 27.0], "AMD": [9.0, 16.0, 23.0]}.items():
        for year, revenue in zip([2021, 2022, 2023], revenues):
            rows += [(ticker, year, "revenue", revenue * 1e9), (ticker, year, "cost_of_revenue", revenue * 0.4e9)]
    financials = pd.DataFrame(rows, columns=["ticker", "fiscal_year", "metric", "value"]).assign(
        unit="USD", period_start=pd.Timestamp("2023-01-01"), period_end=pd.Timestamp("2023-12-31"),
        xbrl_tag="Revenues", accession_number="x", filed_date=pd.Timestamp("2024-02-01"))
    database.replace_rows(con, "financials", financials)
    database.replace_rows(con, "filings", pd.DataFrame({
        "accession_number": ["a1", "a2"], "ticker": ["NVDA", "AMD"], "fiscal_year": [2023, 2023],
        "form": ["10-K", "10-K"], "filing_date": pd.to_datetime(["2023-02-24", "2024-01-31"]),
        "report_date": pd.to_datetime(["2023-01-29", "2023-12-30"]), "document_url": ["u1", "u2"]}))
    database.replace_rows(con, "prices", pd.DataFrame({
        "ticker": ["SPY", "SPY"], "date": pd.to_datetime(["2023-12-29", "2024-12-31"]), "close": [100.0, 125.0]}))
    database.create_views(con)
    con.close()
    return path


@pytest.fixture
def con(db_path):
    con = connect_readonly(db_path)
    yield con
    con.close()


def reply(text: str) -> LLMReply:
    return LLMReply(text, "fake", 100, 10, 0.1)


def scripted_llm(*answers: str):
    """A fake LLM that gives these replies in order and records the prompts it got."""
    remaining = list(answers)
    calls = []

    def generate(system: str, prompt: str) -> LLMReply:
        calls.append((system, prompt))
        return reply(remaining.pop(0))

    generate.calls = calls
    return generate


# ---------- 3.3 safety: check_sql ----------

@pytest.mark.parametrize("sql", [
    "SELECT ticker, revenue FROM financials_wide WHERE ticker = 'NVDA'",
    "WITH r AS (SELECT * FROM financials_wide) SELECT * FROM r",
    "SELECT * FROM financials f JOIN companies c USING (ticker) ORDER BY 1 LIMIT 5",
    "SELECT ticker FROM prices UNION ALL SELECT ticker FROM companies",
    "SELECT * FROM main.prices;",
])
def test_ordinary_queries_pass(sql):
    assert check_sql(sql)


@pytest.mark.parametrize("sql, reason", [
    ("DELETE FROM prices", "Only SELECT"),
    ("DROP TABLE prices", "Only SELECT"),
    ("SELECT 1; DROP TABLE prices", "exactly one statement"),
    ("ATTACH 'other.db' AS other", "Only SELECT"),
    ("COPY prices TO 'stolen.csv'", "Only SELECT"),
    ("SET enable_external_access = true", "Only SELECT"),
    ("SELECT * FROM 'C:/Users/me/secret.csv'", "not allowed"),        # DuckDB reads files named in FROM
    ("SELECT * FROM read_text('.env')", "Table functions"),
    ("SELECT * FROM range(1000000000)", "Table functions"),
    ("SELECT * FROM (SELECT * FROM read_csv('x.csv'))", "Table functions"),
    ("SELECT * FROM chunks", "not allowed"),                            # report text is RAG's job
    ("SELECT * FROM other.main.prices", "Only tables in this database"),
    ("SELECT read_blob('x')", "not allowed"),
    ("SELECT * FROM prices WHERE ticker = (", "could not be parsed"),
])
def test_unsafe_or_broken_queries_are_refused(sql, reason):
    with pytest.raises(UnsafeQueryError, match=reason):
        check_sql(sql)


# ---------- 3.3 safety: the connection and the limits ----------

def test_connection_cannot_write_or_read_files(con, tmp_path):
    # These bypass check_sql on purpose: layer 2 must hold on its own.
    with pytest.raises(QueryError):
        run_query(con, "DELETE FROM prices")
    secret = tmp_path / "secret.txt"
    secret.write_text("API_KEY=abc", encoding="utf-8")
    with pytest.raises(QueryError, match="disabled by configuration"):
        run_query(con, f"SELECT * FROM read_text('{secret.as_posix()}')")
    with pytest.raises(QueryError):
        run_query(con, "SET enable_external_access = true")


def test_row_limit_and_timeout(con):
    table, truncated = run_query(con, "SELECT * FROM range(1000)", max_rows=10)
    assert (len(table), truncated) == (10, True)
    with pytest.raises(QueryError, match="longer than"):
        run_query(con, "SELECT COUNT(*) FROM range(100000) a, range(100000) b, range(100000) c", timeout=0.5)


def test_decimals_become_numbers(con):
    table, _ = run_query(con, "SELECT ROUND(1.234, 2) AS x")
    assert pd.api.types.is_numeric_dtype(table["x"])


# ---------- 3.1 schema ----------

def test_schema_lists_companies_fiscal_year_ends_and_metrics(con):
    schema = describe_schema(con)
    assert "NVDA: NVIDIA CORP (FY2023 ended 2023-01-29)" in schema
    assert "metric is one of: cost_of_revenue, revenue" in schema
    assert "fiscal years 2021 to 2023" in schema


# ---------- 3.2 text-to-SQL ----------

def test_extract_sql_from_code_block_or_bare_text():
    assert extract_sql("Here you go:\n```sql\nSELECT 1;\n```\nDone.") == "SELECT 1"
    assert extract_sql("SELECT 2") == "SELECT 2"


def test_text_to_sql_runs_the_query(con):
    llm = scripted_llm("```sql\nSELECT ticker, fiscal_year, ROUND(gross_margin * 100, 1) AS gm_pct "
                       "FROM financials_wide WHERE ticker = 'NVDA' ORDER BY fiscal_year\n```")
    result = text_to_sql("NVIDIA gross margin?", "schema", con, llm)
    assert result.ok and result.attempts == 1 and not result.errors
    assert result.table["gm_pct"].tolist() == [60.0, 60.0, 60.0]


def test_a_failed_query_goes_back_to_the_llm_once(con):
    llm = scripted_llm("```sql\nSELECT revenue_bn FROM financials_wide\n```",
                       "```sql\nSELECT ROUND(revenue / 1e9, 2) AS revenue_bn FROM financials_wide\n```")
    result = text_to_sql("revenue?", "schema", con, llm)
    assert result.ok and result.attempts == 2
    assert "revenue_bn" in result.errors[0]
    retry_prompt = llm.calls[1][1]
    assert "SELECT revenue_bn FROM financials_wide" in retry_prompt and "failed with this error" in retry_prompt


def test_unsafe_sql_is_never_run_and_gives_up_after_the_retry(con):
    llm = scripted_llm("```sql\nDROP TABLE prices\n```", "```sql\nDELETE FROM prices\n```")
    result = text_to_sql("delete everything", "schema", con, llm)
    assert not result.ok and result.attempts == 2
    assert "Only SELECT" in result.error
    assert con.execute("SELECT COUNT(*) FROM prices").fetchone()[0] == 2  # nothing was touched


def test_cannot_answer_is_respected(con):
    result = text_to_sql("Apple's market share?", "schema", con, scripted_llm("CANNOT_ANSWER: no market share data"))
    assert (result.ok, result.cannot_answer, result.error) == (False, "no market share data", None)


# ---------- 3.4 router ----------

@pytest.mark.parametrize("text, route", [
    ('{"route": "numbers", "reason": "asks for margins"}', "numbers"),
    ('```json\n{"route": "Both", "reason": "x"}\n```', "both"),
    ("text", "text"),
    ("I am not sure", "both"),                      # unreadable -> the safe fallback
    ('{"route": "numbers or text"}', "both"),
])
def test_parse_route(text, route):
    assert parse_route(text)[0] == route


def test_route_question_uses_the_llm():
    decision = route_question("Why did margins fall?", scripted_llm('{"route": "both", "reason": "why"}'))
    assert (decision.route, decision.reason) == ("both", "why")


# ---------- 3.5 combined answers ----------

def test_combined_prompt_has_table_then_sources_then_question():
    result = SqlResult("q", sql="SELECT 1", table=pd.DataFrame({"ticker": ["NVDA"], "revenue_bn": [27.0]}))
    passage = SearchResult("NVDA-2023-7-0001", "NVDA", "NVIDIA CORP", 2023, "7", "MD&A", "Data center grew.", 0.7)
    prompt = build_prompt("Why did revenue grow?", result, [passage])
    assert prompt.index("<database_result>") < prompt.index('<source id="1"') < prompt.index("Question:")
    assert "ticker,revenue_bn\nNVDA,27.0" in prompt


def test_failed_lookup_is_marked_not_available():
    prompt = build_prompt("q", SqlResult("q", cannot_answer="no such data"), [])
    assert "Not available: no such data" in prompt


# ---------- 3.6 charts ----------

def test_trend_gets_one_line_per_company():
    table = pd.DataFrame({"ticker": ["AMD"] * 3 + ["NVDA"] * 3, "fiscal_year": [2021, 2022, 2023] * 2,
                          "gross_margin_pct": [48.0, 45.0, 46.0, 62.0, 57.0, 73.0]})
    fig = make_chart(table, "Gross margins")
    assert [t.name for t in fig.data] == ["AMD", "NVDA"]
    assert fig.layout.yaxis.ticksuffix == "%"


def test_no_chart_without_a_trend():
    two_years = pd.DataFrame({"fiscal_year": [2022, 2023], "revenue_bn": [1.0, 2.0]})
    no_time = pd.DataFrame({"ticker": ["A", "B", "C"], "revenue_bn": [1.0, 2.0, 3.0]})
    assert make_chart(two_years, "t") is None and make_chart(no_time, "t") is None


def test_company_without_values_gets_no_line():
    table = pd.DataFrame({"ticker": ["AAPL"] * 3 + ["JPM"] * 3, "fiscal_year": [2021, 2022, 2023] * 2,
                          "gross_margin_pct": [42.0, 43.0, 44.0, None, None, None]})  # banks report no gross margin
    assert [t.name for t in make_chart(table, "t").data] == ["AAPL"]


def test_different_units_never_share_an_axis():
    table = pd.DataFrame({"fiscal_year": [2021, 2022, 2023], "revenue_bn": [1.0, 2.0, 3.0],
                          "net_income_bn": [0.1, 0.2, 0.3], "net_margin_pct": [10.0, 10.0, 10.0]})
    assert [t.name for t in make_chart(table, "t").data] == ["revenue", "net income"]


def test_saved_chart_works_offline(tmp_path):
    table = pd.DataFrame({"fiscal_year": [2021, 2022, 2023], "revenue_bn": [1.0, 2.0, 3.0]})
    path = save_chart(make_chart(table, "Revenue"), tmp_path / "chart.html")
    assert (tmp_path / "plotly.min.js").exists()
    assert "prefers-color-scheme: dark" in path.read_text(encoding="utf-8")


# ---------- the whole flow, with a fake LLM ----------

def test_numbers_question_end_to_end(db_path):
    llm = scripted_llm(
        '{"route": "numbers", "reason": "revenue figures"}',
        "```sql\nSELECT ticker, fiscal_year, ROUND(revenue / 1e9, 2) AS revenue_bn FROM financials_wide "
        "ORDER BY ticker, fiscal_year\n```",
        "NVIDIA's revenue rose from $10.0B to $27.0B [DB].",
    )
    with Assistant(db_path, generate=llm) as assistant:
        response = assistant.ask("How did NVIDIA's and AMD's revenue grow?")
    assert response.route.route == "numbers" and response.cites_db
    assert [stage for stage, _ in response.calls] == ["router", "SQL", "answer"]
    assert response.chart is not None
    assert "NVDA,2023,27.0" in llm.calls[2][1]  # the answer was written from the real query result


def test_empty_result_needs_no_answer_call(db_path):
    query = "```sql\nSELECT * FROM financials_wide WHERE ticker = 'AAPL'\n```"
    llm = scripted_llm(query, query)  # the retry confirms that no data is the answer
    with Assistant(db_path, generate=llm) as assistant:
        response = assistant.ask("Apple revenue?", route="numbers")
    assert response.text == NO_DATA_ANSWER and len(llm.calls) == 2


def test_an_empty_result_gets_one_retry_with_a_hint(con):
    llm = scripted_llm(
        "```sql\nSELECT ticker, revenue FROM financials_wide WHERE fiscal_year = 2026\n```",  # not reported yet
        "```sql\nSELECT ticker, revenue FROM financials_wide WHERE fiscal_year = 2023\n```",
    )
    result = text_to_sql("revenue in 2026?", "schema", con, llm)
    assert result.attempts == 2 and len(result.table) == 2
    assert "returned no data" in llm.calls[1][1]


def test_only_empty_values_count_as_no_data():
    assert not has_data(pd.DataFrame({"revenue_growth_pct": [None]}))           # the real case from 2026-10-19
    assert not has_data(pd.DataFrame({"ticker": ["AAPL"], "revenue_bn": [None]}))
    assert has_data(pd.DataFrame({"ticker": ["AAPL"], "name": ["Apple Inc."]}))  # a list of names is data
    assert has_data(pd.DataFrame({"ticker": ["AAPL"], "revenue_bn": [394.33]}))


def test_both_route_combines_sql_and_passages(db_path):
    passage = SearchResult("NVDA-2023-7-0001", "NVDA", "NVIDIA CORP", 2023, "7", "MD&A", "Data center grew.", 0.7)
    llm = scripted_llm(
        "```sql\nSELECT ticker, fiscal_year, revenue FROM financials_wide WHERE ticker = 'NVDA'\n```",
        "Revenue reached 27e9 [DB], driven by data center demand [1].",
    )
    with Assistant(db_path, generate=llm) as assistant:
        assistant.search = lambda question, filters, k: [passage]  # no embedding model or vector DB in tests
        response = assistant.ask("How did NVIDIA's revenue change, and why?", route="both")
    assert response.filters.tickers == ["NVDA"]
    assert (response.cites_db, response.cited, response.invalid_citations) == (True, [1], [])
