"""Step 3.2 - turn a number question into SQL, run it, and retry once on failure.

Why not let the LLM answer number questions directly? It doesn't reliably know
exact figures (they may be newer than its training data or misremembered), and
it is bad at arithmetic over many numbers. A database is exact. So the LLM does
what it is good at, translating the question into a query, and DuckDB does
the lookup and the calculations.

The flow:
  1. The LLM gets the schema description (schema.py), a few example
     question -> SQL pairs, and the question. It replies with one query, or
     with CANNOT_ANSWER when the data can't answer the question.
  2. safety.check_sql checks the query; safety.run_query runs it.
  3. If either step fails, the error message goes back to the LLM with its
     query for ONE more try. Database errors are usually specific ("column
     revenue_bn not found, did you mean revenue?"), so one retry fixes most
     of them. More retries would mostly cost time and free-tier requests.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import duckdb
import pandas as pd

from filinglens.llm import LLMReply
from filinglens.sql.safety import QueryError, check_sql, run_query

MAX_ATTEMPTS = 2  # the first try plus one retry

SYSTEM_PROMPT = """\
You write DuckDB SQL queries that answer questions about US public companies, using this database:

{schema}

How to write the query:
- One SELECT query (WITH is fine). The database is read-only.
- Filter companies by ticker ('AAPL'), never by name.
- Return readable numbers: money in billions of USD rounded to 2 decimals in columns ending in _bn \
(ROUND(revenue / 1e9, 2) AS revenue_bn); fractions as percentages rounded to 1 decimal in columns ending in _pct \
(ROUND(gross_margin * 100, 1) AS gross_margin_pct); per-share amounts and stock prices rounded to 2 decimals in columns \
ending in _usd.
- Do every calculation the answer needs (differences, growth, averages, rankings) in SQL, so the answer can be read \
off the result without any arithmetic.
- Return the inputs of a calculation next to its result (e.g. revenue for each year next to the growth over the \
period), so every result can be checked and a missing input is visible instead of a bare NULL.
- For a value at the start or end of a period, pick that year's value \
(MAX(CASE WHEN fiscal_year = 2022 THEN revenue END)), never MIN(revenue) or MAX(revenue): the smallest value is \
not the first year's value when revenue fell.
- Include ticker and fiscal_year (or date) columns so every number can be identified.
- For several companies or years, return one row per company and year, ordered by ticker, then year. This layout \
can be drawn as a chart.
- Keep the result small (under 100 rows). For prices over long periods, use one row per month: the last close of \
each month.

Reply with only the SQL in a ```sql code block. If only part of the question can be answered from this database, \
write the query for that part: another step answers the rest from the annual reports. If no part of it can be \
answered from this database (the data isn't there), or it asks to change data, reply with one line instead: \
CANNOT_ANSWER: <short reason>

Examples:

Question: Compare NVIDIA's and AMD's gross margins since 2021
```sql
SELECT ticker, fiscal_year, ROUND(gross_margin * 100, 1) AS gross_margin_pct
FROM financials_wide
WHERE ticker IN ('NVDA', 'AMD') AND fiscal_year >= 2021
ORDER BY ticker, fiscal_year
```

Question: Which company had the highest net margin in fiscal 2024?
```sql
SELECT ticker, fiscal_year, ROUND(net_margin * 100, 1) AS net_margin_pct
FROM financials_wide
WHERE fiscal_year = 2024 AND net_margin IS NOT NULL
ORDER BY net_margin DESC
LIMIT 5
```

Question: Did Tesla's stock beat the market in 2024?
```sql
WITH closes AS (
    SELECT ticker,
           ARG_MAX(close, date) FILTER (WHERE date < DATE '2024-01-01') AS start_close,
           ARG_MAX(close, date) FILTER (WHERE date <= DATE '2024-12-31') AS end_close
    FROM prices
    WHERE ticker IN ('TSLA', 'SPY')
    GROUP BY ticker
)
SELECT ticker, ROUND((end_close / start_close - 1) * 100, 1) AS return_2024_pct
FROM closes
ORDER BY ticker
```

Question: What is Apple's market share in smartphones?
CANNOT_ANSWER: the database has financial statements and stock prices, not market share."""

RETRY_PROMPT = """\
Question: {question}

Your previous query:
```sql
{sql}
```
failed with this error:
{error}

Reply with a corrected query in a ```sql code block, or with CANNOT_ANSWER: <reason>."""

SQL_BLOCK = re.compile(r"```(?:sql|duckdb)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
CANNOT_ANSWER = re.compile(r"CANNOT_ANSWER:?\s*(.*)", re.IGNORECASE)
MAX_ERROR_CHARS = 600  # DuckDB errors can include long candidate lists; the first part is what matters
# An empty result is usually a filter mistake (a year that isn't in the data yet, a misspelled ticker), so it
# gets the retry too. Seen on 2026-10-19: "growth from 2022 to 2026" asked for fiscal 2026 and got nothing.
NO_DATA = ("It ran, but returned no data (no rows, or only empty values). Check its filters against the data "
           "available (tickers, fiscal years). If no data really is the answer, reply with the same query.")
LABEL_COLUMNS = {"ticker", "company", "name", "fiscal_year", "year", "date", "month"}


@dataclass
class SqlResult:
    question: str
    sql: str | None = None             # the query that ran, as re-written by safety.check_sql
    table: pd.DataFrame | None = None  # its result
    truncated: bool = False            # True if rows beyond safety.MAX_ROWS were dropped
    cannot_answer: str | None = None   # the LLM's reason, when the data can't answer the question
    errors: list[str] = field(default_factory=list)  # one per failed attempt
    attempts: int = 0
    replies: list[LLMReply] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.table is not None

    @property
    def error(self) -> str | None:
        """Why no query worked (None if one did, or if the LLM said CANNOT_ANSWER)."""
        return self.errors[-1] if self.errors and not self.ok and not self.cannot_answer else None


def extract_sql(text: str) -> str:
    """The SQL from the LLM's reply: the ```sql block, or the whole reply if it's bare SQL."""
    match = SQL_BLOCK.search(text)
    sql = match.group(1) if match else text
    return sql.strip().rstrip(";").strip()


def text_to_sql(question: str, schema: str, con: duckdb.DuckDBPyConnection,
                generate: Callable[[str, str], LLMReply], max_attempts: int = MAX_ATTEMPTS) -> SqlResult:
    """`generate` is llm.generate (or a fake one in tests); `con` should come from safety.connect_readonly."""
    system = SYSTEM_PROMPT.format(schema=schema)
    result = SqlResult(question)
    prompt = f"Question: {question}"
    for attempt in range(1, max_attempts + 1):
        reply = generate(system, prompt)
        result.replies.append(reply)
        result.attempts = attempt

        refusal = CANNOT_ANSWER.search(reply.text)
        if refusal and not SQL_BLOCK.search(reply.text):
            result.cannot_answer = refusal.group(1).strip() or "the database doesn't have this data"
            return result

        sql = extract_sql(reply.text)
        try:
            checked = check_sql(sql)
            table, truncated = run_query(con, checked)
        except QueryError as exc:
            if not result.ok:
                result.sql = sql  # keep the failed query so it can be shown
            result.errors.append(str(exc)[:MAX_ERROR_CHARS])
            prompt = RETRY_PROMPT.format(question=question, sql=sql, error=result.errors[-1])
            continue
        # Keep every query that ran: if the retry fails, an empty result still beats none.
        result.sql, result.table, result.truncated = checked, table, truncated
        if has_data(table):
            return result
        result.errors.append(NO_DATA)
        prompt = RETRY_PROMPT.format(question=question, sql=checked, error=NO_DATA)
    return result


def has_data(table: pd.DataFrame) -> bool:
    """False for no rows, or rows whose values are all empty (labels like ticker and year don't count)."""
    if table.empty:
        return False
    values = [c for c in table.columns if c.lower() not in LABEL_COLUMNS]
    return not values or bool(table[values].notna().to_numpy().any())
