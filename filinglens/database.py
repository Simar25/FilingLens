"""Step 5 - store everything in DuckDB.

DuckDB is a database that lives in a single file (like SQLite) but is built for
analysis: fast GROUP BYs and window functions, and it can read pandas DataFrames
directly. Later steps query it: text-to-SQL answers number questions from it, and
the research module joins report language with stock returns here.
"""

from pathlib import Path

import duckdb
import pandas as pd

from filinglens.config import DB_PATH

TABLES = ("companies", "filings", "financials", "prices")

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    ticker  VARCHAR PRIMARY KEY,
    cik     INTEGER NOT NULL,          -- SEC's permanent company ID
    name    VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS filings (
    accession_number VARCHAR PRIMARY KEY,  -- SEC's unique ID for the filing
    ticker           VARCHAR NOT NULL,
    fiscal_year      INTEGER NOT NULL,
    form             VARCHAR NOT NULL,
    filing_date      DATE NOT NULL,        -- when it was made public
    report_date      DATE,                 -- last day of the fiscal year it covers
    document_url     VARCHAR NOT NULL,
    local_path       VARCHAR               -- downloaded copy, relative to the project folder
);

CREATE TABLE IF NOT EXISTS financials (
    ticker           VARCHAR NOT NULL,
    fiscal_year      INTEGER NOT NULL,
    metric           VARCHAR NOT NULL,     -- e.g. 'revenue', 'net_income'
    value            DOUBLE NOT NULL,
    unit             VARCHAR NOT NULL,     -- 'USD' or 'USD/shares'
    period_start     DATE,                 -- empty for balance-sheet values
    period_end       DATE NOT NULL,
    xbrl_tag         VARCHAR NOT NULL,     -- which SEC tag the value came from
    accession_number VARCHAR NOT NULL,     -- which filing it came from
    filed_date       DATE NOT NULL,
    PRIMARY KEY (ticker, fiscal_year, metric)
);

CREATE TABLE IF NOT EXISTS sections (
    accession_number VARCHAR NOT NULL,     -- which 10-K the section comes from
    ticker           VARCHAR NOT NULL,
    fiscal_year      INTEGER NOT NULL,
    item             VARCHAR NOT NULL,     -- '1A', '7', ... or 'AR' (attached annual report)
    title            VARCHAR NOT NULL,     -- 'Risk Factors', ...
    text             VARCHAR NOT NULL,
    n_words          INTEGER NOT NULL,
    PRIMARY KEY (accession_number, item)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id         VARCHAR PRIMARY KEY,  -- e.g. 'AAPL-2024-1A-0007'
    accession_number VARCHAR NOT NULL,
    ticker           VARCHAR NOT NULL,
    fiscal_year      INTEGER NOT NULL,
    item             VARCHAR NOT NULL,
    title            VARCHAR NOT NULL,
    chunk_index      INTEGER NOT NULL,     -- position within its section
    text             VARCHAR NOT NULL,
    n_words          INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS prices (
    ticker  VARCHAR NOT NULL,
    date    DATE NOT NULL,
    open    DOUBLE,
    high    DOUBLE,
    low     DOUBLE,
    close   DOUBLE NOT NULL,               -- adjusted for splits and dividends
    volume  BIGINT,
    PRIMARY KEY (ticker, date)
);
"""

# One row per company-year with metrics as columns, plus common ratios. This is
# the table people (and later the text-to-SQL module) will query most.
FINANCIALS_WIDE_VIEW = """
CREATE OR REPLACE VIEW financials_wide AS
WITH pivoted AS (
    SELECT
        ticker,
        fiscal_year,
        MAX(value) FILTER (WHERE metric = 'revenue')             AS revenue,
        MAX(value) FILTER (WHERE metric = 'cost_of_revenue')     AS cost_of_revenue,
        MAX(value) FILTER (WHERE metric = 'gross_profit')        AS gross_profit_reported,
        MAX(value) FILTER (WHERE metric = 'rnd_expense')         AS rnd_expense,
        MAX(value) FILTER (WHERE metric = 'operating_income')    AS operating_income,
        MAX(value) FILTER (WHERE metric = 'net_income')          AS net_income,
        MAX(value) FILTER (WHERE metric = 'eps_diluted')         AS eps_diluted,
        MAX(value) FILTER (WHERE metric = 'operating_cash_flow') AS operating_cash_flow,
        MAX(value) FILTER (WHERE metric = 'capex')               AS capex,
        MAX(value) FILTER (WHERE metric = 'total_assets')        AS total_assets,
        MAX(value) FILTER (WHERE metric = 'total_liabilities')   AS total_liabilities,
        MAX(value) FILTER (WHERE metric = 'stockholders_equity') AS stockholders_equity,
        MAX(value) FILTER (WHERE metric = 'cash')                AS cash
    FROM financials
    GROUP BY ticker, fiscal_year
),
filled AS (
    -- Some companies (e.g. Alphabet, Meta) don't report gross profit, but it's
    -- simply revenue minus cost of revenue.
    SELECT * EXCLUDE (gross_profit_reported),
           COALESCE(gross_profit_reported, revenue - cost_of_revenue) AS gross_profit
    FROM pivoted
)
SELECT
    *,
    gross_profit        / NULLIF(revenue, 0) AS gross_margin,
    operating_income    / NULLIF(revenue, 0) AS operating_margin,
    net_income          / NULLIF(revenue, 0) AS net_margin,
    rnd_expense         / NULLIF(revenue, 0) AS rnd_to_revenue,
    operating_cash_flow - capex              AS free_cash_flow,
    -- Year-over-year growth, only when the previous year actually exists.
    CASE WHEN LAG(fiscal_year) OVER w = fiscal_year - 1
         THEN revenue / NULLIF(LAG(revenue) OVER w, 0) - 1 END AS revenue_growth
FROM filled
WINDOW w AS (PARTITION BY ticker ORDER BY fiscal_year)
"""


def connect(path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(SCHEMA)
    return con


def replace_rows(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> None:
    """Delete this table's rows for the tickers in `df`, then insert `df`.

    That makes the pipeline safe to re-run: data gets updated, never duplicated.
    Companies that failed to download this time keep their previous rows.
    """
    if df.empty:
        return
    tickers = df["ticker"].unique().tolist()
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(f"DELETE FROM {table} WHERE list_contains(?, ticker)", [tickers])
        con.register("incoming", df)
        # BY NAME matches DataFrame columns to table columns by name, not position.
        con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM incoming")
        con.unregister("incoming")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def create_views(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(FINANCIALS_WIDE_VIEW)


def print_summary(con: duckdb.DuckDBPyConnection) -> None:
    print("\nRows per table:")
    for table in TABLES:
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:<11} {count:>8,}")

    latest = con.execute("""
        SELECT ticker, fiscal_year,
               ROUND(revenue / 1e9, 1)        AS revenue_bn,
               ROUND(gross_margin * 100, 1)   AS gross_margin_pct,
               ROUND(net_margin * 100, 1)     AS net_margin_pct,
               ROUND(revenue_growth * 100, 1) AS revenue_growth_pct
        FROM financials_wide
        QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY fiscal_year DESC) = 1
        ORDER BY revenue DESC NULLS LAST
    """).df()
    if not latest.empty:
        print("\nLatest fiscal year per company:")
        print(latest.to_string(index=False))
