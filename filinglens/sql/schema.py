"""Step 3.1 - describe the database to the LLM.

To write correct SQL, the LLM needs more than table and column names. It has
to know what the columns MEAN:
  - units: is `revenue` in dollars or billions? Is `gross_margin` 0.45 or 45?
  - which table to use: `financials_wide` for almost everything
  - what the values look like: 'AAPL', not 'Apple'
  - the traps: fiscal years differ by company, NULL means "not reported"

Most of this is written by hand below, because the meaning isn't stored in
the database. The rest (which companies exist, when their fiscal years end,
which years and dates are available) is read from the database, so it stays
correct when you add companies in Step 4.

Only the tables in ALLOWED_TABLES are described, and safety.py refuses
queries that use any other table. The report text (`sections`, `chunks`) is
left out on purpose: questions about wording go to RAG instead.
"""

import duckdb

ALLOWED_TABLES = ("financials_wide", "financials", "companies", "filings", "prices")

TABLES = """\
Table financials_wide: one row per company and fiscal year. Use it for almost every question about financial numbers.
  ticker               VARCHAR   e.g. 'AAPL'
  fiscal_year          INTEGER   the company's own fiscal year label (see "Fiscal years" below)
  revenue              DOUBLE    USD, total revenue (net sales)
  cost_of_revenue      DOUBLE    USD
  gross_profit         DOUBLE    USD, revenue - cost_of_revenue when not reported directly
  rnd_expense          DOUBLE    USD, research and development expense
  operating_income     DOUBLE    USD, operating income (negative = operating loss)
  net_income           DOUBLE    USD (negative = net loss)
  eps_diluted          DOUBLE    USD per share, diluted earnings per share
  operating_cash_flow  DOUBLE    USD, cash from operating activities
  capex                DOUBLE    USD, capital expenditure (purchases of property and equipment), a positive number
  free_cash_flow       DOUBLE    USD, operating_cash_flow - capex
  total_assets         DOUBLE    USD, on the last day of the fiscal year
  total_liabilities    DOUBLE    USD, on the last day of the fiscal year
  stockholders_equity  DOUBLE    USD, on the last day of the fiscal year
  cash                 DOUBLE    USD, cash and cash equivalents on the last day of the fiscal year
  gross_margin         DOUBLE    FRACTION of revenue (0.45 means 45%)
  operating_margin     DOUBLE    FRACTION of revenue
  net_margin           DOUBLE    FRACTION of revenue
  rnd_to_revenue       DOUBLE    FRACTION: rnd_expense / revenue
  revenue_growth       DOUBLE    FRACTION: revenue / previous fiscal year's revenue - 1 (NULL in the first year)

Table financials: the same reported numbers in long form, one row per company, fiscal year and metric. Use it only
for dividends_paid (not in financials_wide) or to show where a number came from.
  ticker, fiscal_year, metric VARCHAR, value DOUBLE, unit VARCHAR ('USD' or 'USD/shares'),
  period_start DATE, period_end DATE, xbrl_tag VARCHAR, accession_number VARCHAR, filed_date DATE
  metric is one of: {metrics}

Table companies: one row per company.
  ticker VARCHAR, cik INTEGER (SEC's company ID), name VARCHAR (official SEC name)

Table filings: one row per 10-K annual report.
  ticker, fiscal_year, form, filing_date DATE (the day it was made public), report_date DATE (last day of the fiscal
  year it covers), document_url VARCHAR

Table prices: one row per stock and trading day.
  ticker VARCHAR, date DATE, open DOUBLE, high DOUBLE, low DOUBLE, volume BIGINT,
  close DOUBLE (adjusted for stock splits and dividends: use it for returns and price changes)
  Also contains SPY, an S&P 500 index fund that stands for "the market". SPY is not in the companies table."""

NOTES = """\
- NULL means the company doesn't report that number (banks such as JPM have no cost_of_revenue, gross_profit or
  R&D). Never treat NULL as zero.
- Fiscal years are each company's own label and often don't match calendar years: NVIDIA's fiscal 2025 ended in
  January 2025, Apple's in September 2025. When a question says "2024", use fiscal_year = 2024, unless it clearly
  means the calendar year (for stock prices, it always means the calendar year).
- Financial numbers exist only per fiscal year. For a question about a CALENDAR year's financial numbers, use the
  fiscal year with most of its months in that calendar year: the one whose filings.report_date minus 182 days falls
  in it (JOIN filings USING (ticker, fiscal_year)). Return fiscal_year and report_date so the answer can say which
  fiscal year was used.
- Fiscal years after the last one in the data (see "Data available") haven't been reported yet. If a question asks
  about one, use the latest fiscal year that exists instead, and return fiscal_year so the answer can say which
  year was used.
- A stock's return over a period runs from the last close BEFORE the period starts to the last close in the period."""


def describe_schema(con: duckdb.DuckDBPyConnection) -> str:
    """The text the LLM reads before writing SQL: tables, columns, units, plus facts from the data."""
    metrics = [row[0] for row in con.execute("SELECT DISTINCT metric FROM financials ORDER BY 1").fetchall()]
    companies = con.execute("""
        SELECT c.ticker, c.name, MAX(f.fiscal_year) AS last_year, MAX(f.report_date) AS last_year_end
        FROM companies c LEFT JOIN filings f USING (ticker)
        GROUP BY ALL ORDER BY c.ticker
    """).fetchall()
    first_year, last_year = con.execute("SELECT MIN(fiscal_year), MAX(fiscal_year) FROM financials").fetchone()
    first_day, last_day = con.execute("SELECT MIN(date), MAX(date) FROM prices").fetchone()

    lines = [TABLES.format(metrics=", ".join(metrics)), "", "Companies (ticker: SEC name, when its fiscal year ends):"]
    for ticker, name, year, year_end in companies:
        ends = f"FY{year} ended {year_end}" if year_end else "no reports"
        lines.append(f"  {ticker}: {name} ({ends})")
    lines += [
        "",
        f"Data available: fiscal years {first_year} to {last_year}; daily prices from {first_day} to {last_day}.",
        "",
        "Notes:",
        NOTES,
    ]
    return "\n".join(lines)
