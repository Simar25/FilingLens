# Step 1: the data pipeline

**Goal:** build a clean, trustworthy dataset that every later step uses: the RAG system, text-to-SQL, and the research study.

**Command:** `python -m filinglens.pipeline`

**Result:** one database file, `data/processed/filinglens.duckdb`, plus the raw 10-K report files in `data/raw/filings/`.

---

## Key terms

| Term | Meaning |
|---|---|
| **SEC** | The US Securities and Exchange Commission. Every US public company must file reports with it. |
| **EDGAR** | SEC's public database of those filings. Free to use. |
| **10-K** | A company's annual report: business description, risk factors, management's discussion of results (MD&A), and audited financial statements. Usually 80–150 pages. |
| **CIK** | Central Index Key: SEC's permanent ID for a company (Apple = 320193). Tickers can change; CIKs never do. |
| **Accession number** | The unique ID of one filing, e.g. `0000320193-23-000106`. |
| **XBRL** | A standard tagging system. Every number in a financial statement is labelled with a tag such as `NetIncomeLoss`, so computers can read it. |
| **Fiscal year** | A company's accounting year. It often doesn't match the calendar: Apple's ends in September, NVIDIA's in January. |

---

## The five stages

### 1. Ticker to CIK (`ingest/companies.py`)

SEC's APIs don't accept tickers like `AAPL`, only CIKs. SEC publishes a file that maps every ticker to its CIK, so we download it once and look up our companies.

*Why not hard-code the CIKs?* The lookup file stays correct when companies rename or list new share classes, and a typo in a ticker fails loudly instead of silently downloading the wrong company.

### 2. Financial numbers (`ingest/financials.py`)

SEC's **companyfacts** API returns every XBRL-tagged number a company has ever reported. It's a big nested JSON file (around 5–15 MB per company). We turn it into a tidy table with one row per (company, fiscal year, metric).

This is the hardest part of the step, because the raw data has three traps:

**Trap 1: same idea, different tags.** "Revenue" can be tagged `Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax`, `SalesRevenueNet`, and so on. Companies pick different tags, and a single company may switch tags after an accounting-rule change (in 2018 most did).
**Fix:** `METRICS` maps each friendly name (`revenue`) to a list of tags in order of preference. For each year we use the first tag that has a value.

**Trap 2: the `fy` field is misleading.** A 10-K shows the current year *and* the two previous years side by side for comparison. Every number in that filing is labelled with the filing's year. So Apple's 2023 report contains 2021's revenue labelled `fy: 2023`. Trusting `fy` would put three different revenue figures in "2023".
**Fix:** we identify a number's year by its **period end date**. For each report, the latest end date it contains is the year it covers.

That leaves naming the year, which companies do differently: NVIDIA calls the year ending January 2026 "fiscal 2026", but Target calls the year ending February 2026 "fiscal 2025". Each report's `fy` shows the company's convention, **but `fy` is occasionally just wrong**. When we first ran the pipeline, Salesforce's report for the year ending January 2026 was tagged `fy=2025`, which collided with the real 2025 report. So we don't trust any single report. We learn each company's convention by **majority vote** across all its reports (`fiscal year = calendar year of period end + most common offset`), and one bad label gets outvoted. That bug, and the test that now guards against it, makes a good interview story about data quality.

**Trap 3: restatements.** Companies sometimes correct past numbers, so 2022 revenue might appear in the 2022, 2023 and 2024 reports with slightly different values.
**Fix:** we keep the most recently filed value, the company's latest word on it.

We also keep only full-year values (350–380 days, because some fiscal years have 53 weeks) and drop quarterly ones.

These rules are checked by the tests in `tests/test_financials.py`. Each test builds a tiny fake SEC response that contains one trap and checks we handle it.

### 3. The 10-K reports (`ingest/filings.py` and `pipeline.py`)

SEC's **submissions** API lists everything a company has filed: 8-Ks, 10-Qs, proxy statements and more. We keep only original 10-Ks in our year range. Amendments (10-K/A) usually just patch a few pages, so we skip them.

One detail: the list only holds the latest ~1,000 filings. That covers years of history for most tech companies, but big banks file thousands of documents a year, so the code also fetches older pages when they overlap our date range.

Each report is labelled with its fiscal year using the *same* mapping from stage 2, so the text and the numbers can never disagree about which year is which. The main document (HTML) is saved as `data/raw/filings/<TICKER>/FY<year>_<accession>.htm`. Step 2 of the project will split these files into sections (Risk Factors, MD&A…) for RAG.

### 4. Stock prices (`ingest/prices.py`)

Daily prices from Yahoo Finance for every company plus **SPY**, an S&P 500 index fund that stands in for "the market". The research step will compare each stock's return with SPY's.

We use **adjusted** prices, which are corrected for stock splits and dividends. NVIDIA split each share into 10 in June 2024, so the raw price fell from about $1,200 to $120 overnight. Adjusted prices are rescaled so returns stay correct, and in our data NVIDIA's price is about $120 on both sides of the split date.

### 5. Save to DuckDB (`database.py`)

**DuckDB** is a database in a single file (like SQLite) but designed for analysis. It runs SQL very fast on millions of rows and reads pandas DataFrames directly. No server to install.

We create four tables (`companies`, `filings`, `financials`, `prices`) plus a **view** called `financials_wide`. A view is a saved query that behaves like a table. It turns the tall `financials` table (one row per metric) into one row per company-year with columns like `revenue`, `net_income`, and calculated ratios: gross margin, net margin, R&D as a share of revenue, free cash flow and revenue growth.

---

## Engineering practices worth mentioning in interviews

| Practice | Where | Why it matters |
|---|---|---|
| **Rate limiting** | `SecClient._get` | SEC blocks IPs that send more than 10 requests per second. We wait 0.15 s between requests. |
| **Retries with backoff** | `SecClient.__init__` | Networks fail. Temporary errors (429, 5xx) are retried, waiting longer each time. |
| **Caching** | `SecClient.get_json`, `download_file` | Everything downloaded is saved in `data/raw/`. Re-runs take seconds, and you can work offline. `--refresh` re-downloads the JSON indexes when new reports come out. |
| **Atomic writes** | `_write_atomically` | Files are written to `.part` and then renamed. Pressing Ctrl+C mid-download never leaves a corrupt file that looks complete. |
| **Idempotent loading** | `database.replace_rows` | Running the pipeline twice gives the same database, not duplicates. Old rows for each company are deleted and new ones inserted in one transaction. |
| **Data lineage** | `xbrl_tag`, `accession_number` columns | Every number can be traced to the exact tag and filing it came from. That's essential for an AI system that must cite its sources. |
| **Config separate from code** | `config/companies.yaml` | Change companies or years without touching code. |
| **Secrets kept out of git** | `.env`, `.gitignore` | Your email (and, later, API keys) never get uploaded to GitHub. |
| **Tests with fake data** | `tests/` | Tests run offline in seconds and pin down the tricky rules above. |

---

## Explore the data yourself

**The easy way:** ask one question at a time from the terminal (the `PS E:\FilingLens>` prompt). Put the SQL in double quotes, and text values inside it in single quotes:

```
.\.venv\Scripts\python.exe -m filinglens.query "SELECT ticker, fiscal_year, ROUND(gross_margin * 100, 1) AS gross_margin_pct FROM financials_wide WHERE ticker = 'NVDA' ORDER BY fiscal_year"
```

**In Python:** start Python with `.\.venv\Scripts\python.exe`. The prompt changes to `>>>`. Type **one line at a time** and wait for `>>>` again before typing the next. A `...` prompt means Python thinks your line isn't finished: press **Ctrl + C** to cancel. Type `exit()` to leave, and always leave before re-running the pipeline.

```python
import duckdb
con = duckdb.connect("data/processed/filinglens.duckdb", read_only=True)

# Gross margin by company and year, as a percentage
con.sql("""
    SELECT ticker, fiscal_year, ROUND(gross_margin * 100, 1) AS gross_margin_pct
    FROM financials_wide
    WHERE ticker IN ('NVDA', 'AMD', 'INTC')
    ORDER BY ticker, fiscal_year
""").df()

# Where did a number come from?
con.sql("""
    SELECT f.ticker, f.fiscal_year, f.value / 1e9 AS revenue_bn, f.xbrl_tag, d.document_url
    FROM financials f
    JOIN filings d ON d.ticker = f.ticker AND d.fiscal_year = f.fiscal_year
    WHERE f.metric = 'revenue' AND f.ticker = 'AAPL'
""").df()
```

## Exercises (to learn the code)

1. **Add a metric.** Add `"dividends_paid": Metric(("PaymentsOfDividends",), "USD", "duration")` to `METRICS`, re-run, and query it.
2. **Add companies.** Put `TSLA` and `JPM` in `config/companies.yaml`. Banks don't report gross profit or R&D: what does `financials_wide` show for JPM?
3. **Check a number.** Pick one company and year, open its 10-K (the `document_url`) and find the income statement. Does our revenue match?
4. **Write a query.** Which company grew revenue fastest in its latest fiscal year?
