# FilingLens

**An AI analyst for company annual reports.**

Ask questions in plain English, such as *"How did NVIDIA's gross margin change since 2021, and what did management say caused it?"*, and get an answer that combines the financial numbers with the explanation from the company's own annual report, with citations.

FilingLens combines three things:

- **RAG (retrieval-augmented generation)** over SEC 10-K annual reports
- **Text-to-SQL** over a DuckDB database of financial statements and stock prices
- **A research study**: does a change in the wording of a company's annual report predict its stock returns? This is a small-scale replication of *Lazy Prices* (Cohen, Malloy & Nguyen, 2020).

## Status

| Step | What | Status |
|---|---|---|
| 1 | Data pipeline: SEC filings, financials and stock prices into DuckDB | Done |
| 2 | Split reports into sections; RAG with citations | Next |
| 3 | Text-to-SQL, query router, charts | Planned |
| 4 | Research study: report language vs. stock returns | Planned |
| 5 | Evaluation set; hybrid search + reranker | Planned |
| 6 | Streamlit app, Docker, deployment | Planned |

## Architecture

```
                    +----------------+
  User question --> |  Query router  |  LLM decides: needs numbers, text, or both?
                    +---+--------+---+
          numbers       |        |      "why / what / risks"
                        v        v
          +----------------+  +----------------------+
          |  Text-to-SQL   |  |  RAG over 10-Ks      |
          |  (DuckDB:      |  |  BM25 + embeddings   |
          |  financials,   |  |  + reranker,         |
          |  prices)       |  |  filter by ticker/yr |
          +-------+--------+  +----------+-----------+
                  +----------+-----------+
                             v
                Answer + chart + citations
```

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

copy .env.example .env            # macOS/Linux: cp .env.example .env
# then edit .env and put your name and email in SEC_USER_AGENT (SEC requires it)

python -m filinglens.pipeline     # downloads everything, about 5 minutes the first time
pytest                            # runs the tests
```

Useful options: `--tickers AAPL NVDA` (just these companies), `--skip-documents` (numbers only, no report files), `--refresh` (pick up newly published reports).

Companies and years are set in [`config/companies.yaml`](config/companies.yaml).

## Data sources

| Data | Source | Cost |
|---|---|---|
| 10-K annual reports | SEC EDGAR submissions API | Free |
| Financial statements | SEC XBRL "companyfacts" API | Free |
| Daily stock prices (split/dividend adjusted) | Yahoo Finance via `yfinance` | Free |

## Database

Everything lives in one DuckDB file, `data/processed/filinglens.duckdb`:

| Table | One row per | Main columns |
|---|---|---|
| `companies` | company | ticker, cik, name |
| `filings` | 10-K report | ticker, fiscal_year, filing_date, document_url, local_path |
| `financials` | company, year, metric | metric, value, unit, xbrl_tag, accession_number |
| `prices` | company, trading day | close (adjusted), volume |
| `financials_wide` (view) | company, year | revenue, net_income, ..., gross_margin, net_margin, revenue_growth |

Every financial value records which XBRL tag and which filing it came from, so any number can be traced back to its source.

## Project structure

```
FilingLens/
├── config/companies.yaml      which companies and years to download
├── filinglens/
│   ├── config.py              settings and folder paths
│   ├── pipeline.py            runs the whole data pipeline
│   ├── database.py            DuckDB tables, loading, financials_wide view
│   └── ingest/
│       ├── sec_client.py      polite SEC downloader: rate limit, retries, cache
│       ├── companies.py       ticker -> SEC company ID (CIK)
│       ├── financials.py      XBRL facts -> tidy yearly financials
│       ├── filings.py         find 10-K reports
│       └── prices.py          daily stock prices
├── tests/                     offline tests with SEC-shaped sample data
└── docs/                      step-by-step explanation of each part
```

## Docs

- [Complete step-by-step guide](docs/00_project_guide.md), the full plan from setup to deployment
- [Step 1: data pipeline](docs/01_data_pipeline.md), with how it works and why it's built this way
