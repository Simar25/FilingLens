"""Build the FilingLens dataset.   Run with:  python -m filinglens.pipeline

  1. Look up each company's SEC ID (CIK)
  2. Download its financial numbers       -> 'financials' table
  3. Find and download its 10-K reports   -> 'filings' table + files in data/raw/filings/
  4. Download daily stock prices          -> 'prices' table
  5. Save everything in DuckDB            -> data/processed/filinglens.duckdb

Re-running is safe: downloads are cached and database rows are replaced, not duplicated.
"""

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from filinglens import database
from filinglens.config import PROJECT_ROOT, RAW_DIR, Settings, load_settings
from filinglens.ingest.companies import resolve_companies
from filinglens.ingest.filings import assign_fiscal_years, document_url, list_10k_filings
from filinglens.ingest.financials import extract_annual_financials, fetch_company_facts
from filinglens.ingest.prices import download_prices
from filinglens.ingest.sec_client import SecClient

log = logging.getLogger("filinglens")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SEC filings, financials and prices into DuckDB.")
    parser.add_argument("--tickers", nargs="+", help="Use these tickers instead of config/companies.yaml")
    parser.add_argument("--skip-sec", action="store_true", help="Skip SEC downloads; only update prices")
    parser.add_argument("--skip-prices", action="store_true", help="Skip stock prices")
    parser.add_argument("--skip-documents", action="store_true",
                        help="List 10-K reports but don't download the report files")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-download SEC filing lists and financial data (use after new reports come out)")
    return parser.parse_args(argv)


def ingest_sec(settings: Settings, tickers: list[str], refresh: bool, download_documents: bool):
    client = SecClient(settings.sec_user_agent, RAW_DIR / "sec", refresh=refresh)
    years = range(settings.start_fiscal_year, settings.end_fiscal_year + 1)

    companies = resolve_companies(client, tickers)
    financials, filings = [], []
    for company in companies.itertuples():
        try:
            facts = fetch_company_facts(client, company.cik)
            numbers, year_by_accession = extract_annual_financials(facts, company.ticker)
            numbers = numbers[numbers["fiscal_year"].isin(years)]

            reports = list_10k_filings(client, company.cik, filed_since=date(years.start, 1, 1))
            reports = assign_fiscal_years(reports, year_by_accession)
            reports = reports[reports["fiscal_year"].isin(years)]
            reports = download_reports(client, company, reports, download_documents)
        except requests.RequestException as exc:
            log.error("%s: download failed (%s), skipping this company", company.ticker, exc)
            continue
        log.info("%-5s %-28s %3d financial values, %d annual reports",
                 company.ticker, company.name[:28], len(numbers), len(reports))
        financials.append(numbers)
        filings.append(reports)

    return (
        companies,
        pd.concat(financials, ignore_index=True) if financials else pd.DataFrame(),
        pd.concat(filings, ignore_index=True) if filings else pd.DataFrame(),
    )


def download_reports(client: SecClient, company, reports: pd.DataFrame, download: bool) -> pd.DataFrame:
    urls, paths = [], []
    for report in reports.itertuples():
        url = document_url(company.cik, report.accession_number, report.primary_document)
        suffix = Path(report.primary_document).suffix
        dest = RAW_DIR / "filings" / company.ticker / f"FY{report.fiscal_year}_{report.accession_number}{suffix}"
        if download:
            client.download_file(url, dest)
        urls.append(url)
        paths.append(dest.relative_to(PROJECT_ROOT).as_posix() if dest.exists() else None)
    return reports.assign(ticker=company.ticker, document_url=urls, local_path=paths).drop(
        columns="primary_document"
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    settings = load_settings()
    tickers = [t.upper() for t in args.tickers] if args.tickers else settings.tickers

    con = database.connect()
    try:
        if not args.skip_sec:
            log.info("Downloading SEC data for %d companies", len(tickers))
            companies, financials, filings = ingest_sec(settings, tickers, args.refresh, not args.skip_documents)
            database.replace_rows(con, "companies", companies)
            database.replace_rows(con, "financials", financials)
            database.replace_rows(con, "filings", filings)

        if not args.skip_prices:
            price_tickers = list(dict.fromkeys(tickers + [settings.benchmark]))  # no duplicates
            log.info("Downloading daily prices for %d tickers", len(price_tickers))
            prices = download_prices(price_tickers, start=date(settings.start_fiscal_year - 1, 1, 1))
            database.replace_rows(con, "prices", prices)

        database.create_views(con)
        database.print_summary(con)
    finally:
        con.close()


if __name__ == "__main__":
    main()
