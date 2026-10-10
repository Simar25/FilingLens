"""Step 3 - find each company's 10-K annual reports.

SEC's "submissions" API lists every document a company has filed. Each filing
has an accession number, a unique ID like 0000320193-23-000106, and its main
document is stored at:

    https://www.sec.gov/Archives/edgar/data/<CIK>/<accession number without dashes>/<file name>
"""

from datetime import date

import pandas as pd

from filinglens.ingest.companies import cik10
from filinglens.ingest.sec_client import SecClient

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
DOCUMENT_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{file_name}"

# SEC's field name -> our column name
FIELDS = {
    "accessionNumber": "accession_number",
    "form": "form",
    "filingDate": "filing_date",
    "reportDate": "report_date",
    "primaryDocument": "primary_document",
}


def _rows(block: dict) -> pd.DataFrame:
    # SEC stores the list column by column: {"form": ["10-K", "8-K", ...], "filingDate": [...], ...}
    return pd.DataFrame({ours: block.get(theirs, []) for theirs, ours in FIELDS.items()})


def list_10k_filings(client: SecClient, cik: int, filed_since: date) -> pd.DataFrame:
    index = client.get_json(SUBMISSIONS_URL.format(cik=cik10(cik)), f"submissions/CIK{cik10(cik)}.json")
    pages = [_rows(index["filings"]["recent"])]

    # "recent" only holds the latest ~1,000 filings. Companies that file a lot
    # (big banks file thousands of documents a year) need the older pages too.
    for page in index["filings"].get("files", []):
        if page["filingTo"] >= filed_since.isoformat():
            url = SUBMISSIONS_PAGE_URL.format(name=page["name"])
            pages.append(_rows(client.get_json(url, f"submissions/{page['name']}")))

    filings = pd.concat(pages, ignore_index=True)
    filings["filing_date"] = pd.to_datetime(filings["filing_date"])
    filings["report_date"] = pd.to_datetime(filings["report_date"], errors="coerce")
    # Original 10-Ks only: amendments (10-K/A) usually just patch a few sections.
    filings = filings[(filings["form"] == "10-K") & (filings["filing_date"] >= pd.Timestamp(filed_since))]
    return filings.drop_duplicates("accession_number").reset_index(drop=True)


def assign_fiscal_years(filings: pd.DataFrame, year_by_accession: dict[str, int]) -> pd.DataFrame:
    """Label each report with the fiscal year it covers.

    The label comes from the financial data (see financials.fiscal_year_labels), so
    reports and numbers always agree. If it's missing, fall back to the calendar
    year of the report's period end.
    """
    fiscal_year = filings["accession_number"].map(year_by_accession).fillna(filings["report_date"].dt.year)
    labelled = filings.assign(fiscal_year=fiscal_year).dropna(subset=["fiscal_year"])
    return labelled.assign(fiscal_year=labelled["fiscal_year"].astype(int))


def document_url(cik: int, accession_number: str, primary_document: str) -> str:
    return DOCUMENT_URL.format(cik=cik, accession=accession_number.replace("-", ""), file_name=primary_document)
