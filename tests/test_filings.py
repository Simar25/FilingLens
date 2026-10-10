from datetime import date

import pandas as pd

from filinglens.ingest.filings import assign_fiscal_years, document_url, list_10k_filings


class FakeClient:
    """Stands in for SecClient: returns canned JSON instead of calling SEC."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.requested: list[str] = []

    def get_json(self, url: str, cache_name: str) -> dict:
        self.requested.append(url)
        return self.responses[url]


def block(*filings):
    """Build SEC's column-by-column format from (accession, form, filed, report_date) tuples."""
    return {
        "accessionNumber": [f[0] for f in filings],
        "form": [f[1] for f in filings],
        "filingDate": [f[2] for f in filings],
        "reportDate": [f[3] for f in filings],
        "primaryDocument": [f"{f[0]}.htm" for f in filings],
    }


BASE = "https://data.sec.gov/submissions/"
RESPONSES = {
    BASE + "CIK0000000042.json": {"filings": {
        "recent": block(
            ("a-24", "10-K", "2024-10-30", "2024-09-28"),
            ("a-24-amend", "10-K/A", "2024-12-01", "2024-09-28"),
            ("a-8k", "8-K", "2024-11-01", ""),
        ),
        "files": [
            {"name": "page-001.json", "filingFrom": "2019-01-01", "filingTo": "2021-03-01"},
            {"name": "page-002.json", "filingFrom": "2015-01-01", "filingTo": "2018-12-31"},
        ],
    }},
    BASE + "page-001.json": block(
        ("a-20", "10-K", "2020-10-30", "2020-09-26"),
        ("a-19", "10-K", "2019-10-31", "2019-09-28"),
    ),
}


def test_lists_only_original_10ks_in_date_range():
    client = FakeClient(RESPONSES)
    filings = list_10k_filings(client, cik=42, filed_since=date(2020, 1, 1))
    assert sorted(filings["accession_number"]) == ["a-20", "a-24"]


def test_only_fetches_older_pages_that_overlap_date_range():
    client = FakeClient(RESPONSES)
    list_10k_filings(client, cik=42, filed_since=date(2020, 1, 1))
    assert BASE + "page-001.json" in client.requested
    assert BASE + "page-002.json" not in client.requested


def test_fiscal_year_from_financials_with_fallback_to_report_date():
    filings = pd.DataFrame({
        "accession_number": ["known", "unknown"],
        "report_date": pd.to_datetime(["2024-01-28", "2022-12-31"]),
    })
    labelled = assign_fiscal_years(filings, {"known": 2024})
    assert labelled["fiscal_year"].tolist() == [2024, 2022]


def test_document_url():
    url = document_url(320193, "0000320193-23-000106", "aapl-20230930.htm")
    assert url == "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
