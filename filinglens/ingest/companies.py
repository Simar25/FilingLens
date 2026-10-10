"""Step 1 - turn stock tickers into SEC company IDs.

SEC doesn't organise its data by ticker, because tickers change (Facebook's FB
became META in 2022). Every company instead has a permanent Central Index Key
(CIK); Apple's is 320193. SEC's URLs use the CIK padded to 10 digits:
.../CIK0000320193.json
"""

import pandas as pd

from filinglens.ingest.sec_client import SecClient

TICKER_LIST_URL = "https://www.sec.gov/files/company_tickers.json"


def cik10(cik: int) -> str:
    """320193 -> '0000320193', the format SEC's URLs expect."""
    return f"{cik:010d}"


def resolve_companies(client: SecClient, tickers: list[str]) -> pd.DataFrame:
    """Return a table with columns ticker, cik, name."""
    # The file looks like {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    raw = client.get_json(TICKER_LIST_URL, "company_tickers.json")
    by_ticker = {entry["ticker"].upper(): entry for entry in raw.values()}

    unknown = [t for t in tickers if t not in by_ticker]
    if unknown:
        raise ValueError(f"These tickers aren't in SEC's company list: {unknown}")

    return pd.DataFrame(
        [{"ticker": t, "cik": int(by_ticker[t]["cik_str"]), "name": by_ticker[t]["title"]} for t in tickers]
    )
