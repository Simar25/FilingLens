"""Step 4 addition - sector, industry and size labels for each company, from Yahoo Finance.

SEC data has no modern sector labels (only old 4-digit SIC industry codes), but
the research study needs them: to compare industries, and to check that a
result isn't just "tech stocks did well". Yahoo's labels (11 sectors such as
'Technology', ~150 industries such as 'Semiconductors') are free via yfinance.

These are labels, not reported numbers: SEC stays the source for every
financial figure. market_cap is today's value, so it must never be used as if
it were known in the past (that would be look-ahead bias).
"""

import logging
import time
from datetime import date

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

INFO_COLUMNS = ["ticker", "sector", "industry", "market_cap", "employees", "country", "as_of"]
# Yahoo's field name -> our column name
FIELDS = {"sector": "sector", "industry": "industry", "marketCap": "market_cap",
          "fullTimeEmployees": "employees", "country": "country"}


def download_company_info(tickers: list[str], pause: float = 0.3) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
        except Exception as exc:  # rate limits and timeouts: skip, keep the previous row in the database
            log.warning("%s: company info download failed (%s), skipping", ticker, exc)
            continue
        if not info.get("sector"):
            log.warning("%s: Yahoo has no sector for this ticker, skipping", ticker)
            continue
        rows.append({"ticker": ticker, **{ours: info.get(theirs) for theirs, ours in FIELDS.items()},
                     "as_of": pd.Timestamp(date.today())})
        time.sleep(pause)  # one request per company; be gentle with an unofficial API
    log.info("Company info for %d of %d companies", len(rows), len(tickers))
    return pd.DataFrame(rows, columns=INFO_COLUMNS)
