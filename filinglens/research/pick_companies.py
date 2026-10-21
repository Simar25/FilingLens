"""Step 4, before the study: choose which companies to add, by a rule you can state.

Run with:  python -m filinglens.research.pick_companies

14 companies x 5 years is too few data points for statistics, and they are
mostly large tech companies. This script suggests more, from every industry:

  1. Yahoo Finance's stock screener lists US-listed companies worth $10B+ in
     each of its 11 sectors, largest first.
  2. Keep only companies that file 10-K reports with SEC, with one for every
     fiscal year in config/companies.yaml. Foreign companies (TSMC, ASML,
     Shell) file a different form (20-F), and recent listings (GE Vernova,
     2024) don't have six years of reports.
  3. Fill each sector up to PER_SECTOR companies, counting the ones already
     in the config.

It prints the list for you to paste into config/companies.yaml. The result
depends on the day you run it (market values change), so the config keeps
the list, not this script.

Known bias: picking today's largest companies favours the ones that did well
in the past years (survivorship bias). See docs/04_research_findings.md.
"""

import argparse
import logging
import time
from datetime import date

import pandas as pd
import yfinance as yf
from yfinance import EquityQuery

from filinglens.config import RAW_DIR, load_settings
from filinglens.ingest.companies import TICKER_LIST_URL
from filinglens.ingest.filings import list_10k_filings
from filinglens.ingest.sec_client import SecClient

log = logging.getLogger("filinglens")

SECTORS = ["Technology", "Communication Services", "Consumer Cyclical", "Consumer Defensive", "Healthcare",
           "Financial Services", "Industrials", "Energy", "Utilities", "Real Estate", "Basic Materials"]
PER_SECTOR = 9
MIN_MARKET_CAP = 10e9
US_EXCHANGES = ["NMS", "NGM", "NCM", "NYQ"]  # Nasdaq's three tiers and NYSE


def screen_sector(sector: str, size: int = 60) -> pd.DataFrame:
    """The largest US-listed companies in one Yahoo sector."""
    query = EquityQuery("and", [
        EquityQuery("eq", ["region", "us"]),
        EquityQuery("eq", ["sector", sector]),
        EquityQuery("gte", ["intradaymarketcap", MIN_MARKET_CAP]),
        EquityQuery("is-in", ["exchange", *US_EXCHANGES]),
    ])
    quotes = yf.screen(query, size=size, sortField="intradaymarketcap", sortAsc=False).get("quotes", [])
    return pd.DataFrame([{
        "sector": sector,
        "ticker": q["symbol"].upper(),
        "name": q.get("longName") or q.get("shortName", ""),
        "market_cap": q.get("marketCap", 0),
        "volume": q.get("averageDailyVolume3Month", 0),
    } for q in quotes])


def has_every_report(client: SecClient, cik: int, first_year: int, last_year: int) -> tuple[bool, str]:
    """Does this company file 10-Ks, one for each fiscal year in the range?

    A quick check on filing dates (the pipeline later labels each report with
    its exact fiscal year): a 10-K is filed within ~4 months after the fiscal
    year ends, so years first_year..last_year need at least that many 10-Ks
    filed from January of first_year onwards, the latest one within the past
    14 months.
    """
    reports = list_10k_filings(client, cik, filed_since=date(first_year, 1, 1))
    needed = last_year - first_year + 1
    if len(reports) < needed:
        return False, f"only {len(reports)} 10-Ks since {first_year}"
    latest = reports["filing_date"].max()
    if latest < pd.Timestamp.today() - pd.DateOffset(months=14):
        return False, f"no 10-K since {latest.date()}"
    return True, "ok"


def pick(settings, per_sector: int = PER_SECTOR) -> pd.DataFrame:
    client = SecClient(settings.sec_user_agent, RAW_DIR / "sec")
    sec_list = client.get_json(TICKER_LIST_URL, "company_tickers.json")
    cik_by_ticker = {e["ticker"].upper(): int(e["cik_str"]) for e in sec_list.values()}
    existing = set(settings.tickers)

    rows = []
    for sector in SECTORS:
        candidates = screen_sector(sector)
        time.sleep(1)  # be gentle with Yahoo's unofficial API
        candidates["cik"] = candidates["ticker"].map(cik_by_ticker)
        # Two share classes of one company (GOOG/GOOGL, BRK-A/BRK-B) share a CIK: keep the more traded one.
        candidates = (candidates.sort_values("volume", ascending=False).drop_duplicates("cik")
                      .sort_values("market_cap", ascending=False))
        n_existing = candidates["ticker"].isin(existing).sum()
        n_new = 0
        for c in candidates.itertuples():
            if c.ticker in existing:
                status = "already in config"
            elif pd.isna(c.cik):
                status = "not an SEC registrant"
            elif n_existing + n_new >= per_sector:
                break
            else:
                ok, status = has_every_report(client, int(c.cik), settings.start_fiscal_year,
                                              settings.end_fiscal_year)
                if ok:
                    status, n_new = "added", n_new + 1
            rows.append({"sector": sector, "ticker": c.ticker, "name": c.name,
                         "market_cap_bn": round(c.market_cap / 1e9), "status": status})
        log.info("%-22s %d already in config, %d added", sector, n_existing, n_new)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Suggest companies to add, largest first in every sector.")
    parser.add_argument("--per-sector", type=int, default=PER_SECTOR,
                        help=f"Companies per sector, counting the ones in the config (default {PER_SECTOR})")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    settings = load_settings()

    result = pick(settings, args.per_sector)
    pd.set_option("display.width", 160)
    print(result.to_string(index=False))

    added = result[result["status"] == "added"]
    print(f"\n{len(added)} companies to add (screened {date.today()}). Paste into config/companies.yaml:\n")
    for sector, group in added.groupby("sector", sort=False):
        print(f"  # {sector}")
        for c in group.itertuples():
            print(f"  - {c.ticker:<6} # {c.name}")


if __name__ == "__main__":
    main()
