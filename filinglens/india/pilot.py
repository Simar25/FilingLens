"""India pilot: can the MD&A be taken out of Indian annual reports reliably?

Run with:  python -m filinglens.india.pilot

For 10 large NSE companies and fiscal years 2023-2026 (April-March years,
labelled by the year they end), it downloads each annual report from NSE, finds
the MD&A, saves its text, and measures how much it changed from the year
before with the US study's word measures (research/text_change.py).

The companies: the five that are listed in India AND file a 20-F annual report
with the US SEC (Infosys, Wipro, HDFC Bank, ICICI Bank, Dr. Reddy's), so a later
step can compare what the same company tells both regulators, plus five large
companies from other sectors.

Outputs in data/processed/india/: reports.csv (one row per report, with what
was found), changes.csv (year-on-year change), text/<SYMBOL>/FY<year>_mdna.txt.
"""

import argparse
import logging

import pandas as pd
import requests

from filinglens.config import PROJECT_ROOT
from filinglens.india.nse import NseClient, choose_reports
from filinglens.india.report_text import find_mdna
from filinglens.research.text_change import cosine_similarity, jaccard_similarity, name_words, words

log = logging.getLogger("filinglens")

PILOT = {
    "INFY": "Infosys", "WIPRO": "Wipro", "HDFCBANK": "HDFC Bank", "ICICIBANK": "ICICI Bank",
    "DRREDDY": "Dr. Reddy's Laboratories",                                  # also file a 20-F with SEC
    "RELIANCE": "Reliance Industries", "TCS": "Tata Consultancy Services", "HINDUNILVR": "Hindustan Unilever",
    "LT": "Larsen & Toubro", "BHARTIARTL": "Bharti Airtel",
}
YEARS = range(2023, 2027)
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "india"
RISK_PHRASES = ("risks and concerns", "risk management", "threats")  # where Indian MD&As discuss risk


def process_company(client: NseClient, symbol: str) -> list[dict]:
    reports = choose_reports(client.annual_report_list(symbol))
    rows = []
    for report in reports[reports["fiscal_year"].isin(YEARS)].itertuples():
        base = {"symbol": symbol, "fiscal_year": report.fiscal_year, "company": report.company,
                "published_at": report.published_at, "file_type": report.url.rsplit(".", 1)[-1].lower()}
        try:
            pdfs = client.download(report.url, symbol)
        except (OSError, requests.RequestException) as exc:  # one bad file mustn't skip the company's other years
            log.error("%-10s FY%d  download failed: %s", symbol, report.fiscal_year, exc)
            rows.append({**base, "method": "download failed", "mdna_words": 0, "error": str(exc)[:200]})
            continue
        found = find_mdna(pdfs)
        row = {**base,
               "pdfs": len(pdfs), "pages": found.page_count, "chars_per_page": found.chars_per_page,
               "bookmarks": found.has_bookmarks, "method": found.mdna.method if found.mdna else "not found",
               "mdna_title": found.mdna.title if found.mdna else None,
               "mdna_pages": f"{found.mdna.first_page}-{found.mdna.last_page}" if found.mdna else None,
               "mdna_page_count": found.mdna.pages if found.mdna else 0,
               "mdna_words": len(found.mdna_text.split()),
               "risk_discussion": any(p in found.mdna_text.lower() for p in RISK_PHRASES),
               "pdf_path": found.pdf.relative_to(PROJECT_ROOT).as_posix()}
        if found.mdna_text:
            path = OUT_DIR / "text" / symbol / f"FY{report.fiscal_year}_mdna.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(found.mdna_text, encoding="utf-8")
        log.info("%-10s FY%d  %-9s pages %-9s %6d words", symbol, report.fiscal_year, row["method"],
                 row["mdna_pages"] or "-", row["mdna_words"])
        rows.append(row)
    return rows


def year_on_year_changes(reports: pd.DataFrame) -> pd.DataFrame:
    """Change of each MD&A vs the same company's previous year, ignoring the company's own name."""
    rows = []
    for symbol, group in reports[reports["mdna_words"] >= 500].groupby("symbol"):
        texts = {}
        for r in group.itertuples():
            ignore = name_words([r.company, PILOT[symbol], symbol])
            texts[r.fiscal_year] = words((OUT_DIR / "text" / symbol / f"FY{r.fiscal_year}_mdna.txt")
                                         .read_text(encoding="utf-8"), ignore)
        for year in sorted(texts):
            if year - 1 in texts:
                rows.append({"symbol": symbol, "fiscal_year": year,
                             "change_cosine": 1 - cosine_similarity(texts[year], texts[year - 1]),
                             "change_jaccard": 1 - jaccard_similarity(texts[year], texts[year - 1])})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="India pilot: NSE annual reports -> MD&A -> change scores.")
    parser.add_argument("--symbols", nargs="+", help="Only these NSE symbols (default: the 10 pilot companies)")
    parser.add_argument("--refresh", action="store_true", help="Re-download NSE's report lists")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    client = NseClient(refresh=args.refresh)
    rows = []
    for symbol in args.symbols or PILOT:
        try:
            rows += process_company(client, symbol)
        except Exception as exc:  # one company's problem shouldn't stop the pilot; it shows in the table
            log.error("%s: failed (%s: %s)", symbol, type(exc).__name__, exc)
    reports = pd.DataFrame(rows)
    changes = year_on_year_changes(reports)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reports.to_csv(OUT_DIR / "reports.csv", index=False)
    changes.to_csv(OUT_DIR / "changes.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nWhat was found in each report:")
    print(reports[["symbol", "fiscal_year", "file_type", "pages", "chars_per_page", "bookmarks", "method",
                   "mdna_pages", "mdna_words", "risk_discussion", "published_at"]].to_string(index=False))
    print(f"\nMD&A found in {(reports['mdna_words'] >= 500).sum()} of {len(reports)} reports")
    if not changes.empty:
        print("\nYear-on-year change of the MD&A (1 - similarity; US Risk Factors median was ~0.01):")
        print(changes.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
