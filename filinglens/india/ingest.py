"""Step 6b: the Indian annual reports in the app, for questions about their text.

Run after the pilot:  python -m filinglens.india.ingest
Then:                 python -m filinglens.build_index --tickers INFY WIPRO HDFCBANK ICICIBANK DRREDDY RELIANCE TCS HINDUNILVR LT BHARTIARTL
                      python -m filinglens.online

The pilot found that the MD&A can't be located reliably in these PDFs. Search doesn't need it:
every page goes in, and an answer cites the page it came from. So nothing here depends on the
pilot's weakest step.

The companies get no rows in the financials table. Their figures are in rupees, and the table and
every prompt assume US dollars, so a ranking would compare ₹ crore with $ billions. They live in
the reports instead, in the report's own currency (form 'AR', see Assistant.reports_only).
"""

import re
import unicodedata
from datetime import date, datetime

import pandas as pd
import pymupdf

from filinglens import database
from filinglens.config import PROJECT_ROOT
from filinglens.india.nse import NseClient, choose_reports
from filinglens.india.pilot import OUT_DIR, PILOT
from filinglens.india.report_text import clean, remove_page_furniture
from filinglens.text.chunking import chunk_text

MIN_PAGE_WORDS = 40  # below this a page is a cover or a photo, not something to search
SECTOR = "India"     # the app's market map shows them as one more column
FILE_DATE = re.compile(r"_(\d{8})\d{6}")  # AR_24850_RELIANCE_2023_2024_07082024143222.pdf: 7 August 2024
# PDFs draw the rupee sign and bullets with their own fonts, so the extracted text holds other characters.
RUPEE = re.compile(r"[`¢](?=\s?\d)|\bE(?=[\d,.]+\s?(?:Mn|Bn)\b)")  # `35,234 crore (Infosys), ¢ 2 lakh (L&T), E847 Mn (Airtel)
BULLET = re.compile(r"\x83|\tà|[\uf0d8\uf0fc\uf0a7\uf0b7]")


def fix_characters(text: str) -> str:
    """NFKC turns ligatures and the many kinds of space into plain letters and spaces, for search and for reading."""
    return BULLET.sub("•", RUPEE.sub("₹", unicodedata.normalize("NFKC", text)))


def page_chunks(pages: list[str], report: str, ticker: str, year: int) -> list[dict]:
    """Numbered chunks of a report's pages, each titled with the page it's on, so a citation can be checked."""
    chunks = []
    for number, page in enumerate(remove_page_furniture(pages), start=1):
        if len(page.split()) < MIN_PAGE_WORDS:
            continue
        for text in chunk_text(fix_characters(clean(page))):
            chunks.append({"chunk_id": f"{ticker}-{year}-NSE-{len(chunks):05d}", "accession_number": report,
                           "ticker": ticker, "fiscal_year": year, "item": "NSE", "title": f"page {number}",
                           "chunk_index": len(chunks), "text": text, "n_words": len(text.split())})
    return chunks


def filing_date(url: str, year: int) -> date:
    """NSE's list has no date for many older reports, but the file name always does."""
    found = FILE_DATE.findall(url)
    return datetime.strptime(found[0], "%d%m%Y").date() if found else date(year, 3, 31)


def main() -> None:
    reports = pd.read_csv(OUT_DIR / "reports.csv")
    client = NseClient()
    urls = {symbol: choose_reports(client.annual_report_list(symbol)).set_index("fiscal_year")["url"] for symbol in PILOT}
    filings, chunks = [], []
    for report in reports.itertuples():
        accession = f"NSE-{report.symbol}-{report.fiscal_year}"
        url = urls[report.symbol][report.fiscal_year]
        with pymupdf.open(PROJECT_ROOT / report.pdf_path) as doc:
            pages = [page.get_text() for page in doc]
        found = page_chunks(pages, accession, report.symbol, report.fiscal_year)
        print(f"{report.symbol:<11} FY{report.fiscal_year}  {len(pages):>4} pages  {len(found):>5} passages", flush=True)
        chunks += found
        filings.append({"accession_number": accession, "ticker": report.symbol, "fiscal_year": report.fiscal_year,
                        "form": "AR", "filing_date": filing_date(url, report.fiscal_year),
                        "report_date": date(report.fiscal_year, 3, 31), "document_url": url,
                        "local_path": report.pdf_path})

    con = database.connect()
    try:
        tickers = list(PILOT)
        database.replace_rows(con, "companies", pd.DataFrame({"ticker": tickers, "cik": 0, "name": list(PILOT.values())}))
        database.replace_rows(con, "company_info", pd.DataFrame({"ticker": tickers, "sector": SECTOR, "country": SECTOR,
                                                                 "as_of": date.today()}))
        database.replace_rows(con, "filings", pd.DataFrame(filings))
        database.replace_rows(con, "chunks", pd.DataFrame(chunks))
        database.create_views(con)
    finally:
        con.close()
    print(f"\n{len(filings)} reports, {len(chunks):,} passages")


if __name__ == "__main__":
    main()
