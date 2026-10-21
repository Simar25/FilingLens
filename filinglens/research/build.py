"""Step 4: compute every measure the research study needs and save it in DuckDB.

Run with:  python -m filinglens.research.build
           python -m filinglens.research.build --only returns topics     (just these parts)

  4.1 text_change     how much Risk Factors and MD&A changed since last year   (GPU, a few minutes)
  4.2 tone            FinBERT tone of each MD&A                                (GPU, a few minutes)
  4.3 future_returns  abnormal stock returns after each report                 (seconds)
  4.5 topic_mentions  how often each report mentions AI, tariffs, ...          (about a minute)

It reads the 'sections', 'chunks', 'filings' and 'prices' tables, so run
python -m filinglens.pipeline  and  python -m filinglens.parse_reports  first.
The statistics and charts (4.4) are in notebooks/04_research_study.ipynb.
"""

import argparse
import json
import logging
import time

from filinglens import database
from filinglens.config import RAW_DIR, load_settings
from filinglens.ingest.companies import cik10
from filinglens.research.returns import future_returns
from filinglens.research.text_change import ITEMS, compute_text_change
from filinglens.research.topics import topic_mentions

log = logging.getLogger("filinglens")
PARTS = ("text_change", "tone", "returns", "topics")


def company_names(con, since_year: int) -> dict[str, list[str]]:
    """{ticker: [current SEC name, names it used since `since_year`]}, from SEC's cached filing index."""
    names = {}
    for ticker, cik, name in con.execute("SELECT ticker, cik, name FROM companies").fetchall():
        names[ticker] = [name]
        index = RAW_DIR / "sec" / "submissions" / f"CIK{cik10(cik)}.json"
        if index.exists():
            former = json.loads(index.read_text(encoding="utf-8")).get("formerNames", [])
            names[ticker] += [f["name"] for f in former if f.get("to", "") >= str(since_year)]
    return names


def build_text_change(con) -> None:
    items = list(ITEMS)
    sections = con.execute("""
        SELECT ticker, fiscal_year, item, accession_number, n_words, text
        FROM sections WHERE list_contains(?, item)
    """, [items]).df()
    chunks = con.execute("""
        SELECT chunk_id, accession_number, item, chunk_index, text FROM chunks WHERE list_contains(?, item)
    """, [items]).df()
    names = company_names(con, since_year=load_settings().start_fiscal_year - 1)
    result = compute_text_change(sections, chunks, names)
    database.replace_rows(con, "text_change", result)
    log.info("text_change: %d section pairs (%s)", len(result),
             ", ".join(f"{ITEMS[i]} {n}" for i, n in result["item"].value_counts().items()))


def build_tone(con) -> None:
    from filinglens.research.tone import compute_tone  # loads PyTorch

    sections = con.execute("""
        SELECT ticker, fiscal_year, accession_number, n_words, text FROM sections WHERE item = '7'
    """).df()
    result = compute_tone(sections)
    database.replace_rows(con, "tone", result)
    log.info("tone: %d MD&A sections, %d sentences", len(result), result["n_sentences"].sum())


def build_returns(con) -> None:
    benchmark = load_settings().benchmark
    filings = con.execute("SELECT ticker, fiscal_year, accession_number, filing_date FROM filings").df()
    prices = con.execute("SELECT ticker, date, close FROM prices").df()
    result = future_returns(filings, prices, benchmark)
    database.replace_rows(con, "future_returns", result)
    complete = {h: int(result[f"abnormal_{h}"].notna().sum()) for h in ("3m", "6m", "12m")}
    log.info("future_returns: %d reports; complete windows: %s", len(result), complete)


def build_topics(con) -> None:
    # The whole report: all its sections joined back together.
    reports = con.execute("""
        SELECT ticker, fiscal_year, accession_number, string_agg(text, '\n') AS text
        FROM sections GROUP BY ALL
    """).df()
    result = topic_mentions(reports)
    database.replace_rows(con, "topic_mentions", result)
    log.info("topic_mentions: %d reports x %d topics", len(reports), result["topic"].nunique())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute the research study's measures.")
    parser.add_argument("--only", nargs="+", choices=PARTS, help="Only these parts (default: all)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    builders = {"text_change": build_text_change, "tone": build_tone,
                "returns": build_returns, "topics": build_topics}
    con = database.connect()
    try:
        for part in args.only or PARTS:
            start = time.perf_counter()
            builders[part](con)
            log.info("%s done in %.0f s", part, time.perf_counter() - start)
    finally:
        con.close()
    print("\nDone. Open notebooks/04_research_study.ipynb for the statistics and charts.")


if __name__ == "__main__":
    main()
