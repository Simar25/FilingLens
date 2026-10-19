"""Step 2, parts 1-3: turn the downloaded 10-K reports into sections and chunks.

Run with:  python -m filinglens.parse_reports

For every report in the 'filings' table:
  2.1  HTML -> clean text       saved to data/processed/text/<TICKER>/ so you can read it
  2.2  text -> sections (Items) -> 'sections' table
  2.3  sections -> chunks       -> 'chunks' table

Then it prints a quality report: reports where Risk Factors (1A) or MD&A (7)
looks missing or suspiciously short.
"""

import logging

import pandas as pd

from filinglens import database
from filinglens.config import PROJECT_ROOT
from filinglens.text.chunking import chunk_text
from filinglens.text.html_to_text import html_to_text
from filinglens.text.sections import split_sections

log = logging.getLogger("filinglens")

TEXT_DIR = PROJECT_ROOT / "data" / "processed" / "text"
# Exhibit lists and the one-line "Form 10-K Summary" don't help answer questions.
SKIP_ITEMS_FOR_CHUNKS = {"15", "16"}
MIN_SECTION_WORDS = 50  # skips "None." and "Not applicable." sections
KEY_ITEMS = {"1A": "Risk Factors", "7": "MD&A"}
SUSPICIOUSLY_SHORT = 500  # words


def parse_report(report) -> tuple[list[dict], list[dict]]:
    html = (PROJECT_ROOT / report.local_path).read_text(encoding="utf-8", errors="replace")
    text = html_to_text(html)

    text_path = TEXT_DIR / report.ticker / f"FY{report.fiscal_year}_{report.accession_number}.txt"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text, encoding="utf-8")

    base = {"accession_number": report.accession_number, "ticker": report.ticker,
            "fiscal_year": report.fiscal_year}
    sections, chunks = [], []
    for section in split_sections(text):
        sections.append({**base, "item": section.item, "title": section.title,
                         "text": section.text, "n_words": section.n_words})
        if section.item in SKIP_ITEMS_FOR_CHUNKS or section.n_words < MIN_SECTION_WORDS:
            continue
        for i, chunk in enumerate(chunk_text(section.text)):
            chunks.append({**base, "item": section.item, "title": section.title, "chunk_index": i,
                           "chunk_id": f"{report.ticker}-{report.fiscal_year}-{section.item}-{i:04d}",
                           "text": chunk, "n_words": len(chunk.split())})
    return sections, chunks


def quality_report(sections: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    """One row per report with a problem in a key section."""
    words = sections.pivot_table(index="accession_number", columns="item", values="n_words", aggfunc="sum")
    problems = []
    for report in reports.itertuples():
        row = words.loc[report.accession_number] if report.accession_number in words.index else pd.Series()
        if pd.notna(row.get("AR")):
            where = " (text is in the attached annual report)"
        elif pd.notna(row.get("BODY")):
            where = " (company's own layout: text is in the report body)"
        else:
            where = ""
        for item, name in KEY_ITEMS.items():
            n = row.get(item)
            if pd.isna(n):
                issue = "missing" + where
            elif n < SUSPICIOUSLY_SHORT:
                issue = f"only {int(n)} words" + where
            else:
                continue
            problems.append({"ticker": report.ticker, "fiscal_year": report.fiscal_year,
                             "section": f"{item} {name}", "issue": issue})
    return pd.DataFrame(problems)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    con = database.connect()
    try:
        reports = con.execute("""
            SELECT ticker, fiscal_year, accession_number, local_path FROM filings
            WHERE local_path IS NOT NULL ORDER BY ticker, fiscal_year
        """).df()
        all_sections, all_chunks = [], []
        for ticker, group in reports.groupby("ticker"):
            n_sections = n_chunks = 0
            for report in group.itertuples():
                sections, chunks = parse_report(report)
                all_sections += sections
                all_chunks += chunks
                n_sections += len(sections)
                n_chunks += len(chunks)
            log.info("%-5s %d reports -> %d sections, %d chunks", ticker, len(group), n_sections, n_chunks)

        sections_df, chunks_df = pd.DataFrame(all_sections), pd.DataFrame(all_chunks)
        database.replace_rows(con, "sections", sections_df)
        database.replace_rows(con, "chunks", chunks_df)
    finally:
        con.close()

    print(f"\n{len(reports)} reports -> {len(sections_df):,} sections, {len(chunks_df):,} chunks "
          f"(average {chunks_df['n_words'].mean():.0f} words per chunk)")
    problems = quality_report(sections_df, reports)
    if problems.empty:
        print("Quality check: every report has a full Risk Factors and MD&A section.")
    else:
        print("\nQuality check - reports to look at:")
        print(problems.to_string(index=False))
    print(f"\nClean text of every report: {TEXT_DIR}")


if __name__ == "__main__":
    main()
