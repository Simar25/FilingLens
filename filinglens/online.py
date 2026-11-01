"""The free online version: the same app on a host too small for the AI embedding model and 4.8 GB of data.

Everything is the same code except the search. The vector database and the embedding model don't fit,
so the online version keeps one compressed file per report in online/reports/ and, for each question,
runs the keyword search (BM25, Step 5.3) over the reports of the companies it names. The database shrinks
to what text-to-SQL can read (online/facts.duckdb). On the test questions keyword search found the right
passage in the top 5 for 90% of questions, against 94% for the hybrid search of the full version.

Rebuild the files after the data changes:  python -m filinglens.online
Turn it on with the environment variable FILINGLENS_ONLINE=1 (the deployed app sets it as a secret).
"""

import gzip
import json
import os
import re
from collections.abc import Callable
from functools import lru_cache

import pandas as pd

from filinglens import llm
from filinglens.assistant import Assistant, SetupError
from filinglens.config import DB_PATH, PROJECT_ROOT
from filinglens.database import create_views
from filinglens.llm import LLMReply
from filinglens.rag.embeddings import passage_text
from filinglens.rag.filters import Filters
from filinglens.rag.keyword_search import KeywordIndex
from filinglens.rag.passage import SearchResult

ONLINE = bool(os.environ.get("FILINGLENS_ONLINE"))
ONLINE_DIR = PROJECT_ROOT / "online"
FACTS_PATH = ONLINE_DIR / "facts.duckdb"
REPORTS_DIR = ONLINE_DIR / "reports"
COPIED_TABLES = ("companies", "company_info", "filings", "financials", "prices")
MAX_REPORTS = 12  # enough for two companies over all six years; more would slow every other visitor
# Reports explain a change in other words than a question asks about it: "why did revenue change?" finds
# nothing, while management writes "decreased, primarily due to lower demand". So these words join the search.
WHY_QUESTION = re.compile(r"\b(why|reasons?|because|caused?|causes|drove|drive[ns]?|drivers?|explains?|behind)\b",
                          re.IGNORECASE)
WHY_WORDS = "decreased increased primarily due lower higher compared"  # no topic words: the question brings those


@lru_cache(maxsize=64)
def load_report(ticker: str, year: int) -> tuple[SearchResult, ...]:
    path = REPORTS_DIR / f"{ticker}-{year}.json.gz"
    if not path.exists():
        return ()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        report = json.load(f)
    return tuple(SearchResult(chunk_id, ticker, report["company"], year, item, title, text, 0.0)
                 for chunk_id, item, title, text in report["passages"])


class OnlineAssistant(Assistant):
    def __init__(self, db_path=FACTS_PATH, generate: Callable[[str, str], LLMReply] = llm.generate):
        super().__init__(db_path, generate)

    def search(self, question: str, filters: Filters, k: int = 8, mode: str = "keyword") -> list[SearchResult]:
        if not filters.tickers:
            raise SetupError(f"Name one of the {len(self.companies)} companies on the market map in your question, for example: "
                             "What supply chain risks did Apple describe in 2024?")
        years = filters.years or self.report_years
        wanted = [(ticker, year) for ticker in filters.tickers for year in years
                  if (REPORTS_DIR / f"{ticker}-{year}.json.gz").exists()][:MAX_REPORTS]
        passages = [passage for ticker, year in wanted for passage in load_report(ticker, year)]
        if not passages:
            return []
        table = pd.DataFrame({"chunk_id": [p.chunk_id for p in passages], "ticker": [p.ticker for p in passages],
                              "fiscal_year": [p.fiscal_year for p in passages], "item": [p.item for p in passages]})
        index = KeywordIndex.build(table, [passage_text(p.company, p.ticker, p.fiscal_year, p.title, p.text)
                                           for p in passages], min_df=1)
        found = {p.chunk_id: p for p in passages}
        query = f"{question} {WHY_WORDS}" if WHY_QUESTION.search(question) else question
        return [found[chunk_id] for chunk_id, _ in index.search(query, filters, k)]


def main() -> None:
    import duckdb

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FACTS_PATH.unlink(missing_ok=True)
    con = duckdb.connect(str(FACTS_PATH))
    con.execute(f"ATTACH '{DB_PATH.as_posix()}' AS source (READ_ONLY)")
    for table in COPIED_TABLES:
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM source.{table}")
    # The app only counts the passages and lists the fiscal years, so that's all that's kept of them here.
    con.execute("CREATE TABLE chunks AS SELECT fiscal_year FROM source.chunks")
    create_views(con)
    rows = con.execute("""SELECT c.ticker, c.fiscal_year, co.name, c.chunk_id, c.item, c.title, c.text
                          FROM source.chunks c JOIN source.companies co USING (ticker)
                          ORDER BY c.ticker, c.fiscal_year, c.chunk_index""").df()
    con.execute("DETACH source")
    con.close()
    for (ticker, year), group in rows.groupby(["ticker", "fiscal_year"]):
        report = {"company": group["name"].iat[0], "passages": group[["chunk_id", "item", "title", "text"]].values.tolist()}
        path = REPORTS_DIR / f"{ticker}-{year}.json.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as f:
                if json.load(f) == report:
                    continue  # gzip output differs on every write, and each rewrite would add 80 MB to the repository
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
            json.dump(report, f, ensure_ascii=False, separators=(",", ":"))
    size = sum(p.stat().st_size for p in REPORTS_DIR.iterdir()) / 1e6
    print(f"{rows.groupby(['ticker', 'fiscal_year']).ngroups} reports, {size:.0f} MB; "
          f"facts database {FACTS_PATH.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
