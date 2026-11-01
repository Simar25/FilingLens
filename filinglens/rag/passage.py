"""One report passage found by a search. It lives in its own file so that code which only handles
passages doesn't load the vector database, which the online version can't afford (filinglens/online.py)."""

from dataclasses import dataclass


@dataclass
class SearchResult:
    chunk_id: str
    ticker: str
    company: str
    fiscal_year: int
    item: str
    title: str
    text: str
    # Higher = better match. What it measures depends on the search that found it: cosine similarity
    # (vector, 0 to 1), BM25 (keyword), fused rank (hybrid) or the reranker's score.
    score: float

    @property
    def section(self) -> str:
        # Reports without the usual Items have titles like "Report body (company's own layout)".
        return self.title.split(" (")[0] if self.item in ("AR", "BODY", "NSE") else f"Item {self.item} {self.title}"

    @property
    def label(self) -> str:
        report = "annual report" if self.item == "NSE" else "10-K"
        return f"{self.ticker} {report} FY{self.fiscal_year}, {self.section}"
