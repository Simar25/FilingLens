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
        return self.title if self.item in ("AR", "BODY") else f"Item {self.item} {self.title}"

    @property
    def label(self) -> str:
        return f"{self.ticker} 10-K FY{self.fiscal_year}, {self.section}"
