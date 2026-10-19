"""Step 2.5 - store the embeddings in ChromaDB and search them.

A vector database answers one question fast: "which stored vectors are closest
to this one?" Comparing a question with all 21,000 chunks one by one would be
fine today, but not with 100,000+ chunks and many users. Chroma builds an HNSW
index (a graph that links each vector to its near neighbours), so a search
only visits a small part of the collection.

Each vector is stored with its chunk's text and metadata (company, year,
Item). Metadata filters run before the similarity search, so "Apple, 2024"
only ever compares against Apple's 2024 chunks.

The database lives in data/processed/chroma/. It is rebuilt from the 'chunks'
table by `python -m filinglens.build_index`.
"""

from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd

from filinglens.config import PROJECT_ROOT
from filinglens.rag.filters import Filters

CHROMA_DIR = PROJECT_ROOT / "data" / "processed" / "chroma"
COLLECTION = "chunks"
METADATA_COLUMNS = ["ticker", "company", "fiscal_year", "item", "title", "accession_number", "chunk_index"]


@dataclass
class SearchResult:
    chunk_id: str
    ticker: str
    company: str
    fiscal_year: int
    item: str
    title: str
    text: str
    similarity: float  # cosine similarity: 1 = same direction, 0 = unrelated

    @property
    def section(self) -> str:
        return self.title if self.item in ("AR", "BODY") else f"Item {self.item} {self.title}"

    @property
    def label(self) -> str:
        return f"{self.ticker} 10-K FY{self.fiscal_year}, {self.section}"


def build_where(filters: Filters) -> dict | None:
    """Filters -> Chroma's filter syntax, e.g. {"ticker": {"$in": ["AAPL"]}}."""
    conditions = []
    if filters.tickers:
        conditions.append({"ticker": {"$in": filters.tickers}})
    if filters.years:
        conditions.append({"fiscal_year": {"$in": filters.years}})
    if filters.items:
        conditions.append({"item": {"$in": filters.items}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


class VectorStore:
    def __init__(self, path: Path = CHROMA_DIR):
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path), settings=chromadb.Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            COLLECTION,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,  # we compute embeddings ourselves (embeddings.py)
        )

    def count(self) -> int:
        return self.collection.count()

    def delete_tickers(self, tickers: list[str]) -> None:
        self.collection.delete(where={"ticker": {"$in": tickers}})

    def add(self, chunks: pd.DataFrame, embeddings: np.ndarray) -> None:
        """Add chunks (rows of the 'chunks' table plus a 'company' column) and their vectors."""
        batch = self.client.get_max_batch_size()
        for start in range(0, len(chunks), batch):
            part = chunks.iloc[start:start + batch]
            self.collection.add(
                ids=part["chunk_id"].tolist(),
                embeddings=embeddings[start:start + batch],
                documents=part["text"].tolist(),
                metadatas=[{k: _plain(v) for k, v in row.items()}
                           for row in part[METADATA_COLUMNS].to_dict("records")],
            )

    def search(self, query_embedding: np.ndarray, k: int = 8, filters: Filters | None = None) -> list[SearchResult]:
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=build_where(filters or Filters()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            SearchResult(
                chunk_id=chunk_id, ticker=meta["ticker"], company=meta["company"],
                fiscal_year=meta["fiscal_year"], item=meta["item"], title=meta["title"],
                text=text, similarity=1 - distance,  # Chroma returns cosine distance = 1 - similarity
            )
            for chunk_id, meta, text, distance in zip(
                result["ids"][0], result["metadatas"][0], result["documents"][0], result["distances"][0]
            )
        ]


def _plain(value):
    """numpy numbers -> plain Python numbers (Chroma only stores str, int, float, bool)."""
    return value.item() if isinstance(value, np.generic) else value
