"""Step 5.3 - keyword search (BM25), and merging it with the vector search.

Vector search matches meaning, so it can miss the exact terms a question hinges
on: a product ("H20 chips"), a rule ("Item 1C", "Basel III"), an abbreviation
("R&D"). Keyword search finds the passages that contain the question's words and
scores them with BM25, the formula behind most search engines before embeddings:

    score = sum over the question's words of
            idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * passage length / average length))

  - tf:  how often the word appears in the passage. More is better, with diminishing
         returns: the 10th "tariff" adds much less than the 1st (k1 sets how fast it levels off).
  - idf: rare words count more. "cybersecurity" tells you more than "company".
  - length: a long passage contains more words by chance, so it is scored down a bit (b).

The index is a big table with one row per chunk and one column per word, holding
the BM25 weight of each word in each chunk (almost all zeros, so it is stored as
a sparse matrix). A search just adds up the columns of the question's words.
It's built by `python -m filinglens.build_index` and saved next to the database.

Hybrid search runs both searches and merges the two ranked lists with Reciprocal
Rank Fusion: each passage scores 1 / (60 + its rank) in each list, summed. Only
ranks are used, because a BM25 score (0 to ~50) and a cosine similarity (0 to 1)
can't be compared, and a passage both searches rank well comes out on top.
"""

import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from filinglens.config import PROJECT_ROOT
from filinglens.rag.filters import Filters
from filinglens.rag.vector_store import SearchResult

INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "keyword_index.pkl"
K1, B = 1.2, 0.75  # the usual BM25 settings
# A word is a run of letters and digits, so "2024" and "H20" count; "&" inside a word is kept: R&D, AT&T, P&G.
TOKEN_PATTERN = r"(?u)\w+(?:&\w+)*"
RRF_K = 60  # from the paper that introduced RRF (Cormack et al., 2009): rank 1 and 2 score almost the same
FILTER_COLUMNS = ["chunk_id", "ticker", "fiscal_year", "item"]


def make_vectorizer(**options):
    # Common English words ("the", "how", "did") are dropped: they match almost every passage.
    # shortcut: no stemming, so "spending" doesn't match "spend"; add a stemmer if failures show it matters.
    from sklearn.feature_extraction.text import CountVectorizer  # here: importing scikit-learn takes ~1 s

    return CountVectorizer(token_pattern=TOKEN_PATTERN, stop_words="english", **options)


class KeywordIndex:
    def __init__(self, weights: sp.csc_matrix, words: list[str], chunks: pd.DataFrame):
        self.weights = weights  # chunks x words: BM25 weight of each word in each chunk
        self.words = words
        self.chunks = chunks    # FILTER_COLUMNS, one row per row of `weights`
        self.column = {word: i for i, word in enumerate(words)}
        self.analyze = make_vectorizer().build_analyzer()  # question -> words, the same way as the passages

    @classmethod
    def build(cls, chunks: pd.DataFrame, texts: list[str]) -> "KeywordIndex":
        """`texts[i]` is the text indexed for `chunks.iloc[i]` (the same text the embedding model reads)."""
        # min_df=2: a word found in only one chunk is mostly a typo or a number, and would double the word list.
        vectorizer = make_vectorizer(min_df=2, dtype=np.float32)
        counts = vectorizer.fit_transform(texts).tocoo()  # counts[i, j] = how often word j appears in chunk i
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        doc_freq = np.bincount(counts.col, minlength=counts.shape[1])  # how many chunks contain each word
        idf = np.log(1 + (len(texts) - doc_freq + 0.5) / (doc_freq + 0.5))
        tf = counts.data
        weight = idf[counts.col] * tf * (K1 + 1) / (tf + K1 * (1 - B + B * lengths[counts.row] / lengths.mean()))
        weights = sp.csc_matrix((weight.astype(np.float32), (counts.row, counts.col)), shape=counts.shape)
        return cls(weights, vectorizer.get_feature_names_out().tolist(), chunks[FILTER_COLUMNS].reset_index(drop=True))

    def save(self, path: Path = INDEX_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump({"weights": self.weights, "words": self.words, "chunks": self.chunks}, f)

    @classmethod
    def load(cls, path: Path = INDEX_PATH) -> "KeywordIndex":
        # pickle can run code while loading, which is fine for this file (build_index writes it on your own
        # computer) but never for one downloaded from elsewhere.
        with open(path, "rb") as f:
            return cls(**pickle.load(f))

    def search(self, question: str, filters: Filters, k: int) -> list[tuple[str, float]]:
        """The k best (chunk_id, BM25 score) pairs within the filters, best first."""
        columns = sorted({self.column[w] for w in self.analyze(question) if w in self.column})
        if not columns:
            return []
        scores = np.asarray(self.weights[:, columns].sum(axis=1)).ravel()
        for column, wanted in (("ticker", filters.tickers), ("fiscal_year", filters.years), ("item", filters.items)):
            if wanted:
                scores[~self.chunks[column].isin(wanted).to_numpy()] = 0
        best = np.argsort(-scores, kind="stable")[:k]
        return [(self.chunks["chunk_id"].iat[i], float(scores[i])) for i in best if scores[i] > 0]


def reciprocal_rank_fusion(*ranked_lists: list[SearchResult]) -> list[SearchResult]:
    """Merge ranked lists: each result scores 1 / (RRF_K + rank) per list it's in. Best first."""
    scores, results = {}, {}
    for ranked in ranked_lists:
        for rank, result in enumerate(ranked, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0) + 1 / (RRF_K + rank)
            results.setdefault(result.chunk_id, result)
    order = sorted(scores, key=scores.get, reverse=True)  # a stable sort: ties keep the first list's order
    return [replace(results[chunk_id], score=scores[chunk_id]) for chunk_id in order]
