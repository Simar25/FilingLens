"""Step 4.1 - how much did a report section change since last year?

"Lazy Prices" (Cohen, Malloy & Nguyen, 2020) found that most companies copy
last year's 10-K almost word for word, and that the ones that change a lot
tend to have worse stock returns afterwards. We measure the change of each
company's Risk Factors (Item 1A) and MD&A (Item 7) between consecutive years
in three ways. Each gives a similarity from 0 (nothing in common) to 1 (same):

  cosine      Word counts, as in the paper. Each section becomes a vector of
              how often each word appears; the similarity is the cosine of the
              angle between the two vectors. Rewording a few paragraphs barely
              moves it; adding a long new risk does.
  jaccard     Word sets, also in the paper: words used in both years / words
              used in either year. Ignores how often a word appears, so it
              reacts to new vocabulary ("tariff", "pandemic").
  embedding   Meaning, not exact words (our addition). Each section is cut
              into the same ~300-word chunks as Step 2 and embedded with the
              same model. For every chunk we find its closest chunk in the
              other year; the score is the average of those best matches.
              A rewritten but equivalent paragraph still scores high; a
              genuinely new risk has no close match and pulls it down.

The change score is 1 - similarity, so higher = changed more.

Which words are compared (decided by looking at the text, before any returns):
  - Letters only, no numbers: every MD&A updates its figures and years
    ("2024" -> "2025"), which is routine, not a change in what it says.
  - No stop words ("the", "of", "and"...). They are the most frequent words in
    every text, so with them in, any two reports look ~99.7% alike and the
    score barely varies.
  - Not the company's own name. JPMorgan started writing "JPMorganChase" as one
    word in 2024: two of its most frequent words vanished and a new one appeared,
    which made it the "most changed" report of all. Names used during the study
    period come from SEC, including former ones (Facebook -> Meta, 2021).
Removing the same words from both years can't create a change that isn't there.
"""

import hashlib
import logging
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from filinglens.config import PROJECT_ROOT

log = logging.getLogger(__name__)

ITEMS = {"1A": "Risk Factors", "7": "MD&A"}
MIN_WORDS = 500  # shorter "sections" are usually a pointer to another part of the report
WORD = re.compile(r"[a-z]+")
VECTOR_CACHE = PROJECT_ROOT / "data" / "processed" / "research" / "chunk_vectors.npz"


def words(text: str, ignore: frozenset[str] = frozenset()) -> list[str]:
    """The content words of a text: lowercase, letters only, without stop words and `ignore`."""
    return [w for w in WORD.findall(text.lower())
            if len(w) > 1 and w not in ENGLISH_STOP_WORDS and w not in ignore]


def name_words(names: list[str]) -> frozenset[str]:
    """Words of a company's names, plus those words run together.

    ["JPMORGAN CHASE & CO"] -> {"jpmorgan", "chase", "co", "jpmorganchase", "chaseco", "jpmorganchaseco"}
    """
    result = set()
    for name in names:
        tokens = WORD.findall(name.lower())
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens) + 1):
                result.add("".join(tokens[i:j]))
    return frozenset(result)


def cosine_similarity(a: list[str], b: list[str]) -> float:
    counts_a, counts_b = Counter(a), Counter(b)
    dot = sum(counts_a[w] * counts_b[w] for w in counts_a.keys() & counts_b.keys())
    norm = math.sqrt(sum(n * n for n in counts_a.values())) * math.sqrt(sum(n * n for n in counts_b.values()))
    return dot / norm if norm else float("nan")


def jaccard_similarity(a: list[str], b: list[str]) -> float:
    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else float("nan")


def embedding_similarity(vectors_a: np.ndarray, vectors_b: np.ndarray) -> float:
    """Average best-match similarity between two sets of normalized chunk vectors, in both directions.

    Both directions, because a risk can be added (a chunk this year with no match
    last year) or removed (a chunk last year with no match this year).
    """
    if len(vectors_a) == 0 or len(vectors_b) == 0:
        return float("nan")
    sims = vectors_a @ vectors_b.T  # vectors have length 1, so dot product = cosine similarity
    return float((sims.max(axis=1).mean() + sims.max(axis=0).mean()) / 2)


def consecutive_pairs(sections: pd.DataFrame) -> pd.DataFrame:
    """Pair each company's section with the same section one fiscal year earlier.

    `sections` has columns ticker, fiscal_year, item, accession_number, n_words.
    Pairs are only made when both years exist and are long enough: a gap year or
    a missing section gives no pair instead of a misleading comparison.
    """
    usable = sections[sections["n_words"] >= MIN_WORDS]
    previous = usable.assign(fiscal_year=usable["fiscal_year"] + 1)
    return usable.merge(previous, on=["ticker", "fiscal_year", "item"], suffixes=("", "_prev"))


def text_vectors(texts: list[str], cache: Path = VECTOR_CACHE) -> np.ndarray:
    """One normalized vector per text, embedding only texts that aren't in the cache file yet.

    The cache is keyed by a fingerprint (hash) of the text itself, so if a report is
    parsed differently later, its changed chunks get new vectors automatically.
    Chunks are embedded on their own, without the "Company, year, section" line that
    Step 2 puts in front of them for search: here that line would differ only in the
    year, and the comparison should be about the text itself.
    """
    keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    cached: dict[str, np.ndarray] = {}
    if cache.exists():
        saved = np.load(cache)
        cached = dict(zip(saved["keys"].tolist(), saved["vectors"]))
    todo = {k: t for k, t in zip(keys, texts) if k not in cached}
    if todo:
        from filinglens.rag.embeddings import Embedder  # loads PyTorch: only when there is work to do

        embedder = Embedder()
        log.info("Embedding %d chunks on %s", len(todo), embedder.device.upper())
        vectors = embedder.embed_passages(list(todo.values()), show_progress=True)
        cached.update(zip(todo.keys(), vectors))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, keys=np.array(list(cached.keys())), vectors=np.stack(list(cached.values())))
    return np.stack([cached[k] for k in keys]) if keys else np.empty((0, 0), dtype=np.float32)


def compute_text_change(sections: pd.DataFrame, chunks: pd.DataFrame,
                        company_names: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """One row per company, fiscal year and Item: similarities to the previous year and change scores.

    sections:      ticker, fiscal_year, item, accession_number, n_words, text   (Items 1A and 7)
    chunks:        chunk_id, accession_number, item, chunk_index, text          (the same Items)
    company_names: {ticker: [current name, former names...]}, left out of the word measures
    """
    company_names = company_names or {}
    pairs = consecutive_pairs(sections.drop(columns="text"))
    text = sections.set_index(["accession_number", "item"])["text"]
    needed = chunks[chunks["accession_number"].isin(
        set(pairs["accession_number"]) | set(pairs["accession_number_prev"]))].reset_index(drop=True)
    vectors = text_vectors(needed["text"].tolist())
    rows_by_section = needed.groupby(["accession_number", "item"]).indices  # (accession, item) -> row numbers

    def stacked(accession: str, item: str) -> np.ndarray:
        return vectors[rows_by_section.get((accession, item), [])]

    rows = []
    for p in pairs.itertuples():
        ignore = name_words(company_names.get(p.ticker, []))
        now = words(text[(p.accession_number, p.item)], ignore)
        before = words(text[(p.accession_number_prev, p.item)], ignore)
        rows.append({
            "ticker": p.ticker, "fiscal_year": p.fiscal_year, "item": p.item,
            "accession_number": p.accession_number, "prev_accession_number": p.accession_number_prev,
            "n_words": p.n_words, "prev_n_words": p.n_words_prev,
            "sim_cosine": cosine_similarity(now, before),
            "sim_jaccard": jaccard_similarity(now, before),
            "sim_embedding": embedding_similarity(stacked(p.accession_number, p.item),
                                                  stacked(p.accession_number_prev, p.item)),
        })
    result = pd.DataFrame(rows)
    for measure in ("cosine", "jaccard", "embedding"):
        result[f"change_{measure}"] = 1 - result[f"sim_{measure}"]
    return result
