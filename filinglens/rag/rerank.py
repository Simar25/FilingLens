"""Step 5.4 - re-score the best search results with a cross-encoder (a "reranker").

The embedding model reads the question and each passage separately, turns each
into a vector, and compares the vectors. That is what makes it fast enough to
search 190,000 chunks, but the two texts never "see" each other.

A cross-encoder reads the question and one passage together, as one input, and
outputs how well the passage answers the question. It can notice that a passage
mentions R&D but explains a different year, which two separate vectors blur.
It is far too slow to run on every chunk (one model run per passage), so it only
re-scores the top ~20 results of the fast search and keeps the best k.

Model: cross-encoder/ms-marco-MiniLM-L6-v2 (~90 MB), a small, widely used
reranker trained on Bing search questions and the web passages that answered them.
"""

import os
from dataclasses import replace

from filinglens.rag.embeddings import MODEL_DIR, passage_text
from filinglens.rag.passage import SearchResult

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


class Reranker:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        os.environ.setdefault("HF_HOME", str(MODEL_DIR))  # same model folder as the embedding model
        import torch
        from sentence_transformers import CrossEncoder

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = CrossEncoder(model_name, device=device, local_files_only=True)
        except OSError:  # not downloaded yet
            self.model = CrossEncoder(model_name, device=device)
        self.model_name = model_name

    def rerank(self, question: str, results: list[SearchResult], k: int) -> list[SearchResult]:
        if not results:
            return []
        pairs = [(question, passage_text(r.company, r.ticker, r.fiscal_year, r.title, r.text)) for r in results]
        scores = self.model.predict(pairs, batch_size=32)
        order = sorted(range(len(results)), key=lambda i: -scores[i])[:k]
        return [replace(results[i], score=float(scores[i])) for i in order]
