"""Step 2.4 - turn text into embeddings: lists of numbers that capture meaning.

An embedding model reads a passage and outputs a vector (here 768 numbers).
Passages with similar meanings get vectors that point in similar directions,
even when they share no words: "chip shortages" lands close to "constrained
semiconductor supply". To search, we embed the question the same way and
look for the passages whose vectors are closest (cosine similarity).

Model: BAAI/bge-base-en-v1.5, a free model that ranks well on retrieval
benchmarks for its size. It reads at most 512 tokens per passage (anything
longer is ignored), which is why chunks are capped at 300 words.

Two details specific to this model family:
  - Questions get a short instruction in front of them. Questions and passages
    look very different (one line vs. a paragraph), and the model was trained
    with this prefix to bridge that gap.
  - Vectors are normalized to length 1, so cosine similarity is just a dot product.
"""

import os

import numpy as np

from filinglens.config import PROJECT_ROOT

MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# Hugging Face saves downloaded models in your user folder (on C: on Windows).
# Keep them with the project's other downloads instead. Setting HF_HOME yourself overrides this.
MODEL_DIR = PROJECT_ROOT / "data" / "models"


def passage_text(company: str, ticker: str, fiscal_year: int, title: str, text: str) -> str:
    """The text that gets embedded: the chunk plus a one-line header saying where it's from.

    A Risk Factors paragraph rarely names its own company or year ("our products
    face competition..."). The header puts that context into the vector, so
    "Apple's competition risks" matches Apple's paragraph better than Dell's.
    """
    return f"{company} ({ticker}) 10-K, fiscal year {fiscal_year}, {title}\n{text}"


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        # Imported here, not at the top: loading PyTorch takes a few seconds, and
        # commands like `ask --help` or the tests shouldn't have to wait for it.
        os.environ.setdefault("HF_HOME", str(MODEL_DIR))  # must be set before Hugging Face is imported
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()  # a "Loading weights" bar on every start
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Use the downloaded copy without first asking Hugging Face whether the model has
            # changed: that check adds several seconds to every question.
            self.model = SentenceTransformer(model_name, device=self.device, local_files_only=True)
        except OSError:  # not downloaded yet
            self.model = SentenceTransformer(model_name, device=self.device)
        if self.device == "cuda":
            self.model.half()  # 16-bit numbers on the GPU: about twice as fast, same search results
        self.model_name = model_name

    def embed_passages(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        vectors = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=show_progress)
        return vectors.astype(np.float32)

    def embed_query(self, question: str) -> np.ndarray:
        return self.embed_passages([QUERY_INSTRUCTION + question])[0]

    def count_tokens(self, text: str) -> int:
        # verbose=False: counting a passage over the limit is the point here, not worth a warning
        return len(self.model.tokenizer(text, add_special_tokens=True, verbose=False)["input_ids"])
