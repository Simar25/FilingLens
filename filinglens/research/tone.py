"""Step 4.2 - the tone of each MD&A section, scored by FinBERT.

General sentiment models misread financial language: "liabilities", "tax",
"risk" sound negative but are neutral in a 10-K, while "decreased costs" is
good news. FinBERT (ProsusAI/finbert) is a BERT model trained further on
financial news and labelled sentences (the Financial PhraseBank), so it rates
each sentence positive, negative or neutral the way an analyst would.

How a section gets one number:
  1. Split the MD&A into sentences. Table rows ("Americas | $167,045 | 3%")
     and fragments are skipped: they are numbers, not language.
  2. FinBERT gives each sentence three probabilities that add up to 1.
  3. tone = average over sentences of P(positive) - P(negative).
     -1 = everything negative, 0 = balanced or neutral, +1 = everything positive.
     Most MD&A sentences are neutral, so typical values are small (around 0).

The MD&A is used because it is management's own account of the year. Risk
Factors would be nearly all "negative" by design, so its tone says little.

The model (~440 MB) downloads into data/models/ on the first run, like the
embedding model, and runs on the GPU.
"""

import logging
import os
import re

import numpy as np
import pandas as pd

from filinglens.rag.embeddings import MODEL_DIR

log = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
MIN_SENTENCE_WORDS = 6
MIN_LETTER_SHARE = 0.6  # at least 60% of a sentence's words contain letters (skips number runs)
MAX_TOKENS = 128        # longer sentences are cut; the first 128 tokens carry their tone
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"“])")


def sentences(text: str) -> list[str]:
    """The sentences of a section that are language, not table rows or fragments."""
    result = []
    for line in text.split("\n"):
        if "|" in line:  # html_to_text writes table cells separated by " | "
            continue
        for sentence in SENTENCE_END.split(line):
            tokens = sentence.split()
            if len(tokens) < MIN_SENTENCE_WORDS:
                continue
            if sum(any(ch.isalpha() for ch in t) for t in tokens) / len(tokens) >= MIN_LETTER_SHARE:
                result.append(sentence.strip())
    return result


class FinBert:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        os.environ.setdefault("HF_HOME", str(MODEL_DIR))  # keep the download on E:, not in the user folder
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
        except OSError:  # not downloaded yet
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        # The model's own label order, e.g. {0: 'positive', 1: 'negative', 2: 'neutral'}
        labels = {name.lower(): i for i, name in self.model.config.id2label.items()}
        self.positive, self.negative = labels["positive"], labels["negative"]

    def probabilities(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        """One row per text: [P(positive), P(negative)]."""
        out = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = self.tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                       max_length=MAX_TOKENS, return_tensors="pt").to(self.device)
                probs = self.torch.softmax(self.model(**batch).logits.float(), dim=-1)
                out.append(probs[:, [self.positive, self.negative]].cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, 2))


def section_tone(probs: np.ndarray) -> dict:
    """Summarise a section's sentence probabilities: tone and the share of clearly positive/negative sentences."""
    positive, negative = probs[:, 0], probs[:, 1]
    neutral = 1 - positive - negative
    return {
        "n_sentences": len(probs),
        "tone": float((positive - negative).mean()),
        "positive_share": float(((positive > negative) & (positive > neutral)).mean()),
        "negative_share": float(((negative > positive) & (negative > neutral)).mean()),
    }


def compute_tone(sections: pd.DataFrame, min_words: int = 500) -> pd.DataFrame:
    """One row per MD&A section. `sections`: ticker, fiscal_year, accession_number, n_words, text."""
    usable = sections[sections["n_words"] >= min_words]
    per_section = [sentences(text) for text in usable["text"]]
    all_sentences = [s for group in per_section for s in group]

    model = FinBert()
    log.info("Scoring %d MD&A sentences from %d reports with FinBERT on %s",
             len(all_sentences), len(usable), model.device.upper())
    # Sorting by length puts similar lengths in one batch, so there's little padding: much faster.
    order = np.argsort([len(s) for s in all_sentences])
    probs = np.empty((len(all_sentences), 2))
    probs[order] = model.probabilities([all_sentences[i] for i in order])

    rows, start = [], 0
    for section, group in zip(usable.itertuples(), per_section):
        part = probs[start:start + len(group)]
        start += len(group)
        if len(part):
            rows.append({"ticker": section.ticker, "fiscal_year": section.fiscal_year,
                         "accession_number": section.accession_number, **section_tone(part)})
    return pd.DataFrame(rows)
