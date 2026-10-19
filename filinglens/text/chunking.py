"""Step 2.3 - cut sections into chunks (passages) small enough to search well.

Why chunk at all? A Risk Factors section can be 15,000 words. To answer a
question we want the 2-3 paragraphs that matter, not the whole section, and
the embedding model in step 2.4 only reads about 400 words at a time.

How:
  - Chunks are built from whole paragraphs, up to MAX_WORDS each, so a chunk
    rarely cuts a thought in half. Paragraphs that are too long by themselves
    are split by sentence (and, as a last resort, by word count).
  - Consecutive chunks overlap by about OVERLAP_WORDS, so a sentence near a
    boundary still has its context in at least one chunk.
"""

import re

MAX_WORDS = 300       # ~400 tokens: fits the embedding model's 512-token limit with room to spare
OVERLAP_WORDS = 50

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"“])")


def _word_count(text: str) -> int:
    return len(text.split())


def _split_long(paragraph: str, limit: int) -> list[str]:
    """Split a paragraph longer than `limit` words into pieces of at most `limit` words."""
    pieces, current = [], []
    for sentence in SENTENCE_END.split(paragraph):
        words = sentence.split()
        while len(words) > limit:  # a single giant "sentence" (often a table row run)
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces.append(" ".join(words[:limit]))
            words = words[limit:]
        if len(current) + len(words) > limit:
            pieces.append(" ".join(current))
            current = []
        current.extend(words)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_text(text: str, max_words: int = MAX_WORDS, overlap_words: int = OVERLAP_WORDS) -> list[str]:
    # Units are paragraphs, small enough that one unit plus the overlap fits in a chunk.
    unit_limit = max_words - overlap_words
    units = []
    for paragraph in text.split("\n"):
        if _word_count(paragraph) > unit_limit:
            units.extend(_split_long(paragraph, unit_limit))
        elif paragraph.strip():
            units.append(paragraph)

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for unit in units:
        n = _word_count(unit)
        if current and current_words + n > max_words:
            chunks.append("\n".join(current))
            # Start the next chunk with the last paragraphs of this one (the overlap).
            tail, tail_words = [], 0
            for previous in reversed(current):
                w = _word_count(previous)
                if tail_words + w > overlap_words:
                    break
                tail.insert(0, previous)
                tail_words += w
            current, current_words = tail, tail_words
        current.append(unit)
        current_words += n
    if current:  # always ends with new text, never only the overlap
        chunks.append("\n".join(current))
    return chunks
