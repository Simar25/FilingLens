"""Step 2, parts 4-5: embed every chunk and store it in the vector database.

Run with:  python -m filinglens.build_index            (all companies)
           python -m filinglens.build_index --tickers AAPL NVDA

Reads the 'chunks' table made by `python -m filinglens.parse_reports`, so run
that first. Each company's vectors are deleted and re-added, so re-running is
safe and picks up any change in the chunks (same idea as database.replace_rows).

The first run downloads the embedding model (~440 MB) from Hugging Face into data/models/.
"""

import argparse
import logging
import time

import duckdb

from filinglens.config import DB_PATH
from filinglens.rag.embeddings import Embedder, passage_text
from filinglens.rag.vector_store import CHROMA_DIR, VectorStore

log = logging.getLogger("filinglens")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Embed report chunks into the vector database.")
    parser.add_argument("--tickers", nargs="+", help="Only these companies (default: all)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # Hugging Face's web client logs every request at INFO

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        chunks = con.execute("""
            SELECT c.*, co.name AS company FROM chunks c JOIN companies co USING (ticker)
            ORDER BY c.ticker, c.fiscal_year, c.chunk_id
        """).df()
    finally:
        con.close()
    if args.tickers:
        chunks = chunks[chunks["ticker"].isin([t.upper() for t in args.tickers])]
    if chunks.empty:
        raise SystemExit("No chunks found. Run  python -m filinglens.parse_reports  first.")

    start = time.perf_counter()
    embedder = Embedder()
    log.info("Embedding model %s loaded on %s", embedder.model_name, embedder.device.upper())
    if embedder.device == "cpu":
        log.warning("No GPU found: this will be slow. Check that the CUDA version of PyTorch is installed.")

    store = VectorStore()
    too_long = 0
    for ticker, group in chunks.groupby("ticker"):
        texts = [passage_text(r.company, r.ticker, r.fiscal_year, r.title, r.text) for r in group.itertuples()]
        t0 = time.perf_counter()
        vectors = embedder.embed_passages(texts)
        store.delete_tickers([ticker])
        store.add(group, vectors)
        too_long += sum(embedder.count_tokens(t) > embedder.model.max_seq_length for t in texts)
        log.info("%-5s %5d chunks embedded in %4.1f s", ticker, len(group), time.perf_counter() - t0)

    print(f"\n{len(chunks):,} chunks embedded in {time.perf_counter() - start:.0f} s on {embedder.device.upper()}. "
          f"The vector database now holds {store.count():,} chunks ({CHROMA_DIR}).")
    print(f"{too_long:,} chunks ({too_long / len(chunks):.1%}) are longer than the model's "
          f"{embedder.model.max_seq_length}-token limit; only their beginning is embedded.")


if __name__ == "__main__":
    main()
