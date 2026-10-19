r"""Step 2.7 - ask a question about the annual reports and get a cited answer.

Usage (put the question in double quotes):

    .\.venv\Scripts\python.exe -m filinglens.ask "What supply chain risks did Apple describe in 2024?"

Options:
    --ticker AAPL MSFT    only search these companies (default: detected from the question)
    --year 2023 2024      only search these fiscal years (default: detected from the question)
    --section 1A 7        only search these Items, e.g. 1A = Risk Factors, 7 = MD&A
    -k 8                  how many passages to give the LLM
    --search-only         show the passages the search finds, without asking the LLM
    --show-text           print the full text of every source
"""

import argparse
import sys
import textwrap
import time

import duckdb

from filinglens import llm
from filinglens.config import DB_PATH
from filinglens.rag.answer import Answer, answer_question
from filinglens.rag.embeddings import Embedder
from filinglens.rag.filters import Filters, detect_filters
from filinglens.rag.vector_store import SearchResult, VectorStore

PREVIEW_CHARS = 220


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question about the 10-K reports.")
    parser.add_argument("question")
    parser.add_argument("--ticker", nargs="+", help="Only these companies")
    parser.add_argument("--year", nargs="+", type=int, help="Only these fiscal years")
    parser.add_argument("--section", nargs="+", help="Only these Items, e.g. 1A 7")
    parser.add_argument("-k", type=int, default=8, help="Number of passages to retrieve (default 8)")
    parser.add_argument("--search-only", action="store_true", help="Show the search results; don't call the LLM")
    parser.add_argument("--show-text", action="store_true", help="Print the full text of each source")
    return parser.parse_args(argv)


def choose_filters(args: argparse.Namespace) -> Filters:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        companies = dict(con.execute("SELECT ticker, name FROM companies").fetchall())
        years = [row[0] for row in con.execute("SELECT DISTINCT fiscal_year FROM chunks ORDER BY 1").fetchall()]
    finally:
        con.close()
    filters = detect_filters(args.question, companies, years)
    if args.ticker:
        filters.tickers = [t.upper() for t in args.ticker]
    if args.year:
        filters.years = args.year
    if args.section:
        filters.items = [s.upper() for s in args.section]
    return filters


def print_source(number: int, source: SearchResult, full_text: bool) -> None:
    print(f"  [{number}] {source.label}  (similarity {source.similarity:.2f}, id {source.chunk_id})")
    if full_text:
        body = source.text
    else:
        body = textwrap.fill(textwrap.shorten(source.text, PREVIEW_CHARS, placeholder=" ..."), 100)
    print(textwrap.indent(body, "      "))


def print_answer(answer: Answer, full_text: bool) -> None:
    print("\n" + answer.text + "\n")
    if answer.invalid_citations:
        print(f"Warning: the answer cites sources that don't exist: {answer.invalid_citations}\n")
    if answer.sources:
        print("Sources cited:" if answer.cited else "The answer cites no sources.")
        for number in answer.cited:
            print_source(number, answer.sources[number - 1], full_text)
        unused = [n for n in range(1, len(answer.sources) + 1) if n not in answer.cited]
        if unused:
            print(f"Also retrieved but not cited: {', '.join(f'[{n}]' for n in unused)}")
    if answer.reply:
        r = answer.reply
        print(f"\n({r.model}: {r.input_tokens:,} tokens in, {r.output_tokens:,} out, {r.seconds:.1f} s)")


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(errors="replace")  # reports contain characters some Windows terminals can't show
    args = parse_args(argv)
    filters = choose_filters(args)
    print(f"Searching: {filters.describe()}")

    start = time.perf_counter()
    store = VectorStore()
    if store.count() == 0:
        sys.exit("The vector database is empty. Run  python -m filinglens.build_index  first.")
    embedder = Embedder()
    sources = store.search(embedder.embed_query(args.question), k=args.k, filters=filters)
    print(f"Found {len(sources)} passages in {time.perf_counter() - start:.1f} s")

    if args.search_only:
        for number, source in enumerate(sources, start=1):
            print_source(number, source, args.show_text)
        return
    try:
        answer = answer_question(args.question, sources, filters, llm.generate)
    except llm.LLMError as exc:
        sys.exit(f"\n{exc}")
    print_answer(answer, args.show_text)


if __name__ == "__main__":
    main()
