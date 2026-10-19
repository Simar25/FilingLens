r"""Ask a question about the companies: their numbers, their annual reports, or both.

Usage (double quotes around the question are safest; without them, a word like Apple's breaks
the command in PowerShell, because ' starts a quoted text there):

    .\.venv\Scripts\python.exe -m filinglens.ask "Compare NVIDIA's and AMD's gross margins since 2021"
    .\.venv\Scripts\python.exe -m filinglens.ask "What supply chain risks did Apple describe in 2024?"
    .\.venv\Scripts\python.exe -m filinglens.ask "How did Apple's R&D spending change, and why?"

The question is routed automatically (Step 3): numbers go to SQL over the
database, report questions to the RAG search (Step 2), and mixed ones to both.

Options:
    --route numbers|text|both   skip the router and use this path
    --ticker AAPL MSFT    only search these companies' reports (default: detected from the question)
    --year 2023 2024      only search these fiscal years' reports (default: detected from the question)
    --section 1A 7        only search these Items, e.g. 1A = Risk Factors, 7 = MD&A
    -k 8                  how many report passages to give the LLM
    --search-only         show the report passages the search finds, without asking the LLM
    --show-text           print the full text of every source
    --no-chart            don't save or open a chart
"""

import argparse
import re
import sys
import textwrap
import time
import webbrowser
from datetime import datetime

import pandas as pd

from filinglens import llm
from filinglens.assistant import Assistant, Response, SetupError
from filinglens.charts import save_chart
from filinglens.combine import DB_CITATION
from filinglens.config import PROJECT_ROOT
from filinglens.rag.vector_store import SearchResult
from filinglens.router import ROUTES

PREVIEW_CHARS = 220
TABLE_ROWS_SHOWN = 25
CHART_DIR = PROJECT_ROOT / "data" / "charts"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question about the companies' numbers and 10-K reports.")
    # nargs="+": without quotes, the terminal passes each word separately; accept them all and join them.
    parser.add_argument("question", nargs="+")
    parser.add_argument("--route", choices=ROUTES, help="Skip the router and use this path")
    parser.add_argument("--ticker", nargs="+", help="Only these companies' reports")
    parser.add_argument("--year", nargs="+", type=int, help="Only these fiscal years' reports")
    parser.add_argument("--section", nargs="+", help="Only these Items, e.g. 1A 7")
    parser.add_argument("-k", type=int, default=8, help="Number of report passages to retrieve (default 8)")
    parser.add_argument("--search-only", action="store_true", help="Show the search results; don't call the LLM")
    parser.add_argument("--show-text", action="store_true", help="Print the full text of each source")
    parser.add_argument("--no-chart", action="store_true", help="Don't save or open a chart")
    args = parser.parse_args(argv)
    args.question = " ".join(args.question)
    return args


def print_source(number: int, source: SearchResult, full_text: bool) -> None:
    print(f"  [{number}] {source.label}  (similarity {source.similarity:.2f}, id {source.chunk_id})")
    if full_text:
        body = source.text
    else:
        body = textwrap.fill(textwrap.shorten(source.text, PREVIEW_CHARS, placeholder=" ..."), 100)
    print(textwrap.indent(body, "      "))


def print_sql(response: Response) -> None:
    sql = response.sql
    if sql.cannot_answer:
        print(f"\nSQL: the LLM says the database can't answer this ({sql.cannot_answer})")
        return
    if sql.errors and sql.ok:
        print(f"\nThe first query failed, so the error went back to the LLM for a retry:\n  {sql.errors[0]}")
    print("\nSQL (written by the LLM, checked before it ran):" if sql.ok else "\nSQL that failed:")
    print(textwrap.indent(sql.sql or "(none)", "    "))
    if not sql.ok:
        return
    table = sql.table
    more = " (more were cut off by the row limit)" if sql.truncated else ""
    print(f"\nResult: {len(table)} rows{more}")
    if not table.empty:
        with pd.option_context("display.max_columns", 30, "display.width", 200):
            print(textwrap.indent(table.head(TABLE_ROWS_SHOWN).to_string(index=False), "    "))
        if len(table) > TABLE_ROWS_SHOWN:
            print(f"    ... {len(table) - TABLE_ROWS_SHOWN} more rows")


def print_response(response: Response, full_text: bool) -> None:
    print("\n" + response.text + "\n")
    if response.cites_db:
        print(f"{DB_CITATION} = the database result above.")
    if response.invalid_citations:
        print(f"Warning: the answer cites sources that don't exist: {response.invalid_citations}")
    if response.sources:
        print("Sources cited:" if response.cited else "The answer cites no report passages.")
        for number in response.cited:
            print_source(number, response.sources[number - 1], full_text)
        unused = [n for n in range(1, len(response.sources) + 1) if n not in response.cited]
        if unused:
            print(f"Also retrieved but not cited: {', '.join(f'[{n}]' for n in unused)}")
    if response.calls:
        replies = [r for _, r in response.calls]
        n_in, n_out = sum(r.input_tokens for r in replies), sum(r.output_tokens for r in replies)
        stages = ", ".join(f"{stage} {r.seconds:.1f} s" for stage, r in response.calls)
        print(f"\n{len(replies)} LLM call{'s' * (len(replies) > 1)} to {replies[-1].model} "
              f"({n_in:,} tokens in, {n_out:,} out): {stages}")
    if response.sources:
        print(f"Report search: {response.search_seconds:.1f} s (includes loading the embedding model)")


def show_chart(response: Response) -> None:
    slug = re.sub(r"[^a-z0-9]+", "-", response.question.lower()).strip("-")[:50]
    path = save_chart(response.chart, CHART_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.html")
    print(f"\nChart: {path.relative_to(PROJECT_ROOT)} (opening it in your browser)")
    webbrowser.open(path.as_uri())


def search_only(assistant: Assistant, args: argparse.Namespace) -> None:
    filters = assistant.filters_for(args.question, args.ticker, args.year, args.section)
    print(f"Searching: {filters.describe()}")
    start = time.perf_counter()
    sources = assistant.search(args.question, filters, args.k)
    print(f"Found {len(sources)} passages in {time.perf_counter() - start:.1f} s")
    for number, source in enumerate(sources, start=1):
        print_source(number, source, args.show_text)


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(errors="replace")  # reports contain characters some Windows terminals can't show
    args = parse_args(argv)
    with Assistant() as assistant:
        try:
            if args.search_only:
                search_only(assistant, args)
                return
            start = time.perf_counter()
            response = assistant.ask(args.question, route=args.route, tickers=args.ticker, years=args.year,
                                     items=args.section, k=args.k)
        except (llm.LLMError, SetupError) as exc:
            sys.exit(f"\n{exc}")

    route = response.route
    print(f"Route: {route.route}" + (f" ({route.reason})" if route.reason else ""))
    if response.filters:
        print(f"Report search: {response.filters.describe()}, {len(response.sources)} passages found")
    if response.sql:
        print_sql(response)
    print_response(response, args.show_text)
    if response.chart is not None and not args.no_chart:
        show_chart(response)
    print(f"Total time: {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
