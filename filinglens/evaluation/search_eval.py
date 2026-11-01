"""Step 5.2 - measure the search: does it find the passage that answers each question?

Every search setup gets each test question in eval/questions.jsonl and returns its
top 10 passages. For each, we note the rank of the first relevant passage
(1 = top of the list; empty = not in the top 10). Then:
  - Recall@5: the share of questions with a relevant passage in the top 5.
  - MRR (mean reciprocal rank): the average of 1 / rank (rank 1 -> 1, rank 2 -> 0.5,
    rank 4 -> 0.25, missed -> 0). It rewards putting the answer first, not just near the top.
The company and year filters come from the question, as in the app, so a filter the
question doesn't trigger counts against the search too.

Run:  python -m filinglens.evaluation.search_eval
Writes eval/search_results.csv: one row per question with each setup's rank, for failure analysis.
"""

import json
import sys
import time

import pandas as pd

from filinglens.assistant import SEARCH_MODES, Assistant
from filinglens.evaluation.draft_questions import EVAL_DIR, QUESTIONS_PATH
from filinglens.rag.filters import Filters
from filinglens.rag.passage import SearchResult

RESULTS_PATH = EVAL_DIR / "search_results.csv"
TOP = 10


def first_rank(results: list[SearchResult], relevant: list[str]) -> int | None:
    return next((rank for rank, r in enumerate(results, start=1) if r.chunk_id in relevant), None)


def summarize(ranks: pd.Series) -> dict:
    """ranks: one per question, NaN where the search missed."""
    return {"Recall@5": (ranks <= 5).mean(), "MRR": (1 / ranks).fillna(0).mean()}


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    questions = [json.loads(line) for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines() if line]
    rows, seconds = [], {mode: 0.0 for mode in SEARCH_MODES}
    with Assistant() as assistant:
        for mode in SEARCH_MODES:
            assistant.search("warm-up", Filters(), 1, mode)  # load every model first, so loading isn't timed
        for q in questions:
            filters = assistant.filters_for(q["question"])
            row = {"id": q["id"], "question": q["question"], "relevant": " ".join(q["relevant"]),
                   "filters": filters.describe(),
                   "filters_ok": filters.tickers == [q["ticker"]] and filters.years == [q["fiscal_year"]]}
            for mode in SEARCH_MODES:
                # Searched twice, timing the 2nd: the 1st search of a company-year reads its vectors from disk,
                # and whichever setup ran first would pay for that.
                assistant.search(q["question"], filters, TOP, mode)
                start = time.perf_counter()
                results = assistant.search(q["question"], filters, TOP, mode)
                seconds[mode] += time.perf_counter() - start
                row[mode] = first_rank(results, q["relevant"])
                row[f"{mode}_top"] = results[0].chunk_id if results else ""
            rows.append(row)

    table = pd.DataFrame(rows)
    table[list(SEARCH_MODES)] = table[list(SEARCH_MODES)].astype("Float64")
    table.to_csv(RESULTS_PATH, index=False)

    print(f"{len(table)} questions; the filters matched the question's company and year for "
          f"{table['filters_ok'].sum()}. Ranks for each question: {RESULTS_PATH}\n")
    summary = pd.DataFrame({mode: summarize(table[mode].astype(float)) for mode in SEARCH_MODES}).T
    summary["ms per question"] = [1000 * seconds[mode] / len(table) for mode in SEARCH_MODES]
    print(summary.to_string(float_format=lambda x: f"{x:.2f}"))

    in_top5 = table[list(SEARCH_MODES)].astype(float) <= 5
    print("\nCompared with vector search (questions answered in the top 5):")
    for mode in SEARCH_MODES[1:]:
        gained = (in_top5[mode] & ~in_top5["vector"]).sum()
        lost = (~in_top5[mode] & in_top5["vector"]).sum()
        print(f"  {mode:<8} found {gained} that vector missed, missed {lost} that vector found")


if __name__ == "__main__":
    main()
