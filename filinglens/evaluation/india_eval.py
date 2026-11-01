"""Step 6b - can the Indian reports be searched, and answered from?

eval/india_questions.jsonl has one question per company, each written from a randomly chosen
passage, so its page and its answer are known. A search counts if the page is in the top 5.

Run:  python -m filinglens.evaluation.india_eval             (search only, no LLM calls)
      python -m filinglens.evaluation.india_eval --answers   (full answers from the online version)
Writes eval/india_results.jsonl when asked for answers.
"""

import argparse
import json
import sys

from filinglens.assistant import Assistant
from filinglens.evaluation.draft_questions import EVAL_DIR
from filinglens.online import OnlineAssistant

QUESTIONS_PATH = EVAL_DIR / "india_questions.jsonl"
RESULTS_PATH = EVAL_DIR / "india_results.jsonl"


def right_page(found: list, q: dict) -> int | None:
    """Rank of the first result on the question's page, from 1."""
    wanted = (q["ticker"], q["fiscal_year"], f"page {q['page']}")
    return next((rank for rank, p in enumerate(found, start=1) if (p.ticker, p.fiscal_year, p.title) == wanted), None)


def has_answer(text: str, q: dict) -> bool:
    return any(phrase in text.lower() for phrase in q["expect"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Measure search and answers on the Indian reports.")
    parser.add_argument("--answers", action="store_true", help="Ask the online version for full answers instead of measuring search")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")
    questions = [json.loads(line) for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines() if line]

    for name, assistant_type in () if args.answers else (("full, hybrid", Assistant), ("online, keyword", OnlineAssistant)):
        with assistant_type() as assistant:
            ranks = [right_page(assistant.search(q["question"], assistant.filters_for(q["question"]), 8), q)
                     for q in questions]
        print(f"{name}: right page in the top 5 for {sum(r is not None and r <= 5 for r in ranks)} of {len(ranks)}. "
              f"Ranks: {ranks}")

    if args.answers:
        rows = []
        with OnlineAssistant() as assistant:
            for q in questions:
                response = assistant.ask(q["question"])
                rows.append({"id": q["id"], "question": q["question"], "route": response.route.route, "text": response.text,
                             "right": has_answer(response.text, q), "cited": response.cited,
                             "sources": [s.label for s in response.sources]})
                print(f"{q['id']} {'right' if rows[-1]['right'] else 'WRONG'} {response.text[:160]!r}", flush=True)
        RESULTS_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"Answers with the expected fact: {sum(r['right'] for r in rows)} of {len(rows)}")


if __name__ == "__main__":
    main()
