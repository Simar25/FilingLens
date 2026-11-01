"""Step 5.5 - measure the answers: are the numbers right, is the route right, what does it cost?

Every question in eval/answer_questions.jsonl (40 number, 20 mixed) goes through the whole
app, like a user's question. So do 20 of the 50 text questions from Step 5.1; the other 30
only go through the router, because each full answer uses up free-tier tokens.

  - Numbers: the expected values come from a hand-written query in each test question, not
    from an LLM. A question counts as right only if EVERY expected value is in the answer.
    The same check on the SQL's result table shows which step went wrong: a bad query, or
    a good query reported wrongly.
  - Figures not in the table: the answer is told to copy figures, never calculate them.
  - Router: its choice against each question's label. A question sent to "both" still gets
    all the evidence it needs (it only costs more), so only the other mistakes do harm.
  - Cost and speed: tokens and seconds for every LLM call, by route.

Free tiers run out, so each finished question is saved at once and a re-run skips it.

Run:  python -m filinglens.evaluation.answer_eval             (asks what's missing, then the summary)
      python -m filinglens.evaluation.answer_eval --summary   (no LLM calls)
Writes eval/answer_results.jsonl: one row per question, for failure analysis. Then run
filinglens.evaluation.faithfulness to grade the cited sentences.
"""

import argparse
import json
import re
import sys
import time

import pandas as pd

from filinglens.assistant import Assistant
from filinglens.evaluation.draft_questions import EVAL_DIR, QUESTIONS_PATH
from filinglens.llm import LLMError, current_settings
from filinglens.rag.answer import CITATION
from filinglens.router import route_question

ANSWER_QUESTIONS_PATH = EVAL_DIR / "answer_questions.jsonl"
RESULTS_PATH = EVAL_DIR / "answer_results.jsonl"
FULL_TEXT_ANSWERS = 20         # text questions answered in full; the rest only go through the router
TOKENS_PER_MINUTE = 8000       # Groq's free tier: the pause after each question stays under this
PRICE_PER_MILLION = 0.15, 0.60  # USD, input and output: Groq's list price for gpt-oss-120b on 2026-10-31

# 394.33, 1,234, 12.5%, $3 billion. Not the 2024 in FY2024, nor the 10 in 10-K.
NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?!-[A-Za-z])\s*(%|percent|trillion|billion|bn|million)?",
                    re.IGNORECASE)
SCALES = {"trillion": 1e12, "billion": 1e9, "bn": 1e9, "million": 1e6, "%": 1, "percent": 1}
ANY_SCALE = (1, 1e3, 1e6, 1e9)  # a bare number, e.g. in a table under the heading "USD bn", can be in any unit
MAX_ROUNDING = 0.01             # "$9 billion" may stand for 8.97, but a stray "1" must not stand for 1.2


def find_numbers(text: str) -> list[tuple[float, int, str]]:
    """Every number in the text, without [n] citations: (value, decimals shown, unit word)."""
    return [(float(digits.replace(",", "") + fraction), len(fraction) - 1 if fraction else 0, unit.lower())
            for digits, fraction, unit in NUMBER.findall(CITATION.sub(" ", text))]


def states(gold: float, numbers: list[tuple[float, int, str]]) -> bool:
    """Does one of the numbers show this value, rounded to as many decimals as that number has?
    shortcut: the sign is ignored ("loss of $2 billion" has no minus), so "rose 3%" passes for -3%; read the failures."""
    gold = abs(gold)
    for value, decimals, unit in numbers:
        for scale in (SCALES[unit],) if unit else ANY_SCALE:
            shown = gold / scale
            if abs(shown - value) <= 0.5 * 10 ** -decimals + 1e-9 and abs(shown - value) <= MAX_ROUNDING * shown:
                return True
    return False


def has_everything(text: str, gold: list[float], names: list[list[str]]) -> bool:
    """All expected values and, for "which company" questions, one spelling of each expected name."""
    numbers = find_numbers(text)
    return (all(states(value, numbers) for value in gold)
            and all(any(name.lower() in text.lower() for name in spellings) for spellings in names))


def figures_not_in_table(answer: str, table_csv: str) -> list[float]:
    """Figures in the answer that aren't in the table (rounded or not). Years don't count."""
    table = [value for value, _, _ in find_numbers(table_csv)]
    return [value for value, decimals, _ in find_numbers(answer)
            if not (decimals == 0 and 1900 <= value <= 2100)
            and not any(states(cell, [(value, decimals, "")]) for cell in table)]


def gold_values(con, q: dict) -> list[float]:
    values = [value for row in con.execute(q["gold_sql"]).fetchall() for value in row]
    if not values or any(not isinstance(value, (int, float)) for value in values):
        raise ValueError(f"{q['id']}: the expected values must be numbers, got {values}")
    return values


def score(row: dict, names: list[list[str]]) -> dict:
    """Done when results are read, not when they're asked, so fixing the scoring costs no LLM tokens."""
    table = row["table"].replace(",", " ")  # a CSV comma isn't a thousands separator: 2024-12-31,248.62 is not 31,248.62
    return row | {"sql_right": has_everything(table, row["gold"], []),  # tables have tickers, not names
                  "answer_right": has_everything(row["text"], row["gold"], names),
                  "extra_figures": figures_not_in_table(row["text"], table) if table else []}


def price(calls: list[dict]) -> float:
    return sum(c["in"] * PRICE_PER_MILLION[0] + c["out"] * PRICE_PER_MILLION[1] for c in calls) / 1e6


def answer(assistant: Assistant, q: dict) -> dict:
    """Ask one test question and keep what the scoring and the failure analysis need."""
    response = assistant.ask(q["question"])
    sql = response.sql
    table = sql.table.to_csv(index=False, lineterminator="\n") if sql and sql.ok else ""
    row = {"answered": True, "route": response.route.route, "text": response.text, "sql": sql.sql if sql else None,
           "table": table, "sql_error": sql.error if sql else None, "cannot_answer": sql.cannot_answer if sql else None,
           "sql_attempts": sql.attempts if sql else 0, "sources": [s.chunk_id for s in response.sources],
           "cited": response.cited, "invalid_citations": response.invalid_citations, "cites_db": response.cites_db,
           "search_seconds": response.search_seconds,
           "calls": [{"stage": stage, "in": r.input_tokens, "out": r.output_tokens, "seconds": r.seconds}
                     for stage, r in response.calls]}
    if "gold_sql" in q:
        row["gold"] = gold_values(assistant.con, q)
    return row


def route_only(assistant: Assistant, q: dict) -> dict:
    decision = route_question(q["question"], assistant.generate)
    r = decision.reply
    return {"answered": False, "route": decision.route,
            "calls": [{"stage": "router", "in": r.input_tokens, "out": r.output_tokens, "seconds": r.seconds}]}


def load_results() -> list[dict]:
    if not RESULTS_PATH.exists():
        return []
    return [json.loads(line) for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines() if line]


def run(questions: list[dict]) -> None:
    done = {row["id"] for row in load_results()}
    todo = [q for q in questions if q["id"] not in done]
    if not todo:
        return
    print(f"{len(done)} questions already done, {len(todo)} to go, with {current_settings()[1]}")
    with Assistant() as assistant, RESULTS_PATH.open("a", encoding="utf-8") as results:
        for q in todo:
            start = time.perf_counter()
            try:
                row = route_only(assistant, q) if q.get("route_only") else answer(assistant, q)
            except LLMError as exc:
                print(f"\nStopped at {q['id']}: {exc}\nEverything finished so far is saved. Run again to continue.")
                return
            row = {"id": q["id"], "type": q["type"], "category": q.get("category", "text"),
                   "question": q["question"], "model": current_settings()[1]} | row
            results.write(json.dumps(row) + "\n")
            results.flush()
            right = score(row, q.get("names", []))["answer_right"] if "gold" in row else ""
            print(f"{q['id']} {row['route']:<8} {right} {q['question'][:70]}", flush=True)
            tokens = sum(c["in"] + c["out"] for c in row["calls"])
            time.sleep(max(0.0, tokens / TOKENS_PER_MINUTE * 60 - (time.perf_counter() - start)))


def summarize(rows: list[dict]) -> None:
    table = pd.DataFrame(rows)
    print(f"\n{len(table)} questions\n\nRouter (rows: the right route; columns: its choice)")
    print(pd.crosstab(table["type"], table["route"]).to_string())
    print(f"Right: {(table['route'] == table['type']).sum()} of {len(table)}. Harmful mistakes (not sent to 'both'): "
          f"{((table['route'] != table['type']) & (table['route'] != 'both')).sum()}")

    scored = table[table["answer_right"].notna()]
    if len(scored):
        print("\nNumbers: every expected value in...")
        print(scored.groupby("category")[["sql_right", "answer_right"]].agg(["sum", "count"]).to_string())
        for kind, part in scored.groupby("type"):
            print(f"  {kind}: result table {int(part['sql_right'].sum())}/{len(part)}, "
                  f"answer {int(part['answer_right'].sum())}/{len(part)}")
        # Number questions only: a mixed answer may quote figures from the reports, which faithfulness checks.
        numbers = scored[scored["type"] == "numbers"]
        flagged = numbers[numbers["extra_figures"].map(len) > 0]
        print(f"Number answers with a figure that isn't in the table: {len(flagged)} of {len(numbers)}: "
              f"{' '.join(flagged['id'])}")

    answered = table[table["answered"]].copy()
    if len(answered):
        answered["tokens"] = answered["calls"].map(lambda calls: sum(c["in"] + c["out"] for c in calls))
        answered["llm_calls"] = answered["calls"].map(len)
        answered["usd"] = answered["calls"].map(price)
        answered["seconds"] = answered["calls"].map(lambda calls: sum(c["seconds"] for c in calls)) \
            + answered["search_seconds"]
        print("\nCost and speed per question, by the route taken (median seconds: waits for free-tier limits inflate means)")
        print(answered.groupby("route").agg(questions=("id", "size"), llm_calls=("llm_calls", "mean"),
                                            tokens=("tokens", "mean"), usd=("usd", "mean"),
                                            seconds=("seconds", "median")).to_string(float_format=lambda x: f"{x:.5g}"))
        print(f"All answered questions: ${answered['usd'].mean():.5f} each, ${answered['usd'].mean() * 1000:.2f} per 1,000")
        print(f"Invalid citations in {(answered['invalid_citations'].map(len) > 0).sum()} of {len(answered)} answers")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Measure the app's answers on the test questions.")
    parser.add_argument("--summary", action="store_true", help="Only summarize the saved results")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")
    numbers = [json.loads(line) for line in ANSWER_QUESTIONS_PATH.read_text(encoding="utf-8").splitlines() if line]
    text = [json.loads(line) for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines() if line]
    for q in text[FULL_TEXT_ANSWERS:]:
        q["route_only"] = True
    if not args.summary:
        run(numbers + text)
    names = {q["id"]: q.get("names", []) for q in numbers}
    rows = [score(row, names[row["id"]]) if "gold" in row else row for row in load_results()]
    if rows:
        summarize(rows)


if __name__ == "__main__":
    main()
