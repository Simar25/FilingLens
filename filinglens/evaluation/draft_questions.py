"""Step 5.1 - draft test questions for the search, from real report passages.

To measure the search we need questions whose right answer we already know.
Writing 40 from scratch is slow, so an LLM drafts them: pick a passage, ask the
LLM for a question that this passage answers, and that passage becomes the
right answer ("relevant" chunk). You then check every draft by hand, reading
eval/questions_review.md, before any number goes in the README.

Known weak spots of drafting questions this way, and what is done about them:
  - An LLM tends to reuse the passage's words, which flatters keyword search.
    It is told to ask in its own words, the way an investor would.
  - Passages with nothing specific in them make vague questions. The LLM skips them.
  - Another passage may answer the question too, and a search that finds it is
    marked wrong. The failure analysis looks for these and adds them to "relevant".

Run:  python -m filinglens.evaluation.draft_questions --n 50
      python -m filinglens.evaluation.draft_questions --review-only   (rewrite the review file after edits)
"""

import argparse
import json
import re
import sys

import duckdb
import pandas as pd

from filinglens import llm
from filinglens.config import DB_PATH, PROJECT_ROOT
from filinglens.rag.embeddings import passage_text

EVAL_DIR = PROJECT_ROOT / "eval"
QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"
REVIEW_PATH = EVAL_DIR / "questions_review.md"
SEED = 5
SECTIONS = ("1", "1A", "7")  # Business, Risk Factors, MD&A: the parts of a 10-K people ask about
MIN_WORDS = 150              # shorter chunks are mostly headings, tables or one-line notices

SYSTEM_PROMPT = """\
You write test questions for a search engine over US companies' annual reports (Form 10-K). You get one passage \
from a report. Write one question that this passage answers, the way an investor or analyst would ask it.

- Name the company by its everyday name (e.g. "Apple", not "APPLE INC") and the fiscal year (e.g. "in fiscal 2023").
- Ask about something specific in this passage: a reason, an event, a program, a figure, a named risk. Not a broad \
topic the rest of the report also covers, like "What risks does Apple face?".
- Use your own words. Don't copy phrases from the passage, except names (of products, laws, programs, places).
- The answer must be in this passage.
- If the passage has nothing specific to ask about (boilerplate, a list of exhibits, a table without explanation, \
a forward-looking-statements notice), reply with only: SKIP

Otherwise reply with JSON only: {"question": "...", "answer": "a one-sentence answer, taken from the passage"}"""


def candidate_passages(con: duckdb.DuckDBPyConnection, seed: int = SEED) -> pd.DataFrame:
    """One random Business, Risk Factors or MD&A passage per company, companies in random order."""
    chunks = con.execute(f"""
        SELECT c.chunk_id, c.ticker, co.name AS company, c.fiscal_year, c.item, c.title, c.text
        FROM chunks c JOIN companies co USING (ticker)
        WHERE c.item IN {SECTIONS} AND c.n_words >= {MIN_WORDS}
        ORDER BY c.chunk_id
    """).df()
    return chunks.sample(frac=1, random_state=seed).drop_duplicates("ticker").reset_index(drop=True)


def parse_draft(text: str) -> dict | None:
    """The LLM's reply -> {"question", "answer"}, or None for SKIP or a reply that isn't usable."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        draft = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return draft if isinstance(draft, dict) and draft.get("question") and draft.get("answer") else None


def review_entry(question: dict, passage: str) -> str:
    edit = f"*Edited after review ({question['edit']}). LLM draft: {question['draft']}*\n\n" if "edit" in question else ""
    also = f"*Also relevant: {question['relevant_note']}*\n\n" if "relevant_note" in question else ""
    return (f"### {question['id']}: {question['ticker']} FY{question['fiscal_year']}, Item {question['item']} "
            f"(`{question['relevant'][0]}`)\n\n**Q:** {question['question']}\n\n{edit}**A:** {question['answer']}\n\n"
            + also + "\n".join(f"> {line}" for line in passage.splitlines()) + "\n")


def write_review(con: duckdb.DuckDBPyConnection, questions: list[dict]) -> None:
    """Each question with its answer and the passage it was drafted from, for checking by hand."""
    entries = []
    for q in questions:
        company, title, text = con.execute(
            "SELECT co.name, c.title, c.text FROM chunks c JOIN companies co USING (ticker) WHERE c.chunk_id = ?",
            [q["relevant"][0]]).fetchone()
        entries.append(review_entry(q, passage_text(company, q["ticker"], q["fiscal_year"], title, text)))
    REVIEW_PATH.write_text("# Test questions to check by hand\n\nFor each one: does the passage answer the question, "
                           "and is the answer right? Note anything to fix or drop by its id (t01...).\n\n"
                           + "\n".join(entries), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Draft test questions from report passages with the LLM.")
    parser.add_argument("--n", type=int, default=50, help="How many questions (default 50)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing questions file")
    parser.add_argument("--review-only", action="store_true", help="Only rewrite the review file from the questions")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        if args.review_only:
            write_review(con, [json.loads(line) for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
                               if line])
            print(f"Review file rewritten: {REVIEW_PATH}")
        elif QUESTIONS_PATH.exists() and not args.overwrite:
            sys.exit(f"{QUESTIONS_PATH} exists and may hold hand-checked questions. Use --overwrite to replace it.")
        else:
            draft_questions(con, args.n)
    finally:
        con.close()


def draft_questions(con: duckdb.DuckDBPyConnection, n: int) -> None:
    questions = []
    for row in candidate_passages(con).itertuples():
        if len(questions) == n:
            break
        text = passage_text(row.company, row.ticker, row.fiscal_year, row.title, row.text)
        reply = llm.generate(SYSTEM_PROMPT, f"<passage>\n{text}\n</passage>")
        draft = parse_draft(reply.text)
        print(f"{row.chunk_id:<22} {'SKIP' if draft is None else draft['question']}")
        if draft is None:
            continue
        question = {"id": f"t{len(questions) + 1:02d}", "type": "text", "question": draft["question"],
                    "answer": draft["answer"], "ticker": row.ticker, "fiscal_year": row.fiscal_year,
                    "item": row.item, "relevant": [row.chunk_id], "source": f"drafted by {reply.model}",
                    "checked": False}
        questions.append(question)

    EVAL_DIR.mkdir(exist_ok=True)
    QUESTIONS_PATH.write_text("".join(json.dumps(q) + "\n" for q in questions), encoding="utf-8")
    write_review(con, questions)
    print(f"\n{len(questions)} questions saved to {QUESTIONS_PATH}; read them in {REVIEW_PATH}")


if __name__ == "__main__":
    main()
