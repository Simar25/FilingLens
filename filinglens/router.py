"""Step 3.4 - decide which path a question needs: numbers, text, or both.

  numbers  "What was NVIDIA's gross margin in 2024?"        -> text-to-SQL
  text     "What supply chain risks did Apple describe?"    -> RAG over the reports
  both     "How did Apple's R&D spending change, and why?"  -> SQL for the numbers,
                                                               RAG for the explanation

A short LLM call makes the decision. Keyword rules ("how much" -> numbers)
break on questions like "how much does Apple rely on China?", which is a text
question. The LLM is told which numbers the database holds, so "market share"
or "number of employees" (not in the database) go to the reports.

When the reply can't be read, the router picks "both": it costs one more
search, but whichever kind of evidence the question needed is there.
The `ask` command prints the route it took, and --route overrides it.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from filinglens.llm import LLMReply

ROUTES = ("numbers", "text", "both")
FALLBACK_ROUTE = "both"

SYSTEM_PROMPT = """\
You route questions about US public companies to the right source of information.

- "numbers": the answer is figures from a database of reported financial statements and stock prices. It has, per \
company and fiscal year: revenue, cost of revenue, gross profit, R&D expense, operating income, net income, diluted \
EPS, operating cash flow, capital expenditure, free cash flow, total assets, total liabilities, stockholders' equity, \
cash, dividends paid, and margins and growth rates computed from these. Plus daily stock prices (and returns) for \
each company and for the S&P 500.
- "text": the answer is in the wording of the companies' annual reports (10-K): risks, strategy, products, \
customers, competition, supply chain, regulation, lawsuits, employees, management's explanations, and any figure \
that is not in the database above (e.g. market share, number of employees, segment details).
- "both": the question needs figures from the database AND an explanation or context from the reports, e.g. \
"How did X change, and why?"

Reply with JSON only: {"route": "numbers" | "text" | "both", "reason": "<a few words>"}"""

JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)
BARE_ROUTE = re.compile(r"\b(numbers|text|both)\b", re.IGNORECASE)


@dataclass
class RouteDecision:
    route: str            # "numbers", "text" or "both"
    reason: str
    reply: LLMReply | None = None  # None when the route was given with --route


def parse_route(text: str) -> tuple[str, str]:
    """(route, reason) from the LLM's reply. Falls back to "both" if it's unreadable."""
    match = JSON_OBJECT.search(text)
    if match:
        try:
            data = json.loads(match.group(0))
            route = str(data.get("route", "")).strip().lower()
            if route in ROUTES:
                return route, str(data.get("reason", "")).strip()
        except (json.JSONDecodeError, AttributeError):
            pass
    found = {m.lower() for m in BARE_ROUTE.findall(text)}
    if len(found) == 1:  # exactly one route word, e.g. the model replied just "numbers"
        return found.pop(), ""
    return FALLBACK_ROUTE, "the router's reply was unclear, so both paths are used"


def route_question(question: str, generate: Callable[[str, str], LLMReply]) -> RouteDecision:
    """`generate` is llm.generate (or a fake one in tests)."""
    reply = generate(SYSTEM_PROMPT, f"Question: {question}")
    route, reason = parse_route(reply.text)
    return RouteDecision(route, reason, reply)
