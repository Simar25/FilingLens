"""Step 3 - one entry point for every question: route it, look things up, answer, chart.

    question -> router --+-- numbers --> text-to-SQL --------------+
                         +-- text -----> RAG search ----------------+--> answer (+ chart)
                         +-- both -----> text-to-SQL + RAG search --+

The `ask` command uses it now, and the web app will in Step 6. An Assistant
keeps the database connection open and, once loaded, the embedding model. A
web app that keeps one Assistant around answers each question without
reloading them. The embedding model (PyTorch, ~10 s to load) is only loaded
for the first question that needs the reports, so number questions skip it.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import plotly.graph_objects as go

from filinglens import llm
from filinglens.charts import make_chart
from filinglens.combine import DB_CITATION, write_answer
from filinglens.config import DB_PATH
from filinglens.llm import LLMReply
from filinglens.rag.answer import answer_question, find_citations
from filinglens.rag.filters import Filters, detect_filters
from filinglens.rag.vector_store import SearchResult
from filinglens.router import ROUTES, RouteDecision, route_question
from filinglens.sql.safety import connect_readonly
from filinglens.sql.schema import describe_schema
from filinglens.sql.text_to_sql import SqlResult, has_data, text_to_sql

NO_DATA_ANSWER = "The query ran, but the database has no rows matching this question."


class SetupError(Exception):
    """A step that has to be run first (shown to the user as-is)."""


@dataclass
class Response:
    question: str
    route: RouteDecision
    text: str = ""
    sql: SqlResult | None = None              # numbers and both
    filters: Filters | None = None            # text and both
    sources: list[SearchResult] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)              # passage numbers used in the answer
    invalid_citations: list[int] = field(default_factory=list)  # [n] with no passage n
    cites_db: bool = False                    # the answer cites the database result as [DB]
    chart: go.Figure | None = None
    calls: list[tuple[str, LLMReply]] = field(default_factory=list)  # every LLM call: (stage, reply), in order
    search_seconds: float = 0.0               # report search, including loading the model the first time


class Assistant:
    def __init__(self, db_path=DB_PATH, generate: Callable[[str, str], LLMReply] = llm.generate):
        """`generate` is llm.generate (or a fake one in tests)."""
        self.con = connect_readonly(db_path)
        self.generate = generate
        self.companies = dict(self.con.execute("SELECT ticker, name FROM companies").fetchall())
        self.report_years = [y for (y,) in self.con.execute("SELECT DISTINCT fiscal_year FROM chunks ORDER BY 1")
                             .fetchall()]
        self.schema = describe_schema(self.con)
        self._embedder = None
        self._store = None

    def close(self) -> None:
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    # ---------- the pieces ----------

    def filters_for(self, question: str, tickers=None, years=None, items=None) -> Filters:
        """Companies and years detected in the question; anything given explicitly wins."""
        filters = detect_filters(question, self.companies, self.report_years)
        if tickers:
            filters.tickers = [t.upper() for t in tickers]
        if years:
            filters.years = list(years)
        if items:
            filters.items = [i.upper() for i in items]
        return filters

    def search(self, question: str, filters: Filters, k: int = 8) -> list[SearchResult]:
        if self._store is None:
            # Imported here: loading PyTorch takes seconds, and number questions don't need it.
            from filinglens.rag.embeddings import Embedder
            from filinglens.rag.vector_store import VectorStore

            self._store = VectorStore()
            if self._store.count() == 0:
                raise SetupError("The vector database is empty. Run  python -m filinglens.build_index  first.")
            self._embedder = Embedder()
        return self._store.search(self._embedder.embed_query(question), k=k, filters=filters)

    # ---------- the whole flow ----------

    def ask(self, question: str, route: str | None = None, tickers=None, years=None, items=None,
            k: int = 8) -> Response:
        """Answer a question. `route` ("numbers", "text" or "both") skips the router."""
        if route:
            if route not in ROUTES:
                raise ValueError(f"route must be one of {ROUTES}, not {route!r}")
            decision = RouteDecision(route, "chosen by you (--route)")
        else:
            decision = route_question(question, self.generate)
        response = Response(question, decision, calls=[("router", decision.reply)] if decision.reply else [])

        if decision.route in ("numbers", "both"):
            response.sql = text_to_sql(question, self.schema, self.con, self.generate)
            response.calls += [("SQL" if i == 0 else "SQL retry", r) for i, r in enumerate(response.sql.replies)]
        if decision.route in ("text", "both"):
            response.filters = self.filters_for(question, tickers, years, items)
            start = time.perf_counter()
            response.sources = self.search(question, response.filters, k)
            response.search_seconds = time.perf_counter() - start

        if decision.route == "text":
            answer = answer_question(question, response.sources, response.filters, self.generate)
            response.text, response.cited, response.invalid_citations = (
                answer.text, answer.cited, answer.invalid_citations)
            if answer.reply:
                response.calls.append(("answer", answer.reply))
        else:
            self._answer_with_numbers(response)

        if response.sql and response.sql.ok:
            response.chart = make_chart(response.sql.table, question)
        return response

    def _answer_with_numbers(self, response: Response) -> None:
        sql = response.sql
        has_rows = sql.ok and has_data(sql.table)
        if not has_rows and not response.sources:
            # Nothing to answer from, so no LLM call: say plainly what happened.
            if sql.cannot_answer:
                response.text = f"The database can't answer this: {sql.cannot_answer}"
            elif sql.error:
                response.text = f"No working SQL query after {sql.attempts} tries. Last error: {sql.error}"
            else:
                response.text = NO_DATA_ANSWER
            return
        reply = write_answer(response.question, sql, response.sources, self.generate)
        response.calls.append(("answer", reply))
        response.text = reply.text
        response.cites_db = DB_CITATION in reply.text
        numbers = find_citations(reply.text)
        response.cited = [n for n in numbers if 1 <= n <= len(response.sources)]
        response.invalid_citations = [n for n in numbers if not 1 <= n <= len(response.sources)]
