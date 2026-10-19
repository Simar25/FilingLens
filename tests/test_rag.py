import numpy as np
import pandas as pd
import pytest

from filinglens import llm
from filinglens.llm import LLMReply
from filinglens.rag.answer import NO_SOURCES_ANSWER, answer_question, build_prompt, find_citations
from filinglens.rag.embeddings import passage_text
from filinglens.rag.filters import Filters, detect_filters
from filinglens.rag.vector_store import SearchResult, VectorStore, build_where

COMPANIES = {
    "AAPL": "Apple Inc.", "AMD": "ADVANCED MICRO DEVICES INC", "NVDA": "NVIDIA CORP",
    "GOOGL": "Alphabet Inc.", "META": "Meta Platforms, Inc.", "JPM": "JPMORGAN CHASE & CO",
}
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


# ---------- filters: which companies and years is a question about? ----------

def test_company_name_and_year():
    f = detect_filters("What supply chain risks did Apple describe in 2024?", COMPANIES, YEARS)
    assert (f.tickers, f.years) == (["AAPL"], [2024])


def test_tickers_possessives_and_since():
    f = detect_filters("Compare NVIDIA's and AMD's gross margins since 2022", COMPANIES, YEARS)
    assert (f.tickers, f.years) == (["AMD", "NVDA"], [2022, 2023, 2024, 2025])


def test_nicknames_and_short_fiscal_year():
    f = detect_filters("How did Google and Facebook describe ad revenue in FY23?", COMPANIES, YEARS)
    assert (f.tickers, f.years) == (["GOOGL", "META"], [2023])


def test_year_range_and_fy_prefix():
    assert detect_filters("JPMorgan credit risk from 2021 to 2023", COMPANIES, YEARS).years == [2021, 2022, 2023]
    assert detect_filters("Meta's FY2022 and fiscal 2024 capex plans", COMPANIES, YEARS).years == [2022, 2024]


def test_generic_words_are_not_companies():
    f = detect_filters("Which companies mention advanced packaging or metadata?", COMPANIES, YEARS)
    assert (f.tickers, f.years) == ([], [])


# ---------- vector store ----------

def test_where_clause():
    assert build_where(Filters()) is None
    assert build_where(Filters(tickers=["AAPL"])) == {"ticker": {"$in": ["AAPL"]}}
    assert build_where(Filters(tickers=["AAPL"], years=[2024])) == {
        "$and": [{"ticker": {"$in": ["AAPL"]}}, {"fiscal_year": {"$in": [2024]}}]}


def make_chunks() -> pd.DataFrame:
    rows = [("AAPL", 2024, "1A"), ("AAPL", 2023, "1A"), ("MSFT", 2024, "7")]
    return pd.DataFrame([{
        "chunk_id": f"{t}-{y}-{i}-0000", "ticker": t, "company": f"{t} Inc.", "fiscal_year": y, "item": i,
        "title": "Risk Factors", "accession_number": f"acc-{t}-{y}", "chunk_index": 0, "text": f"text {t} {y}",
    } for t, y, i in rows])


def test_search_respects_filters_and_ranks_by_similarity(tmp_path):
    store = VectorStore(tmp_path)
    vectors = np.array([[1, 0, 0], [0.8, 0.6, 0], [0, 0, 1]], dtype=np.float32)  # already length 1
    store.add(make_chunks(), vectors)
    assert store.count() == 3

    query = np.array([1, 0, 0], dtype=np.float32)
    everything = store.search(query, k=3)
    assert [r.chunk_id for r in everything] == ["AAPL-2024-1A-0000", "AAPL-2023-1A-0000", "MSFT-2024-7-0000"]
    assert everything[0].similarity == pytest.approx(1.0)
    assert everything[1].similarity == pytest.approx(0.8)

    only_2023 = store.search(query, k=3, filters=Filters(tickers=["AAPL"], years=[2023]))
    assert [(r.ticker, r.fiscal_year, r.text) for r in only_2023] == [("AAPL", 2023, "text AAPL 2023")]


def test_reindexing_a_company_replaces_its_vectors(tmp_path):
    store = VectorStore(tmp_path)
    chunks = make_chunks()
    store.add(chunks, np.eye(3, dtype=np.float32))
    store.delete_tickers(["AAPL"])
    assert store.count() == 1
    store.add(chunks[chunks["ticker"] == "AAPL"], np.eye(3, dtype=np.float32)[:2])
    assert store.count() == 3


# ---------- embeddings ----------

def test_passage_header_names_company_year_and_section():
    text = passage_text("Apple Inc.", "AAPL", 2024, "Risk Factors", "Our products face competition.")
    assert text == "Apple Inc. (AAPL) 10-K, fiscal year 2024, Risk Factors\nOur products face competition."


# ---------- answers and citations ----------

def source(n: int, ticker: str = "AAPL", item: str = "1A") -> SearchResult:
    return SearchResult(f"{ticker}-2024-{item}-{n:04d}", ticker, "Apple Inc.", 2024, item, "Risk Factors",
                        f"passage {n}", 0.5)


def test_prompt_numbers_sources_and_ends_with_question():
    prompt = build_prompt("What are the risks?", [source(1), source(2)])
    assert prompt.startswith('<source id="1" company="Apple Inc." ticker="AAPL" fiscal_year="2024" '
                             'section="Item 1A Risk Factors">\npassage 1\n</source>')
    assert '<source id="2"' in prompt
    assert prompt.endswith("Question: What are the risks?")


def test_find_citations():
    assert find_citations("Supply is concentrated [1][3]. Prices rose [2, 4]. See 2024 [x].") == [1, 2, 3, 4]


def test_answer_keeps_valid_citations_and_flags_invented_ones():
    def fake_llm(system: str, prompt: str) -> LLMReply:
        assert "only" in system and "passage 2" in prompt
        return LLMReply("Risk A [1]. Risk B [9].", "fake", 100, 10, 0.1)

    answer = answer_question("q", [source(1), source(2)], Filters(), fake_llm)
    assert (answer.cited, answer.invalid_citations) == ([1], [9])


def test_no_sources_means_no_llm_call():
    def must_not_be_called(system: str, prompt: str) -> LLMReply:
        raise AssertionError("LLM called without sources")

    assert answer_question("q", [], Filters(), must_not_be_called).text == NO_SOURCES_ANSWER


def test_unknown_llm_provider_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(llm.LLMError, match="gemini"):
        llm.current_settings()


class FakeGemini:
    """Stands in for genai.Client: each request gives the next outcome (an exception, or reply text)."""

    def __init__(self, monkeypatch, *outcomes):
        from google import genai
        from types import SimpleNamespace

        self.outcomes, self.calls, self.waits = list(outcomes), 0, []
        fake = self

        class Client:
            def __init__(self, **options):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

            class interactions:
                @staticmethod
                def create(**kwargs):
                    fake.calls += 1
                    outcome = fake.outcomes.pop(0)
                    if isinstance(outcome, Exception):
                        raise outcome
                    usage = SimpleNamespace(total_input_tokens=10, total_output_tokens=2, total_thought_tokens=0)
                    return SimpleNamespace(output_text=outcome, usage=usage)

        monkeypatch.setattr(genai, "Client", Client)
        monkeypatch.setattr(llm.time, "sleep", self.waits.append)  # record waits instead of waiting
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_MODEL", "gemini-test")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def gemini_error(status: int, headers: dict | None = None, message: str = "error") -> Exception:
    import httpx
    from google.genai._gaos.lib.compat_errors import APIError

    response = httpx.Response(status, headers=headers, request=httpx.Request("POST", "https://gemini.test"))
    return APIError.generate(status, {"error": {"message": message}}, None, response)


def test_gemini_rate_limit_is_a_clear_error(monkeypatch):
    fake = FakeGemini(monkeypatch, *[gemini_error(429)] * 3)  # a per-minute limit that doesn't clear
    with pytest.raises(llm.LLMError, match="free-tier limit"):
        llm.generate("system", "prompt")
    assert (fake.calls, fake.waits) == (3, [2, 5])


def test_daily_limit_stops_at_once_instead_of_sleeping_for_hours(monkeypatch):
    # The real reply seen on 2026-10-19. The client library used to sleep the 49,340 s it asks for.
    daily = gemini_error(429, {"retry-after": "49340"}, "limit: 20 requests per day on Free Tier. "
                                                         "Please retry in 13h42m20s")
    fake = FakeGemini(monkeypatch, daily)
    with pytest.raises(llm.LLMError, match="used up. Gemini says: limit: 20 requests per day"):
        llm.generate("system", "prompt")
    assert (fake.calls, fake.waits) == (1, [])


def test_a_short_requested_wait_is_waited_out(monkeypatch):
    fake = FakeGemini(monkeypatch, gemini_error(429, {"retry-after": "1"}), "the answer")
    assert llm.generate("system", "prompt").text == "the answer"
    assert (fake.calls, fake.waits) == (2, [1.0])


def test_gemini_overload_and_timeout_are_clear_errors(monkeypatch):
    import httpx
    from google.genai._gaos.lib.compat_errors import APITimeoutError

    FakeGemini(monkeypatch, *[gemini_error(503)] * 3)
    with pytest.raises(llm.LLMError, match="overloaded"):
        llm.generate("system", "prompt")

    FakeGemini(monkeypatch, APITimeoutError(httpx.Request("POST", "https://gemini.test")))
    with pytest.raises(llm.LLMError, match="didn't answer within"):
        llm.generate("system", "prompt")


class FakeGroq:
    """Stands in for requests.post to Groq: each call gives the next (status, body, headers) reply."""

    def __init__(self, monkeypatch, *replies):
        import json

        import requests

        self.replies, self.requests, self.waits = list(replies), [], []

        def post(url, json=None, timeout=None, headers=None):
            self.requests.append(json)
            status, body, extra_headers = self.replies.pop(0)
            response = requests.Response()
            response.status_code, response._content = status, _json.dumps(body).encode()
            response.headers.update(extra_headers or {})
            return response

        _json = json
        monkeypatch.setattr(llm.requests, "post", post)
        monkeypatch.setattr(llm.time, "sleep", self.waits.append)
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "test-key")


GROQ_OK = (200, {"choices": [{"message": {"content": "the answer"}}],
                 "usage": {"prompt_tokens": 50, "completion_tokens": 7}}, None)


def test_groq_answers_with_medium_reasoning(monkeypatch):
    fake = FakeGroq(monkeypatch, GROQ_OK)
    reply = llm.generate("system", "prompt")
    assert (reply.text, reply.model, reply.input_tokens, reply.output_tokens) == (
        "the answer", "openai/gpt-oss-120b", 50, 7)
    assert fake.requests[0]["reasoning_effort"] == "medium"


def test_groq_per_minute_limit_is_waited_out(monkeypatch):
    fake = FakeGroq(monkeypatch, (429, {"error": {"message": "tokens per minute"}}, {"retry-after": "7"}), GROQ_OK)
    assert llm.generate("system", "prompt").text == "the answer"
    assert fake.waits == [7.0]


def test_groq_daily_limit_stops_at_once(monkeypatch):
    fake = FakeGroq(monkeypatch, (429, {"error": {"message": "tokens per day (TPD): Limit 200000"}},
                                  {"retry-after": "3600"}))
    with pytest.raises(llm.LLMError, match="used up. Groq says: tokens per day"):
        llm.generate("system", "prompt")
    assert (len(fake.requests), fake.waits) == (1, [])


def test_groq_prompt_too_large_suggests_fewer_passages(monkeypatch):
    FakeGroq(monkeypatch, (413, {"error": {"message": "Request too large: Limit 8000, Requested 9500"}}, None))
    with pytest.raises(llm.LLMError, match="-k 4"):
        llm.generate("system", "prompt")


def test_model_quirks_are_cleaned():
    # What gpt-oss on Groq really sent on 2026-10-19 (plus its longer citation form).
    raw = ("fiscal\u202f2024, **$108.81\u202fbillion**\u3010DB\u3011 because of year\u2011over\u2011year "
           "demand\u30102\u2020L1-L3\u3011.")
    assert llm.clean_text(raw) == "fiscal 2024, **$108.81 billion**[DB] because of year-over-year demand[2]."
