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


def test_gemini_rate_limit_is_a_clear_error(monkeypatch):
    import httpx
    from google import genai
    from google.genai._gaos.lib.compat_errors import APIError

    response = httpx.Response(429, request=httpx.Request("POST", "https://gemini.test"))

    class FakeClient:  # stands in for genai.Client, so no network call is made
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        class interactions:
            @staticmethod
            def create(**kwargs):
                raise APIError.generate(429, None, "Resource exhausted", response)

    monkeypatch.setattr(genai, "Client", FakeClient)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with pytest.raises(llm.LLMError, match="free-tier limit"):
        llm.generate("system", "prompt")
