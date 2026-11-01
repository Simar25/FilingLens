import gzip
import json

import pytest
from test_sql import db_path, scripted_llm  # noqa: F401 (db_path is a fixture)

from filinglens import online
from filinglens.assistant import SetupError
from filinglens.online import OnlineAssistant
from filinglens.rag.filters import Filters


@pytest.fixture
def assistant(db_path, tmp_path, monkeypatch):  # noqa: F811
    """The online assistant on the small test database, with two tiny reports."""
    monkeypatch.setattr(online, "REPORTS_DIR", tmp_path)
    online.load_report.cache_clear()
    for ticker, year, passages in (
            ("NVDA", 2023, [["NVDA-2023-7-0001", "7", "MD&A", "Data center revenue grew on demand for accelerators."],
                            ["NVDA-2023-7-0002", "7", "MD&A", "Gaming revenue fell as channel inventory corrected."],
                            ["NVDA-2023-1A-0001", "1A", "Risk Factors", "Export controls could limit our sales in China."]]),
            ("AMD", 2023, [["AMD-2023-7-0001", "7", "MD&A", "Client revenue fell on weak personal computer demand."]])):
        with gzip.open(tmp_path / f"{ticker}-{year}.json.gz", "wt", encoding="utf-8") as f:
            json.dump({"company": f"{ticker} INC", "passages": passages}, f)
    with OnlineAssistant(db_path, scripted_llm()) as a:
        a.report_years = [2023]
        yield a
    online.load_report.cache_clear()


def test_search_finds_the_passage_within_the_named_company(assistant):
    results = assistant.search("export controls in China", Filters(tickers=["NVDA"], years=[2023]))
    assert [r.chunk_id for r in results] == ["NVDA-2023-1A-0001"]
    assert results[0].company == "NVDA INC" and results[0].section == "Item 1A Risk Factors"


def test_search_stays_inside_the_filters(assistant):
    results = assistant.search("revenue fell", Filters(tickers=["AMD"]))
    assert {r.ticker for r in results} == {"AMD"}


def test_no_year_means_all_of_the_companys_reports(assistant):
    assert assistant.search("gaming revenue", Filters(tickers=["NVDA"]))[0].chunk_id == "NVDA-2023-7-0002"


def test_a_question_naming_no_company_gets_a_clear_message(assistant):
    with pytest.raises(SetupError, match="companies on the market map"):
        assistant.search("What are the biggest risks?", Filters())


def test_a_report_that_does_not_exist_finds_nothing(assistant):
    assert assistant.search("revenue", Filters(tickers=["NVDA"], years=[1999])) == []


def test_why_questions_also_search_the_words_reports_explain_with(assistant, monkeypatch):
    searched = []
    original = online.KeywordIndex.search

    def spy(self, question, filters, k):
        searched.append(question)
        return original(self, question, filters, k)

    monkeypatch.setattr(online.KeywordIndex, "search", spy)
    assistant.search("Why did gaming revenue fall?", Filters(tickers=["NVDA"]))
    assistant.search("What are the export controls?", Filters(tickers=["NVDA"]))
    assert "primarily due" in searched[0] and "primarily due" not in searched[1]
