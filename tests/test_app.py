from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from test_sql import db_path, scripted_llm  # noqa: F401 (db_path is a fixture)

from filinglens.assistant import Assistant
from filinglens.charts import growth_label
from filinglens.llm import LLMError, QuotaError

APP = str(Path(__file__).parent.parent / "app.py")
SQL = ("```sql\nSELECT ticker, fiscal_year, ROUND(revenue / 1e9, 2) AS revenue_bn FROM financials_wide "
       "WHERE ticker = 'NVDA' ORDER BY fiscal_year\n```")


@pytest.fixture
def run_app(db_path, monkeypatch):  # noqa: F811
    """Start the real app on the small test database, with a fake LLM."""
    opened = []

    def start(generate=None):
        def assistant():
            opened.append(Assistant(db_path, generate or scripted_llm()))
            return opened[-1]

        monkeypatch.setattr("filinglens.assistant.Assistant", assistant)
        st.cache_resource.clear()
        return AppTest.from_file(APP, default_timeout=30).run()

    yield start
    for assistant in opened:
        assistant.close()


def click(app: AppTest, label: str) -> AppTest:
    return next(b for b in app.button if b.label == label).click().run()


def test_every_tab_loads(run_app):
    app = run_app()
    assert not app.exception and not app.error
    assert [t.label for t in app.tabs] == ["F1 ASK", "F2 EXPLORE", "F3 RESEARCH"]
    assert len(app.table) == 2  # Explore's figures for the default companies, and Research's results
    assert any("Lazy Prices" in m.value for m in app.markdown)
    board = next(m.value for m in app.markdown if 'class="board"' in m.value)
    assert 'href="?company=NVDA"' in board and 'title="NVIDIA CORP"' in board and "<b>AMD</b>" in board


def test_a_square_on_the_board_starts_a_question(run_app):
    app = run_app(scripted_llm('{"route": "numbers", "reason": "figures"}', SQL, "Revenue rose [DB]."))
    app.query_params["company"] = "NVDA"
    app = app.run()
    assert "NVIDIA CORP" in app.text_input(key="question").value
    app.text_input(key="question").set_value("My own question")
    app = click(app, "GO")
    assert app.text_input(key="question").value == "My own question"  # the link doesn't overwrite what's typed next
    app.query_params["company"] = "NOT-A-TICKER"
    assert not app.run().exception


def test_asking_shows_the_answer_with_dollars_kept(run_app):
    app = run_app(scripted_llm('{"route": "numbers", "reason": "figures"}', SQL, "Revenue reached $27.0 billion [DB]."))
    app.text_input(key="question").set_value("What was NVIDIA's revenue?")
    app = click(app, "GO")
    assert not app.exception and not app.error
    assert any(r"\$27.0 billion" in m.value for m in app.markdown)  # unescaped, $ would start a formula
    assert any(":orange[✓ DB]" in m.value for m in app.markdown)
    assert any("Answered from the database in" in c.value for c in app.caption)
    assert any("Database result" in e.label for e in app.expander)


def test_an_example_button_asks_at_once(run_app):
    app = run_app(scripted_llm('{"route": "numbers", "reason": "figures"}', SQL, "Margins rose [DB]."))
    app = click(app, "Compare NVIDIA's and AMD's gross margins since 2021")
    assert any("Margins rose" in m.value for m in app.markdown)


def test_an_llm_failure_shows_its_message(run_app):
    def out_of_tokens(system, prompt):
        raise LLMError("The free-tier limit is used up.")

    app = run_app(out_of_tokens)
    app.text_input(key="question").set_value("Anything?")
    app = click(app, "GO")
    assert not app.exception
    assert "free-tier limit" in app.error[0].value


def test_a_missing_database_gets_a_plain_message(monkeypatch):
    def fail():
        raise RuntimeError("no database file")

    monkeypatch.setattr("filinglens.assistant.Assistant", fail)
    st.cache_resource.clear()
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert "no database file" in app.error[0].value


def test_growth_labels_never_show_minus_zero():
    labels = [growth_label(g) for g in (0.446, -0.005, -0.0003, 0.0002, -0.12)]
    assert labels == ["+45%", "-0.5%", "0.0%", "0.0%", "-12%"]


@pytest.mark.parametrize("error, message", [(QuotaError("limit reached, see .env"), "quota is used up"),
                                            (LLMError("rejected the API key"), "isn't responding")])
def test_online_visitors_see_a_plain_message_for_each_kind_of_llm_failure(run_app, monkeypatch, capsys, error, message):
    def fails(system, prompt):
        raise error

    app = run_app(fails)  # builds the fake Assistant the online branch below reuses
    monkeypatch.setattr("filinglens.online.ONLINE", True)
    monkeypatch.setattr("filinglens.online.OnlineAssistant", __import__("filinglens.assistant").assistant.Assistant)
    st.cache_resource.clear()
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.text_input(key="question").set_value("Anything?")
    app = click(app, "GO")
    assert message in app.error[0].value and ".env" not in app.error[0].value
    assert str(error) in capsys.readouterr().err  # the real reason still reaches the app's logs
