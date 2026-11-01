from types import SimpleNamespace

from filinglens.evaluation.answer_eval import figures_not_in_table, find_numbers, has_everything, price, states
from filinglens.evaluation.faithfulness import cited_sentences, grade, parse_verdicts
from filinglens.llm import LLMReply


def shown(text: str, gold: float) -> bool:
    return states(gold, find_numbers(text))


def test_a_figure_counts_however_it_is_written():
    gold = 211_915_000_000
    assert shown("Revenue was $211.92 billion [DB].", gold)
    assert shown("Revenue was $211.9bn.", gold)
    assert shown("Revenue was $212 billion.", gold)
    assert shown("Revenue was $211,915 million.", gold)
    assert shown("| 2023 | 211.92 |", gold)  # a table under the heading "USD bn"
    assert shown("Gross margin: 46.2%", 46.206)
    assert shown("a net loss of $11.82 billion", -11_817_000_000)


def test_a_wrong_figure_does_not_count():
    gold = 211_915_000_000
    assert not shown("Revenue was $211.92 million.", gold)   # wrong unit
    assert not shown("Revenue was $221.92 billion.", gold)
    assert not shown("Revenue was $200 billion.", gold)      # rounded too far
    assert not shown("1. Revenue [3] grew", 1_200_000_000)   # a list number and a citation aren't figures
    assert not shown("EPS was 1", 1.2)
    assert not shown("See the 10-K of FY2024", 10)


def test_rounding_up_to_one_decimal_is_accepted_only_for_that_decimal():
    assert shown("6.1", 6.13)
    assert not shown("6.2", 6.13)


def test_has_everything_needs_every_value_and_a_name():
    assert has_everything("Visa at 64.3% beat Mastercard at 55.8%.", [64.313, 55.813], [["Visa"]])
    assert not has_everything("Visa was at 64.3%.", [64.313, 55.813], [["Visa"]])
    assert not has_everything("Mastercard: 64.3%, other: 55.8%.", [64.313, 55.813], [["Visa"]])
    assert has_everything("WMT led with 611.29 [DB].", [611_289_000_000], [["Walmart", "WMT"]])


def test_figures_not_in_table_finds_only_new_figures():
    table = "ticker,fiscal_year,revenue_bn\nAAPL,2024,391.04\n"
    assert figures_not_in_table("Apple's fiscal 2024 revenue was $391.04 billion [DB].", table) == []
    assert figures_not_in_table("About $391.0 billion in 2023 [1].", table) == []
    assert figures_not_in_table("It grew 57% to $391.04 billion.", table) == [57.0]
    assert figures_not_in_table("R&D was $17.528 billion [DB].", "rnd_expense_usd\n17528000000.0\n") == []


def test_price_uses_input_and_output_rates():
    assert price([{"in": 1_000_000, "out": 0}, {"in": 0, "out": 1_000_000}]) == 0.75


def test_cited_sentences_keep_their_citations_and_count_the_rest():
    answer = ("**Short answer:**\nRevenue fell 14% [DB]. Demand was weak in PCs [1][3].\n"
              "- Inventory corrections hit the data center business [2].\n"
              "The reports do not explain anything else about this.")
    sentences, uncited = cited_sentences(answer)
    assert sentences == ["Revenue fell 14% [DB].", "Demand was weak in PCs [1][3].",
                         "Inventory corrections hit the data center business [2]."]
    assert uncited == 1


def test_a_citation_after_the_period_stays_with_its_sentence():
    assert cited_sentences("Sales rose. [2] Costs fell [1].")[0] == ["Sales rose. [2]", "Costs fell [1]."]


def test_parse_verdicts_reads_json_and_leaves_gaps():
    reply = 'Here: {"verdicts": [{"id": 1, "supported": true, "reason": "ok"}, {"id": 3, "supported": false}]}'
    assert parse_verdicts(reply, 3) == [True, None, False]
    assert parse_verdicts("not json", 2) == [None, None]
    assert parse_verdicts('{"verdicts": [{"id": 1, "supported": "yes"}]}', 1) == [None]


def test_grade_sends_numbered_statements():
    calls = []

    def generate(system, prompt):
        calls.append(prompt)
        return LLMReply('{"verdicts": [{"id": 1, "supported": true}, {"id": 2, "supported": false}]}', "fake", 1, 1, 0.1)

    assert grade("<source id=\"1\">x</source>", ["First [1].", "Second [1]."], generate) == [True, False]
    assert "1. First [1].\n2. Second [1]." in calls[0]


def test_answer_records_what_the_scoring_needs(tmp_path):
    from filinglens.evaluation import answer_eval
    import pandas as pd

    reply = LLMReply("x", "fake", 100, 10, 0.5)
    sql = SimpleNamespace(ok=True, table=pd.DataFrame({"revenue_bn": [211.92]}), sql="SELECT 1", error=None,
                          cannot_answer=None, attempts=1)
    response = SimpleNamespace(route=SimpleNamespace(route="numbers"), text="Revenue: $211.92 billion [DB].", sql=sql,
                               sources=[], cited=[], invalid_citations=[], cites_db=True, search_seconds=0.0,
                               calls=[("router", reply), ("SQL", reply), ("answer", reply)])
    assistant = SimpleNamespace(ask=lambda question: response,
                                con=SimpleNamespace(execute=lambda sql: SimpleNamespace(
                                    fetchall=lambda: [(211_915_000_000.0,)])))
    row = answer_eval.answer(assistant, {"question": "q", "gold_sql": "SELECT 1"})
    assert [c["stage"] for c in row["calls"]] == ["router", "SQL", "answer"]
    scored = answer_eval.score(row, [])
    assert scored["sql_right"] and scored["answer_right"] and scored["extra_figures"] == []
    assert not answer_eval.score(row, [["Apple"]])["answer_right"]


def test_csv_commas_are_not_thousands_separators():
    from filinglens.evaluation import answer_eval

    row = {"table": "ticker,date,close_usd\nAAPL,2024-12-31,248.62\n", "gold": [248.6158], "text": "It closed at $248.62."}
    scored = answer_eval.score(row, [])
    assert scored["sql_right"] and scored["extra_figures"] == []
