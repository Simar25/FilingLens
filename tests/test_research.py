import numpy as np
import pandas as pd
import pytest

from filinglens.research import pick_companies
from filinglens.research.analysis import assign_groups, group_returns, high_minus_low, regression
from filinglens.research.returns import future_returns
from filinglens.research.text_change import (
    consecutive_pairs, cosine_similarity, embedding_similarity, jaccard_similarity, name_words, words,
)
from filinglens.research.tone import section_tone, sentences
from filinglens.research.topics import count_mentions, topic_mentions, topic_trends


# ---------- 4.1 text change ----------

def test_words_keeps_content_words_only():
    text = "In 2024, the Company's revenue grew 5% and JPMorganChase's risks changed."
    assert words(text, ignore=name_words(["JPMORGAN CHASE & CO"])) == ["company", "revenue", "grew", "risks", "changed"]


def test_name_words_include_names_run_together():
    # JPMorgan started writing its name as one word in 2024; both spellings must be ignored.
    assert {"jpmorgan", "chase", "jpmorganchase", "facebook"} <= name_words(["JPMorgan Chase & Co", "Facebook Inc"])


def test_cosine_similarity():
    assert cosine_similarity(["risk", "supply"], ["risk", "supply"]) == pytest.approx(1.0)
    assert cosine_similarity(["risk"], ["tariff"]) == 0.0
    # Word counts matter: one extra mention of a word moves it, but only a little.
    assert 0.9 < cosine_similarity(["risk", "supply", "chain"], ["risk", "supply", "chain", "risk"]) < 1


def test_jaccard_similarity_uses_sets():
    # {risk, supply} vs {risk, tariff}: 1 shared word out of 3 different words
    assert jaccard_similarity(["risk", "supply", "supply"], ["risk", "tariff"]) == pytest.approx(1 / 3)


def test_embedding_similarity_drops_when_a_new_topic_appears():
    a, b, c = np.eye(3)  # three unrelated unit vectors
    assert embedding_similarity(np.stack([a, b]), np.stack([a, b])) == pytest.approx(1.0)
    # A new chunk (c) with no match last year lowers the score, in either direction.
    added = embedding_similarity(np.stack([a, b, c]), np.stack([a, b]))
    removed = embedding_similarity(np.stack([a, b]), np.stack([a, b, c]))
    assert added == pytest.approx(removed) and added < 1


def test_consecutive_pairs_need_both_years_and_long_sections():
    sections = pd.DataFrame({
        "ticker": ["A", "A", "A", "A", "B", "B"],
        "fiscal_year": [2021, 2022, 2024, 2025, 2021, 2022],
        "item": ["1A"] * 6,
        "accession_number": ["a21", "a22", "a24", "a25", "b21", "b22"],
        "n_words": [5000, 5000, 5000, 5000, 5000, 40],  # B's 2022 section is only a pointer
    })
    pairs = consecutive_pairs(sections)
    # 2022 vs 2021 and 2025 vs 2024; no 2024 vs 2022 across the gap; nothing for B.
    assert sorted(zip(pairs["accession_number"], pairs["accession_number_prev"])) == [("a22", "a21"), ("a25", "a24")]


# ---------- 4.2 tone ----------

def test_sentences_skip_table_rows_and_fragments():
    text = ("Item 7. MD&A\n"
            "Revenue increased due to higher demand for our products in all regions.\n"
            "Americas | $167,045 | 3% | $162,560\n"
            "Fiscal 2024 Highlights\n"
            "12,345 23,456 34,567 45,678 56,789 67,890 78,901.")
    assert sentences(text) == ["Revenue increased due to higher demand for our products in all regions."]


def test_section_tone():
    probs = np.array([[0.8, 0.1], [0.1, 0.8], [0.1, 0.1]])  # positive, negative, neutral sentence
    tone = section_tone(probs)
    assert tone["tone"] == pytest.approx(0.0)
    assert tone["positive_share"] == pytest.approx(1 / 3) and tone["negative_share"] == pytest.approx(1 / 3)


# ---------- 4.3 returns ----------

@pytest.fixture
def prices():
    days = pd.bdate_range("2023-01-02", periods=400)
    stock = 100 * 1.001 ** np.arange(400)   # rises 0.1% a day
    market = 100 * 1.0005 ** np.arange(400)  # rises 0.05% a day
    return pd.concat([pd.DataFrame({"ticker": "AAA", "date": days, "close": stock}),
                      pd.DataFrame({"ticker": "SPY", "date": days, "close": market})])


def test_returns_start_the_trading_day_after_the_filing(prices):
    filings = pd.DataFrame({"ticker": ["AAA"], "fiscal_year": [2023], "accession_number": ["x"],
                            "filing_date": [pd.Timestamp("2023-01-06")]})  # a Friday
    r = future_returns(filings, prices).iloc[0]
    assert r["start_date"] == pd.Timestamp("2023-01-09")  # the next Monday
    assert r["ret_3m"] == pytest.approx(1.001 ** 63 - 1)
    assert r["abnormal_3m"] == pytest.approx((1.001 ** 63 - 1) - (1.0005 ** 63 - 1))
    assert np.isnan(r["past_abnormal_12m"])  # no year of history before it


def test_incomplete_windows_are_missing_not_partial(prices):
    filings = pd.DataFrame({"ticker": ["AAA"], "fiscal_year": [2024], "accession_number": ["x"],
                            "filing_date": [prices["date"].max() - pd.Timedelta(days=150)]})
    r = future_returns(filings, prices).iloc[0]
    assert not np.isnan(r["abnormal_3m"])
    assert np.isnan(r["abnormal_6m"]) and np.isnan(r["abnormal_12m"])
    assert r["past_abnormal_12m"] > 0  # the stock beat the market over the year before


# ---------- 4.5 topics ----------

def test_topic_patterns_avoid_false_hits():
    counts = count_mentions("We said AI and artificial intelligence matter. The Inflation Reduction Act; "
                            "inflationary pressure. The economic climate; climate-related risks. Aid.")
    assert counts["Artificial intelligence"] == 2
    assert counts["Inflation"] == 1
    assert counts["Climate change"] == 1


def test_topic_trends():
    reports = pd.DataFrame({"ticker": ["A", "B"], "fiscal_year": [2024, 2024], "accession_number": ["a", "b"],
                            "text": ["tariffs " * 10 + "word " * 90, "word " * 100]})
    trends = topic_trends(topic_mentions(reports)).set_index("topic").loc["Tariffs"]
    assert trends["share_mentioning"] == 0.5
    assert trends["per_10k_words"] == pytest.approx((10 / 100 * 10_000 + 0) / 2)


# ---------- 4.4 statistics ----------

@pytest.fixture
def study():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"ticker": [f"T{i % 80}" for i in range(n)], "fiscal_year": 2021 + np.arange(n) // 80,
                       "sector": rng.choice(["Tech", "Energy"], n), "change": rng.uniform(0, 0.1, n),
                       "tone": rng.normal(0, 0.1, n), "past_abnormal_12m": rng.normal(0, 0.2, n),
                       "log_assets": rng.normal(25, 1, n)})
    df["abnormal_6m"] = -2.0 * df["change"] + rng.normal(0, 0.05, n)  # more change, lower return
    return df


def test_groups_are_equal_sized_within_each_year(study):
    groups = assign_groups(study, "change")
    assert groups.groupby(study["fiscal_year"]).value_counts().unique().tolist() == [16]
    # Group 5 has the most change in every year.
    assert (study.groupby(groups)["change"].mean().diff().dropna() > 0).all()


def test_high_minus_low_finds_a_planted_effect(study):
    result = high_minus_low(study, "change", "abnormal_6m")
    assert result["difference"] < 0 and result["p_value"] < 0.01
    assert len(group_returns(study, "change", "abnormal_6m")) == 5


def test_regression_finds_a_planted_effect(study):
    result = regression(study, "change", "abnormal_6m")
    # change ~ uniform(0, 0.1) has std 0.029: 1 std of change = -2 * 0.029 = -5.8 percentage points
    assert result.params["change"] == pytest.approx(-5.8, abs=1.0)
    assert result.pvalues["change"] < 0.001


# ---------- picking companies ----------

def test_has_every_report(monkeypatch):
    def fake_filings(dates):
        return lambda client, cik, filed_since: pd.DataFrame({"filing_date": pd.to_datetime(dates)})

    today = pd.Timestamp.today()
    yearly = [today - pd.DateOffset(years=k, months=2) for k in range(6)]
    monkeypatch.setattr(pick_companies, "list_10k_filings", fake_filings(yearly))
    assert pick_companies.has_every_report(None, 1, 2020, 2025) == (True, "ok")

    monkeypatch.setattr(pick_companies, "list_10k_filings", fake_filings(yearly[:2]))  # a 2024 listing
    assert pick_companies.has_every_report(None, 1, 2020, 2025) == (False, "only 2 10-Ks since 2020")
