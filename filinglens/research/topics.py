"""Step 4.5 - which topics do annual reports talk about more, year by year?

A simple, transparent method: count mentions of a few patterns in each whole
report, then per fiscal year compute
  - the share of reports that mention the topic at least once, and
  - mentions per 10,000 words (so long reports don't count more).

Patterns are regular expressions, written to catch the usual spellings
("supply chain", "supply-chain", "supply chains") and to avoid false hits:
"AI" must be in capitals as its own word, so "said" or "Aid" don't count;
"climate" only counts as "climate change" or "climate-related", not "the
economic climate"; and the Inflation Reduction Act (a 2022 law about energy
and drug prices) doesn't count as a mention of inflation.
A report mentioning "artificial intelligence" in a list of generic risks
counts the same as one about its AI products: this measures attention, not
importance.
"""

import re

import pandas as pd

TOPICS = {
    "Artificial intelligence": re.compile(r"\bartificial intelligence\b|\b(?-i:AI)\b", re.I),
    "Cybersecurity": re.compile(r"\bcyber", re.I),  # cybersecurity, cyber-attack, cyber incidents
    "Supply chain": re.compile(r"\bsupply[- ]chains?\b", re.I),
    "Tariffs": re.compile(r"\btariffs?\b", re.I),
    "Inflation": re.compile(r"\binflation(?!\s+reduction\s+act)", re.I),  # inflation, inflationary
    "COVID-19 / pandemic": re.compile(r"\bcovid|\bpandemic|\bcoronavirus", re.I),
    "Climate change": re.compile(r"\bclimate[- ](?:change|related)", re.I),
}


def count_mentions(text: str) -> dict[str, int]:
    return {topic: len(pattern.findall(text)) for topic, pattern in TOPICS.items()}


def topic_mentions(reports: pd.DataFrame) -> pd.DataFrame:
    """One row per report and topic. `reports`: ticker, fiscal_year, accession_number, text."""
    rows = []
    for r in reports.itertuples():
        n_words = len(r.text.split())
        for topic, count in count_mentions(r.text).items():
            rows.append({"ticker": r.ticker, "fiscal_year": r.fiscal_year, "accession_number": r.accession_number,
                         "topic": topic, "mentions": count, "n_words": n_words})
    return pd.DataFrame(rows)


def topic_trends(mentions: pd.DataFrame) -> pd.DataFrame:
    """Per topic and fiscal year: share of reports mentioning it, and mentions per 10,000 words."""
    per_report = mentions.assign(mentioned=mentions["mentions"] > 0,
                                 per_10k_words=mentions["mentions"] / mentions["n_words"] * 10_000)
    return (per_report.groupby(["topic", "fiscal_year"])
            .agg(reports=("mentioned", "size"), share_mentioning=("mentioned", "mean"),
                 per_10k_words=("per_10k_words", "mean"))
            .reset_index())
