"""Work out which companies and fiscal years a question is about.

"What supply chain risks did Apple describe in 2024?" should only search
Apple's FY2024 report. Without this filter, the closest passages might come
from Microsoft or from 2021, because many reports use similar wording.

Simple rules are enough here:
  - Companies: the ticker in capitals (AAPL), the company's name (Apple),
    or a common nickname (Google, Facebook).
  - Years: "2024", "FY24", "2021-2023", "2021 to 2023", "since 2021".
The `ask` command prints what it detected, and --ticker / --year override it.
"""

import re
from dataclasses import dataclass, field

# Names people use that aren't part of the SEC company name.
NICKNAMES = {"google": "GOOGL", "facebook": "META", "jp morgan": "JPM"}
# Words dropped from SEC names: "ADVANCED MICRO DEVICES INC" -> "advanced micro devices".
NAME_SUFFIXES = {"inc", "corp", "corporation", "co", "com", "company", "ltd", "plc", "platforms", "holdings", "group"}
# First words too common to stand for a company on their own ("advanced packaging" isn't AMD).
GENERIC_FIRST_WORDS = {"advanced", "american", "general", "united", "first", "international", "national", "bank"}

YEAR = re.compile(r"(?:\bFY\s*'?|\b)(20\d{2})\b|\bFY\s*'?(\d{2})\b", re.IGNORECASE)  # 2024, FY2024, FY24
YEAR_RANGE = re.compile(r"\b(20\d{2})\s*(?:-|–|to|through|until)\s*(20\d{2})\b", re.IGNORECASE)
SINCE_YEAR = re.compile(r"\bsince\s+(?:fiscal\s+|FY\s*)?(20\d{2})\b", re.IGNORECASE)


@dataclass
class Filters:
    tickers: list[str] = field(default_factory=list)  # empty = all companies
    years: list[int] = field(default_factory=list)    # empty = all years
    items: list[str] = field(default_factory=list)    # empty = all sections

    def describe(self) -> str:
        parts = [
            ", ".join(self.tickers) if self.tickers else "all companies",
            "fiscal year " + ", ".join(map(str, self.years)) if self.years else "all years",
        ]
        if self.items:
            parts.append("Item " + ", ".join(self.items))
        return "; ".join(parts)


def company_aliases(companies: dict[str, str]) -> dict[str, str]:
    """{ticker: SEC name} -> {lowercase alias: ticker}."""
    aliases = {}
    for ticker, name in companies.items():
        words = [w for w in re.sub(r"[^a-z0-9 ]", " ", name.lower()).split() if w not in NAME_SUFFIXES]
        if words:
            aliases[" ".join(words)] = ticker              # "advanced micro devices", "jpmorgan chase"
            if len(words[0]) >= 4 and words[0] not in GENERIC_FIRST_WORDS:
                aliases[words[0]] = ticker                 # "apple", "jpmorgan", "amazon"
    aliases.update({alias: t for alias, t in NICKNAMES.items() if t in companies})
    return aliases


def detect_companies(question: str, companies: dict[str, str]) -> list[str]:
    found = set()
    for ticker in companies:
        # Case-sensitive, so "AMD" matches but a word like "amd" inside text doesn't.
        # One-letter tickers (Visa is "V") would match ordinary capital letters, so they're skipped.
        if len(ticker) > 1 and re.search(rf"\b{re.escape(ticker)}\b", question):
            found.add(ticker)
    lowered = question.lower()
    for alias, ticker in company_aliases(companies).items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):  # also matches "Apple's"
            found.add(ticker)
    return sorted(found)


def detect_years(question: str, available_years: list[int]) -> list[int]:
    years = set()
    for match in YEAR.finditer(question):
        years.add(int(match.group(1)) if match.group(1) else 2000 + int(match.group(2)))
    for match in YEAR_RANGE.finditer(question):
        years.update(range(int(match.group(1)), int(match.group(2)) + 1))
    for match in SINCE_YEAR.finditer(question):
        years.update(y for y in available_years if y >= int(match.group(1)))
    return sorted(years)


def detect_filters(question: str, companies: dict[str, str], available_years: list[int]) -> Filters:
    return Filters(tickers=detect_companies(question, companies), years=detect_years(question, available_years))
