"""Step 2.2 - split a 10-K's text into its standard sections ("Items").

Every 10-K follows the same SEC outline: Item 1 Business, Item 1A Risk Factors,
Item 7 Management's Discussion and Analysis (MD&A), and so on. Finding where
each Item starts is harder than it sounds:

  - The table of contents lists every Item too, all bunched together.
    -> For each Item, we keep the heading followed by the LONGEST stretch of text.
  - Some companies repeat a bare "Item 1" at the top of every page.
    -> A real heading must carry the Item's title ("Risk Factors"...) on the
       same line or the line right after it.
  - Banks such as JPMorgan put a short "see pages 52-167" under Item 7 and
    attach the full annual report after the last Item.
    -> A very long final section is labelled "Attached annual report".
  - A few companies (Intel) write the report in their own layout and list the
    Items only in a cross-reference index at the end.
    -> Normally only the cover page and table of contents come before the first
       Item. If that part is very long, it IS the report, kept as "Report body".
"""

import re
from dataclasses import dataclass

# Item -> (standard title, words that a real heading's title must contain)
ITEMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "1": ("Business", ("business",)),
    "1A": ("Risk Factors", ("risk factors",)),
    "1B": ("Unresolved Staff Comments", ("unresolved",)),
    "1C": ("Cybersecurity", ("cybersecurity",)),
    "2": ("Properties", ("properties",)),
    "3": ("Legal Proceedings", ("legal proceedings",)),
    "4": ("Mine Safety Disclosures", ("mine safety", "submission of matters")),
    "5": ("Market for Common Equity", ("market for",)),
    "6": ("Reserved / Selected Financial Data", ("reserved", "selected")),
    "7": ("Management's Discussion and Analysis", ("discussion and analysis",)),
    "7A": ("Market Risk Disclosures", ("market risk",)),
    "8": ("Financial Statements and Supplementary Data", ("financial statements",)),
    "9": ("Changes in and Disagreements with Accountants", ("disagreements",)),
    "9A": ("Controls and Procedures", ("controls and procedures",)),
    "9B": ("Other Information", ("other information",)),
    "9C": ("Foreign Jurisdictions that Prevent Inspections", ("foreign jurisdiction",)),
    "10": ("Directors, Executive Officers and Corporate Governance", ("directors",)),
    "11": ("Executive Compensation", ("executive compensation",)),
    "12": ("Security Ownership", ("security ownership",)),
    "13": ("Certain Relationships and Related Transactions", ("relationships",)),
    "14": ("Principal Accountant Fees and Services", ("accountant", "accounting fees")),
    "15": ("Exhibits and Financial Statement Schedules", ("exhibit",)),
    "16": ("Form 10-K Summary", ("summary",)),
}
ATTACHED_REPORT = ("AR", "Attached annual report")
REPORT_BODY = ("BODY", "Report body (company's own layout)")

# "Item 1A. Risk Factors", "ITEM 7 - MANAGEMENT'S...", and combined ones like "Items 1 and 2."
HEADING = re.compile(
    r"^items?\s*(\d{1,2}[a-c]?)\b(?:\s*(?:and|&)\s*\d{1,2}[a-c]?\b)?[\s.:|\-–—]*(.*)$", re.IGNORECASE
)
MAX_HEADING_CHARS = 200
# A final section longer than this is almost certainly an attached annual report,
# not an exhibit list or a one-paragraph summary.
ATTACHED_REPORT_MIN_WORDS = 30_000
# Text before the first Item longer than this is the report itself, not a cover page.
REPORT_BODY_MIN_WORDS = 5_000


@dataclass
class Section:
    item: str
    title: str
    text: str

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def _is_real_title(item: str, title: str) -> bool:
    keywords = ITEMS[item][1]
    title = title.lower()
    return len(title) <= MAX_HEADING_CHARS and any(k in title for k in keywords)


def find_headings(text: str) -> list[tuple[int, str]]:
    """Every line that looks like a real Item heading, as (character position, item)."""
    lines = text.split("\n")
    starts, pos = [], 0
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1

    headings = []
    for i, line in enumerate(lines):
        match = HEADING.match(line)
        if not match or len(line) > MAX_HEADING_CHARS:
            continue
        item = match.group(1).upper()
        if item not in ITEMS:
            continue
        same_line, next_line = match.group(2), lines[i + 1] if i + 1 < len(lines) else ""
        if _is_real_title(item, same_line) or (not same_line.strip() and _is_real_title(item, next_line)):
            headings.append((starts[i], item))
    return headings


def split_sections(text: str) -> list[Section]:
    headings = find_headings(text)
    if not headings:
        return [Section(*REPORT_BODY, text=text.strip())] if text.strip() else []

    # For each Item, keep the heading followed by the most text before the next heading.
    # Table-of-contents entries are only a few characters apart, so they lose.
    best: dict[str, tuple[int, int]] = {}  # item -> (gap, position)
    for (pos, item), (next_pos, _) in zip(headings, headings[1:] + [(len(text), "")]):
        gap = next_pos - pos
        if item not in best or gap > best[item][0]:
            best[item] = (gap, pos)

    chosen = sorted((pos, item) for item, (_, pos) in best.items())
    sections = []
    for (pos, item), (next_pos, _) in zip(chosen, chosen[1:] + [(len(text), "")]):
        body = text[pos:next_pos].strip()
        sections.append(Section(item=item, title=ITEMS[item][0], text=body))

    last = sections[-1]
    if last.item in ("15", "16") and last.n_words >= ATTACHED_REPORT_MIN_WORDS:
        last.item, last.title = ATTACHED_REPORT

    before_first_item = Section(*REPORT_BODY, text=text[:chosen[0][0]].strip())
    if before_first_item.n_words >= REPORT_BODY_MIN_WORDS:
        sections.insert(0, before_first_item)
    return sections
