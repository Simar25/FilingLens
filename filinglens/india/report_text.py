"""India pilot - find the MD&A in an annual-report PDF and get its text.

Indian annual reports aren't built like a 10-K. There is no "Item 7": SEBI's
listing rules (LODR, Schedule V) require a "Management Discussion and Analysis"
section, but each company places and titles it its own way (a chapter of its
own, part of the Board's or Directors' report, "Management's discussion and
analysis"...). The reports are designed PDFs of 200-600 pages, not web pages.

Two ways to find the MD&A, in this order:
  1. Bookmarks. Many PDFs carry a clickable table of contents with the page
     where each section starts. The MD&A runs from its bookmark to the page
     before the next bookmark at the same or a higher level. Exact, when there.
  2. Headings. Otherwise, the first page with the MD&A title as a line of its
     own (a sentence that mentions the MD&A isn't its heading; contents pages,
     which list every title, are skipped). Titles split over two lines are
     joined first. It ends at the earliest of:
       - a "Cautionary statement" heading (that page included): many Indian
         MD&As close with this forward-looking-statements disclaimer;
       - the page before the next major section's heading (corporate
         governance report, sustainability report, auditor's report, financial
         statements, AGM notice);
       - the page before the next bookmark, if the PDF has bookmarks but none
         for the MD&A (e.g. an MD&A inside a bookmarked Directors' report).
     When the MD&A sits inside the Directors' report and none of these mark
     its end, the rest of that report is included: the end is approximate.
"""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

MDNA = re.compile(r"management\W{0,3}s?\W*discussion\s*(?:and|&)\s*analysis(?:\s+report)?", re.IGNORECASE)
NEXT_SECTION = re.compile(
    r"(?:report\s+on\s+)?corporate\s+governance(?:\s+report)?"
    r"|business\s+responsibility(?:\s+(?:and|&)\s+sustainability)?\s+report"
    r"|independent\s+auditor\W{0,3}s?\s+report|(?:standalone|consolidated)\s+financial\s+statements"
    r"|notice\s+of\s+(?:the\s+)?[\w\s]{0,20}annual\s+general\s+meeting",
    re.IGNORECASE)
CAUTIONARY = re.compile(r"cautionary\s+statements?", re.IGNORECASE)
MAX_HEADING_CHARS = 90  # longer lines are sentences, not headings
# A heading may carry a little more than the title: a number ("4.", a page number), "Annexure 5 -",
# "Report". Any other word means the line is part of a sentence: "...Analysis in the Annual Report".
ALLOWED_EXTRA = re.compile(r"(?:[\W\d_]|annexure|report|section|part|chapter|[ivx]+\b)*", re.IGNORECASE)
SUBTITLE = re.compile(r"\s*[–—:|-]\s*\S")  # running headers like "MD&A – Business Overview"
CONTENTS_TITLES = 4     # a page naming this many sections is a table of contents
EDGE_LINES = 8          # lines at the top and bottom of a page, where headers, footers and navigation sit
FURNITURE_SHARE = 0.3   # a line at the edges of this share of all pages is page furniture


@dataclass
class Section:
    first_page: int  # 1-based, as a PDF viewer numbers pages
    last_page: int
    method: str      # "bookmarks" or "headings"
    title: str

    @property
    def pages(self) -> int:
        return self.last_page - self.first_page + 1


def section_from_bookmarks(toc: list, page_count: int, pattern: re.Pattern = MDNA) -> Section | None:
    """toc: PyMuPDF's [level, title, page] list (pages 1-based; -1 = bookmark without a page)."""
    for i, (level, title, page) in enumerate(toc):
        if page < 1 or not pattern.search(title):
            continue
        last = page_count
        for next_level, _, next_page in toc[i + 1:]:
            if next_level <= level and next_page > page:  # deeper levels are the MD&A's own sub-headings
                last = next_page - 1
                break
        return Section(page, last, "bookmarks", " ".join(title.split()))
    return None


def heading_lines(page_text: str) -> list[str]:
    """The page's short lines, plus each joined with the next one (a title may be split in two)."""
    lines = [" ".join(line.split()) for line in page_text.split("\n")]
    lines = [line for line in lines if line]
    result = []
    for i, line in enumerate(lines):
        if len(line) > MAX_HEADING_CHARS:
            continue
        result.append(line)
        if i + 1 < len(lines) and len(line) + len(lines[i + 1]) < MAX_HEADING_CHARS:
            result.append(f"{line} {lines[i + 1]}")
    return result


def has_heading(page_text: str, pattern: re.Pattern) -> str | None:
    """The line that is this title (almost the whole line), or None."""
    for line in heading_lines(page_text):
        match = pattern.search(line)
        if not match:
            continue
        before, after = line[:match.start()], line[match.end():]
        if ALLOWED_EXTRA.fullmatch(before + " " + after):
            return line
        # "Management Discussion and Analysis – Business Overview": a title, a dash, a sub-title.
        if ALLOWED_EXTRA.fullmatch(before) and SUBTITLE.match(after):
            return line
    return None


def _is_contents_page(page_text: str) -> bool:
    titles = MDNA.findall(page_text) + NEXT_SECTION.findall(page_text)
    return len(titles) >= CONTENTS_TITLES


def section_from_headings(pages: list[str], toc: list | None = None, pattern: re.Pattern = MDNA) -> Section | None:
    """pages: the text of each page, in order. toc: the PDF's bookmarks, if any."""
    first = next((i for i, text in enumerate(pages)
                  if has_heading(text, pattern) and not _is_contents_page(text)), None)
    if first is None:
        return None
    last = len(pages) - 1
    later_bookmarks = [page - 1 for _, _, page in (toc or []) if page - 1 > first]
    if later_bookmarks:
        last = min(later_bookmarks) - 1
    for i in range(first + 1, last + 1):
        if has_heading(pages[i], CAUTIONARY):
            last = i
            break
        if has_heading(pages[i], NEXT_SECTION) and not has_heading(pages[i], pattern):
            last = i - 1
            break
    return Section(first + 1, last + 1, "headings", has_heading(pages[first], pattern))


def remove_page_furniture(pages: list[str]) -> list[str]:
    """Drop lines repeated at the top or bottom of many pages: running headers, footers, navigation strips.

    L&T prints a navigation strip ("Report / Management / Discussion and Analysis /
    Financial / Statements") at the bottom of every other page, which looked like an
    MD&A heading on 333 of its 669 pages. A line at the page edges of at least 30% of
    all pages is furniture, not content. The MD&A's own running header appears only
    on its own pages (5-17% of a report in the pilot), so it stays.
    """
    def edge_lines(text: str) -> set[str]:
        lines = [" ".join(line.split()) for line in text.split("\n")]
        lines = [line for line in lines if line]
        return set(lines[:EDGE_LINES] + lines[-EDGE_LINES:])

    counts = Counter(line for page in pages for line in edge_lines(page))
    furniture = {line for line, n in counts.items() if n >= max(3, FURNITURE_SHARE * len(pages))}
    return ["\n".join(line for line in page.split("\n") if " ".join(line.split()) not in furniture)
            for page in pages]


def clean(text: str) -> str:
    """Join words hyphenated across lines ("manage-\\nment" -> "management")."""
    return re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)


@dataclass
class ReportText:
    pdf: Path
    page_count: int
    chars_per_page: float  # median; near 0 means scanned images, which would need OCR
    has_bookmarks: bool
    mdna: Section | None
    mdna_text: str


def find_mdna(pdf_paths: list[Path]) -> ReportText:
    """The MD&A of a report that may come as several PDFs (some zip files hold the report in parts)."""
    result = None
    for path in pdf_paths:
        with pymupdf.open(path) as doc:
            pages = remove_page_furniture([page.get_text() for page in doc])
            toc = doc.get_toc()
        lengths = sorted(len(p) for p in pages)
        section = section_from_bookmarks(toc, len(pages)) or section_from_headings(pages, toc)
        text = clean("\n".join(pages[section.first_page - 1:section.last_page])) if section else ""
        result = ReportText(path, len(pages), lengths[len(lengths) // 2] if lengths else 0.0, bool(toc),
                            section, text)
        if section:
            break
    return result
