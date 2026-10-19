"""Step 2.1 - turn a 10-K web page (HTML) into clean plain text.

A 10-K's HTML is built for printing, not for programs:
  - Hidden XBRL data (the tagged numbers from Step 1) sits inside <ix:header>
    and display:none blocks. It isn't meant to be read, so we delete it.
  - Text is split into many small <span> tags just to change fonts. Splitting
    on every tag would break sentences, so new lines start only at "block"
    tags like <div> and <p>.
  - Tables become one line per row, cells separated by " | ", because MD&A
    tables often hold useful numbers.
  - Page footers ("Apple Inc. | 2024 Form 10-K | 22"), running headers and
    navigation links repeat on every page; we drop them.
"""

import re
import warnings
from collections import Counter

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# 10-Ks start with an XML declaration, but the lenient HTML parser reads them best.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

BLOCK_TAGS = ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"]

PAGE_NUMBER = re.compile(r"^(page\s*)?[\divxlc]{1,4}$", re.IGNORECASE)  # "12", "Page 12", "iv"
NAVIGATION = re.compile(r"^(table of contents|index)$", re.IGNORECASE)
KEEP_EVEN_IF_REPEATED = re.compile(r"^item\s|million|thousand|billion|except", re.IGNORECASE)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # 1. Delete things that aren't meant to be read.
    for tag in soup(["script", "style", "head", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.IGNORECASE)):
        tag.decompose()

    # 2. Replace each table row with one line of text: "cell | cell | cell".
    for row in soup.find_all("tr"):
        if row.parent is None:  # inside a table row we already replaced
            continue
        cells = [clean_line(cell.get_text(" ")) for cell in row.find_all(["td", "th"])]
        row.replace_with("\n" + " | ".join(c for c in cells if c) + "\n")

    # 3. Mark where the other lines end, then extract all the text at once.
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(BLOCK_TAGS):
        block.append("\n")
    lines = [clean_line(line) for line in soup.get_text().split("\n")]
    lines = [line for line in lines if line and not PAGE_NUMBER.match(line) and not NAVIGATION.match(line)]
    return "\n".join(remove_repeated_lines(lines))


def clean_line(line: str) -> str:
    line = line.replace("\xa0", " ")            # non-breaking spaces -> normal spaces
    line = re.sub(r"\s+", " ", line)             # collapse runs of spaces
    line = re.sub(r"(\s*\|\s*)+", " | ", line)   # empty table cells: "a |  | | b" -> "a | b"
    line = line.strip(" |")
    # Join currency signs, brackets and percent signs to their numbers:
    # "$ | 383,285" -> "$383,285", "(3,040 | )" -> "(3,040)", "2 | %" -> "2%"
    line = re.sub(r"\$ (\| )?(?=[\d(])", "$", line)
    line = re.sub(r" (\| )?\)", ")", line)
    line = re.sub(r" (\| )?%", "%", line)
    return line


def remove_repeated_lines(lines: list[str]) -> list[str]:
    """Drop page headers/footers: short lines that repeat on many pages.

    Digits are ignored when comparing, so "Form 10-K | 22" and "Form 10-K | 23"
    count as the same line.
    """
    def pattern(line: str) -> str:
        return re.sub(r"\d", "#", line)

    counts = Counter(pattern(line) for line in lines)

    def is_noise(line: str) -> bool:
        if len(line) > 80 or KEEP_EVEN_IF_REPEATED.search(line):
            return False
        n = counts[pattern(line)]
        if "10-k" in line.lower():                 # footers that name the form
            return n >= 5
        return n >= 10 and "|" not in line         # other running headers (not table rows)

    return [line for line in lines if not is_noise(line)]
