from filinglens.text.chunking import chunk_text
from filinglens.text.html_to_text import html_to_text
from filinglens.text.sections import split_sections


# ---------- 2.1 HTML -> text ----------

def test_hidden_xbrl_is_removed_and_spans_stay_on_one_line():
    html = """<html><body>
      <div style="display:none"><ix:header>HIDDEN XBRL DATA</ix:header></div>
      <div><span style="font-weight:700">Item 1A.</span><span>&#160;Risk Factors</span></div>
      <p>Our <span>business</span> is risky.</p>
    </body></html>"""
    assert html_to_text(html) == "Item 1A. Risk Factors\nOur business is risky."


def test_table_rows_become_one_line_with_numbers_joined():
    html = """<table>
      <tr><td><p>iPhone</p></td><td>$</td><td><p>201,183</p></td><td>(2</td><td>)%</td></tr>
    </table>"""
    assert html_to_text(html) == "iPhone | $201,183 | (2)%"


def test_page_numbers_and_repeated_footers_are_removed():
    pages = "".join(f"<p>Text of page {n}.</p><p>Apple Inc. | 2024 Form 10-K | {n}</p><p>{n}</p>" for n in range(6))
    text = html_to_text(f"<html><body>{pages}</body></html>")
    assert "Form 10-K" not in text
    assert text.split("\n") == [f"Text of page {n}." for n in range(6)]


# ---------- 2.2 sections ----------

def body(words: int) -> str:
    return " ".join(["word"] * words)


TABLE_OF_CONTENTS = "Item 1. Business\nItem 1A. Risk Factors\nItem 7. Management's Discussion and Analysis\n"


def test_real_headings_win_over_table_of_contents():
    text = (TABLE_OF_CONTENTS
            + f"Item 1. Business\n{body(300)}\n"
            + f"Item 1A. Risk Factors\n{body(500)}\n"
            + f"Item 7. Management's Discussion and Analysis\n{body(400)}")
    sections = {s.item: s for s in split_sections(text)}
    assert sections["1A"].text.startswith("Item 1A. Risk Factors\nword")
    # heading words + body words: "Item 1. Business" = 3, "Item 1A. Risk Factors" = 4, ...
    assert [s.n_words for s in (sections["1"], sections["1A"], sections["7"])] == [3 + 300, 4 + 500, 6 + 400]


def test_running_page_header_without_title_is_not_a_heading():
    text = f"ITEM 1A. RISK FACTORS\n{body(200)}\nItem 1A\n{body(200)}\nITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS\n{body(50)}"
    sections = {s.item: s for s in split_sections(text)}
    assert sections["1A"].n_words == 4 + 200 + 2 + 200  # the "Item 1A" page header stays inside 1A


def test_title_on_the_next_line_counts():
    sections = split_sections(f"Item 1A.\nRisk Factors\n{body(100)}")
    assert [s.item for s in sections] == ["1A"]


def test_title_on_the_next_line_counts_after_other_words():
    # Bank of America: a page header shares the heading line, the title is on the next line.
    # Until this was fixed its MD&A was only a 38-word table-of-contents entry.
    text = (TABLE_OF_CONTENTS + "Item 7. Bank of America Corporation and Subsidiaries\n"
            f"Management's Discussion and Analysis of Financial Condition\n{body(500)}")
    sections = {s.item: s for s in split_sections(text)}
    assert sections["7"].n_words > 500


def test_combined_items_heading():
    sections = split_sections(f"Items 1 and 2. Business and Properties\n{body(100)}")
    assert [s.item for s in sections] == ["1"]


def test_long_final_section_is_an_attached_annual_report():
    text = f"Item 7. Management's Discussion and Analysis\nSee pages 52-167.\nItem 15. Exhibits\n{body(31_000)}"
    assert [(s.item, s.title) for s in split_sections(text)] == [
        ("7", "Management's Discussion and Analysis"), ("AR", "Attached annual report")]


def test_long_text_before_first_item_is_the_report_body():
    text = f"Our Business\n{body(6_000)}\nForm 10-K Cross-Reference Index\nItem 1A. Risk Factors Pages 31-46"
    assert [s.item for s in split_sections(text)] == ["BODY", "1A"]


def test_text_without_any_items_is_kept_whole():
    assert [s.item for s in split_sections(body(100))] == ["BODY"]


# ---------- 2.3 chunks ----------

def numbered_paragraphs(n_paragraphs: int, words_each: int) -> str:
    return "\n".join(" ".join(f"p{p}w{w}" for w in range(words_each)) for p in range(n_paragraphs))


def test_chunks_respect_size_limit_and_overlap():
    chunks = chunk_text(numbered_paragraphs(20, 40), max_words=100, overlap_words=40)
    assert all(len(c.split()) <= 100 for c in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        assert current.split("\n")[0] == previous.split("\n")[-1]  # last paragraph carried over


def test_every_paragraph_ends_up_in_a_chunk():
    text = numbered_paragraphs(20, 40)
    chunked = "\n".join(chunk_text(text, max_words=100, overlap_words=40))
    assert all(paragraph in chunked for paragraph in text.split("\n"))


def test_very_long_paragraph_is_split_by_sentence():
    paragraph = " ".join(f"This is sentence number {i}." for i in range(100))  # 500 words
    chunks = chunk_text(paragraph, max_words=100, overlap_words=20)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 100 for c in chunks)
    assert all(c.rstrip().endswith(".") for c in chunks)  # cut between sentences, not mid-sentence
