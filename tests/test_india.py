import zipfile

import pandas as pd
import requests

from filinglens.india.nse import NseClient, choose_reports, is_complete, unzip_pdfs
from filinglens.india.report_text import clean, section_from_bookmarks, section_from_headings


# ---------- choosing the report for each year ----------

def nse_row(from_year, to_year, kind, when, report_id):
    """One entry in the same shape as NSE's annual-report list."""
    return {"companyName": "Infosys Limited", "fromYr": str(from_year), "toYr": str(to_year),
            "submission_type": kind, "disseminationDateTime": when,
            "fileName": f"https://nsearchives.nseindia.com/annual_reports/AR_{report_id}_INFY_{from_year}_{to_year}.pdf"}


def test_first_submission_wins_over_a_revised_one():
    rows = [nse_row(2025, 2026, "Revised", "30-MAY-2026 20:04:14", 29313),
            nse_row(2025, 2026, "New", "29-MAY-2026 20:09:27", 29297),
            nse_row(2024, 2025, "New", "02-JUN-2025 15:39:48", 26481)]
    chosen = choose_reports(rows).set_index("fiscal_year")
    assert chosen.loc[2026, "report_id"] == 29297  # what investors saw first
    assert chosen.loc[2026, "published_at"] == pd.Timestamp("2026-05-29 20:09:27")
    assert list(chosen.index) == [2025, 2026]  # April 2024 - March 2025 is fiscal 2025


def test_missing_publication_time_stays_missing():
    chosen = choose_reports([nse_row(2022, 2023, "-", "-", 21960)])
    assert pd.isna(chosen.loc[0, "published_at"])  # never guessed


def test_unzip_keeps_only_pdfs_inside_the_folder(tmp_path):
    archive = tmp_path / "report.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("Annual Report 2023.pdf", b"%PDF-1.7 part 1")
        z.writestr("../../escape.pdf", b"%PDF-1.7 sneaky path")
        z.writestr("notes.txt", b"not a pdf")
    paths = unzip_pdfs(archive, tmp_path / "report")
    assert [p.name for p in paths] == ["Annual Report 2023.pdf", "escape.pdf"]
    assert all(p.parent == tmp_path / "report" for p in paths)  # nothing written outside the folder
    assert not (tmp_path.parent / "escape.pdf").exists()


def test_cut_off_downloads_are_detected():
    # Found in the pilot: NSE's server stopped sending Wipro's 2024-25 report at 2.3 of ~9 MB, with no error.
    whole = b"%PDF-1.7\n" + b"x" * 5000 + b"\n%%EOF\n"
    assert is_complete(whole, ".pdf")
    assert not is_complete(whole[:3000], ".pdf")
    assert not is_complete(b"<html>Access denied</html>", ".pdf")


class FakeResponse:
    def __init__(self, chunks, status=200, break_after=False):
        self.chunks, self.status_code, self.break_after = chunks, status, break_after

    def iter_content(self, chunk_size):
        yield from self.chunks
        if self.break_after:
            raise requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_a_broken_download_resumes_where_it_stopped(tmp_path):
    # Found in the pilot: NSE dropped L&T's 253 MB report after 13.8 MB.
    whole = b"%PDF-1.7 " + b"x" * 100 + b" %%EOF"
    requests_seen = []

    def fake_get(url, stream=False, headers=None):
        requests_seen.append((headers or {}).get("Range"))
        if len(requests_seen) == 1:
            return FakeResponse([whole[:40]], break_after=True)  # first try: 40 bytes, then the line drops
        return FakeResponse([whole[40:]], status=206)         # second try: only the rest

    client = NseClient(cache_dir=tmp_path)
    client._get = fake_get
    [path] = client.download("https://nsearchives.nseindia.com/annual_reports/AR_1_X_2024_2025.pdf", "X")
    assert path.read_bytes() == whole
    assert requests_seen == [None, "bytes=40-"]  # asked only for the missing part


# ---------- finding the MD&A ----------

TOC = [  # PyMuPDF's bookmark list: [level, title, first page]
    [1, "Statutory Reports", 30],
    [2, "Board's report", 31],
    [2, "Management’s discussion and analysis", 72],
    [3, "Industry overview", 73],  # the MD&A's own sub-heading: not where it ends
    [2, "Corporate governance report", 83],
    [1, "Financial statements", 182],
]


def test_mdna_from_bookmarks_ends_before_the_next_section():
    section = section_from_bookmarks(TOC, page_count=370)
    assert (section.first_page, section.last_page, section.method) == (72, 82, "bookmarks")


def test_no_mdna_bookmark_gives_none():
    assert section_from_bookmarks([[1, "Board's report", 5]], page_count=50) is None


def test_mdna_from_headings_skips_the_contents_page():
    pages = [
        "Contents\nBoard's report 10\nManagement Discussion and Analysis 20\nCorporate Governance Report 30\n"
        "Business Responsibility and Sustainability Report 40\nIndependent Auditor's Report 50",
        "Board's Report\nDear members, ...",
        "Management Discussion and Analysis\nThe economy grew ...",
        "Management Discussion and Analysis\nRisks and concerns ...",  # running header on every MD&A page
        "Report on Corporate Governance\nOur philosophy ...",
    ]
    section = section_from_headings(pages)
    assert (section.first_page, section.last_page, section.method) == (3, 4, "headings")


def test_words_split_across_lines_are_joined():
    assert clean("risk manage-\nment and cyber-\nSecurity") == "risk management and cyber-\nSecurity"


def test_title_split_over_two_lines_in_the_middle_of_a_page():
    # Wipro: "Management Discussion" / "and Analysis", below other text on the page.
    pages = ["Our values ...", "Integrated report\nSome intro text here.\nManagement Discussion\nand Analysis\nThe year ..."]
    assert section_from_headings(pages).first_page == 2


def test_a_sentence_mentioning_the_mdna_is_not_its_heading():
    pages = ["Details are in the Management Discussion and Analysis section on page 40.",
             "MANAGEMENT DISCUSSION AND ANALYSIS\nThe economy ..."]
    assert section_from_headings(pages).first_page == 2


def test_cautionary_statement_closes_the_mdna():
    pages = ["Management Discussion and Analysis\n...", "Outlook ...", "Cautionary Statement\nStatements in this ...",
             "Dividend\nThe Board recommends ..."]  # the rest of the Directors' report: not MD&A
    assert (section_from_headings(pages).first_page, section_from_headings(pages).last_page) == (1, 3)


def test_next_bookmark_bounds_an_mdna_without_its_own_bookmark():
    # HDFC Bank: the MD&A is inside the bookmarked Directors' report (pages 2-4); the next bookmark is page 5.
    pages = ["Contents", "Directors' Report\n...", "Management Discussion and Analysis\n...", "...", "Annexure 1\n...",
             "..."]
    toc = [[1, "Directors Report", 2], [1, "Directors Report - Annexure", 5]]
    section = section_from_headings(pages, toc)
    assert (section.first_page, section.last_page) == (3, 4)


def test_a_title_wrapped_inside_a_sentence_is_not_a_heading():
    # Dr. Reddy's: "...as part of the" / "Management Discussion and Analysis in" / "the Annual Report".
    pages = ["forms part of the\nManagement Discussion and Analysis in\nthe Annual Report.",
             "4. Management Discussion and Analysis Report\nIndustry structure ..."]
    assert section_from_headings(pages).first_page == 2


def test_running_header_with_a_subtitle_is_a_heading():
    # Reliance: "Management Discussion and Analysis – Business Overview" at the top of each MD&A page.
    pages = ["Chairman's message ...", "Management Discussion and Analysis \u2013 Business Overview\nJio ..."]
    assert section_from_headings(pages).first_page == 2


def test_an_incomplete_file_is_retried_then_reported_not_retried_forever(tmp_path, monkeypatch):
    # Found in the pilot: NSE once answered "416, nothing after 5.4 MB" for an incomplete file,
    # then sent more data on the next try. So a 416 is retried, and after the last attempt it's reported.
    monkeypatch.setattr("filinglens.india.nse.time.sleep", lambda seconds: None)
    partial = b"%PDF-1.7 " + b"x" * 100  # no end marker
    calls = []

    def fake_get(url, stream=False, headers=None):
        calls.append((headers or {}).get("Range"))
        if len(calls) == 1:
            return FakeResponse([partial])
        response = requests.Response()
        response.status_code = 416
        raise requests.HTTPError(response=response)

    client = NseClient(cache_dir=tmp_path)
    client._get = fake_get
    try:
        client.download("https://nsearchives.nseindia.com/annual_reports/AR_2_X_2024_2025.pdf", "X")
        raise AssertionError("expected an error")
    except IOError as exc:
        assert "still incomplete" in str(exc)
    assert len(calls) == 8 and calls[1] == f"bytes={len(partial)}-"
