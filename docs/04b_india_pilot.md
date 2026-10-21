# India pilot: can FilingLens read Indian annual reports?

**The question:** before building anything big for Indian companies, can we reliably get the **MD&A** (management's own account of the year) out of Indian annual reports, automatically? If yes, scaling up is "more of the same". If not, we find out cheaply.

**The pilot:** 10 large NSE-listed companies × 4 fiscal years (2022-23 to 2025-26) = 40 annual reports.

| Group | Companies | Why |
| --- | --- | --- |
| Listed in India **and** filing a 20-F with the US SEC | Infosys, Wipro, HDFC Bank, ICICI Bank, Dr. Reddy's | Sets up the "same company, two regulators" comparison later |
| Other sectors | Reliance (energy, telecom, retail), TCS (IT), Hindustan Unilever (consumer), Larsen & Toubro (engineering), Bharti Airtel (telecom) | Different industries, different report designs |

**Command:** `python -m filinglens.india.pilot` (results in `data/processed/india/`).

---

## The answer

**Getting the reports: yes.** All 10 companies have all 4 years on NSE. The source is slow (~0.2 MB/s) and flaky, but with completeness checks and resumable downloads every file arrived.

**Finding the MD&A automatically: only partly.** Every extracted section was checked by reading its first page, its last page and the page after. All 40 reports were checked:

| Result | Reports | Companies |
| --- | --- | --- |
| **Correct start and end** | 14 | Infosys (4, via bookmarks), ICICI Bank (4), TCS (4), L&T (2, via bookmarks) |
| **Start right, end approximate** (part of the next section is included) | 12 | HDFC Bank (4), Wipro (3), Hindustan Unilever (2), Bharti Airtel (1), Dr. Reddy's (1), L&T (1) |
| **Wrong start, or doubtful** | 13 | Reliance (4), Dr. Reddy's (3), Bharti Airtel (3), Hindustan Unilever (2), Wipro (1) |
| **Not found** | 1 | L&T (1) |

So simple rules give a clean MD&A for **3 of the 10 companies in every year** (14 of 40 reports, counting L&T's two bookmarked years), and at least the right start for 26 of 40. That is the pilot's main result: it is cheaper to learn this from 40 reports than from 2,500.

**First look at year-on-year change** (only the 3 companies with clean extraction, 9 pairs, far too few to generalise): ICICI Bank barely rewrites its MD&A (0.6–0.8% a year by word counts, as "lazy" as a typical US Risk Factors section), while Infosys and TCS rewrite 4–14% a year. Median 4.9%, against about 2–3% for US MD&As.

**What scaling up would take:** the 40 files came to 851 MB (median 13 MB per report; the largest, L&T's 2024-25 report, is 253 MB). 500 companies × 5 years would be roughly 30–50 GB and, at NSE's speed one file at a time, two to three days of downloading.

## What next (recommendation)

1. **Read each report's own table of contents.** Most reports print one ("Management Discussion and Analysis ... 86") and print page numbers on their pages. Matching the two gives the MD&A's first and last page from the company's own declaration, whatever its design. This is the most promising general fix.
2. **Use an LLM only for the hard cases**, e.g. give it the first lines of each page between two candidate pages and ask where the MD&A ends. Cheap if used only where the rules are unsure.
3. **Measure, don't guess:** write down the correct first and last page for all 40 reports (opening the PDFs is enough, no code) and score every method against that list. This is the same idea as Step 5's test set.
4. **Start the "same company, two regulators" idea with what already works:** Infosys and ICICI Bank have clean MD&As and both file a 20-F with the US SEC.

---

## Where the reports come from: NSE, not MCA

| Source | What it offers | Verdict |
| --- | --- | --- |
| **MCA** (Ministry of Corporate Affairs registry) | Every company's filed financial statements (form AOC-4) | Login and a fee per company, no bulk access, and it's filed *after* the shareholders' meeting, later than the exchange copy |
| **NSE** (National Stock Exchange) | Every listed company's annual report as a PDF, with the exact time it was made public (recent years) | Free, complete, and the first public copy. Used here |
| BSE (Bombay Stock Exchange) | The same reports | A backup source |

Under SEBI's listing rules (Regulation 34), every listed company must send its annual report to the stock exchanges. NSE's website lists them, through the same data service its own web page uses. It isn't an official API: it only answers requests that look like a browser and carry NSE's cookie, so the downloader opens NSE's page first, waits 1.5 seconds between requests, and saves everything so no file is downloaded twice.

**Indian fiscal years** run April to March. As with the US data, a year is named after the year it ends: April 2024 – March 2025 = fiscal 2025.

---

## How it works, in plain words

1. **List** each company's annual reports on NSE and pick, for each year, the **first** version (some are re-filed as "Revised" days later; investors saw the first one first).
2. **Download** the file (a PDF, or a zip holding the PDF for older years) and **check it's complete** before using it.
3. **Find the MD&A**, in this order:
   - **Bookmarks:** many PDFs have a clickable table of contents that says on which page each section starts. Exact when present.
   - **Headings:** otherwise, the first page where "Management Discussion and Analysis" stands as a heading of its own, until the next major section (corporate governance report, sustainability report, auditor's report, financial statements) or a closing "Cautionary statement".
4. **Measure the year-on-year change** of each MD&A with the same word measures as the US study (word counts and word sets, ignoring the company's own name).

---

## What went wrong, and what it taught

### Getting the files (the source is unreliable)

1. **NSE only answers "browsers".** Its report list returns nothing to a plain script. Opening NSE's web page first gives the cookie it checks for; after that, a normal script works.
2. **Downloads were cut off silently.** Wipro's 2024-25 report arrived as 2.3 MB of a ~8 MB file, with no error, and PyMuPDF opened it as 0 pages. Now every file is checked before use: a complete PDF starts with `%PDF-` and ends with an `%%EOF` marker.
3. **Big files broke off mid-transfer.** NSE's server sends about 0.2 MB a second and drops long transfers; L&T's 2024-25 report is **253 MB**. Starting again from zero would never finish. The downloader now **resumes**: it saves what arrived and asks for "the rest, from byte 13,803,520 onwards" (an HTTP Range request, as browsers do). Checked first that NSE supports this (it answers "206 Partial Content").
4. **I almost blamed the source for my own conclusion.** After a break, NSE once answered "there is nothing after 5.4 MB" (HTTP 416) for an incomplete file, and I concluded its copy was damaged. The next attempt sent the rest of the file. One server reply isn't evidence; now a 416 is retried like any other hiccup, and only repeated failure is reported.
5. **One bad file hid a company's other years.** The first version stopped processing a company at its first failed download. Errors are now recorded per report, so a failure shows up as one row in the table instead of a missing company.

### Finding the MD&A (every company does it differently)

6. **Bookmarks rarely help.** Most PDFs have bookmarks, but only **Infosys** bookmarks its MD&A. Others bookmark only big blocks ("Directors Report (168-199)") or files ("infosys-ar-25").
7. **The title takes many shapes**, each found in a real report: split over two lines ("Management Discussion / and Analysis", Wipro); in the middle of a page (HDFC Bank); with a page number ("69 Management Discussion and Analysis", TCS); as a running header with a subtitle ("Management Discussion and Analysis – Business Overview", Reliance); and *mentioned* inside sentences ("…as part of the Management Discussion and Analysis in the Annual Report", Dr. Reddy's), which must not count. Each rule was added for a real case, with a test.
8. **Page furniture looks like headings.** L&T prints a navigation strip ("Report / Management / Discussion and Analysis / Financial / Statements") at the bottom of every other page: 333 of 669 pages seemed to have an MD&A heading. Lines repeated at the top or bottom of 30% or more of all pages are now removed first, a standard step in PDF processing (the same idea as removing repeated footers from the SEC web pages in Step 2).
9. **Some MD&As have no end.** HDFC Bank's MD&A is a chapter inside its Directors' Report, and nothing marks where it stops, so the rest of that report is included. Reliance publishes **spreads** (two printed pages side by side on each PDF page), so "pages" mean different things in different reports.
10. **"Found" isn't "right".** Most of these problems were invisible in the summary table (a page range and a word count look fine either way). They only showed up by reading the first page, the last page and the page after for every report, and by comparing word counts across a company's years (Reliance: 39,000 words one year, 9,000 the next).

### For a returns study later

11. **Publication times are missing for older years.** NSE gives the exact second for recent reports, but only "-" for most 2022-23 and some 2023-24 entries. A study of price reactions needs the date the report became public, so those years would need another source (e.g. the exchange announcement of the annual report or the AGM notice).

---

## Interview questions you can now answer

- **Why not get Indian reports from MCA?** MCA charges per company, has no bulk access, and its copy comes after the shareholders' meeting. The stock exchange copy is free, complete and earlier, which matters for any study of how prices react.
- **Why is India harder than the US?** SEC filings are web pages with a fixed outline (Item 1A, Item 7). Indian annual reports are designed PDFs of 200–600 pages, each company with its own layout, and the MD&A has no fixed title or place.
- **How did you handle an unreliable source?** Check every file before using it, resume broken downloads instead of starting over, report a file that is broken on the server instead of retrying forever, and record failures per report so one bad file doesn't hide the rest.
- **How do you know your extraction is right?** By checking it: for every report, the first and last page of the extracted section and the page after it were read to confirm the section really starts and ends there. "Found something" is not the same as "found the right thing".
- **What would you do next?** See "What next" above.
